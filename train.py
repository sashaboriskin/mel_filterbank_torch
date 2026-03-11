import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchaudio.datasets import SPEECHCOMMANDS

from utils import count_parameters, count_flops
from melbanks import LogMelFilterBanks

LABELS = ["no", "yes"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
TARGET_SR = 16000
TARGET_LENGTH = 16000  # 1 second at 16 kHz


class YesNoSpeechCommands(SPEECHCOMMANDS):
    """SPEECHCOMMANDS filtered to YES / NO classes only."""

    def __init__(self, subset: str, data_dir: str = "./data"):
        super().__init__(root=data_dir, subset=subset, download=True)
        # _walker contains full file paths; parent dir name == label
        self._walker = [
            w for w in self._walker if os.path.basename(os.path.dirname(w)) in LABELS
        ]


def make_collate_fn(mel_transform: LogMelFilterBanks):
    """Return a collate function that extracts LogMel features on the fly."""

    def collate_fn(batch):
        features, labels = [], []
        for waveform, _sr, label, *_ in batch:
            # pad / truncate to exactly 1 second
            if waveform.shape[1] < TARGET_LENGTH:
                waveform = F.pad(waveform, (0, TARGET_LENGTH - waveform.shape[1]))
            else:
                waveform = waveform[:, :TARGET_LENGTH]
            with torch.no_grad():
                feat = mel_transform(waveform).squeeze(0)  # (n_mels, n_frames)
            features.append(feat)
            labels.append(LABEL_TO_IDX[label])
        return torch.stack(features), torch.tensor(labels)

    return collate_fn


class ChannelShuffle(nn.Module):
    """ShuffleNet channel shuffle: mix channels across groups."""

    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        # x: (batch, channels, length)
        b, c, l = x.shape
        x = x.view(b, self.groups, c // self.groups, l)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, l)


class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → [ChannelShuffle] → pool."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        groups: int = 1,
        shuffle: bool = False,
        pool: nn.Module = nn.MaxPool1d(2),
    ):
        super().__init__()
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        ]
        if shuffle and groups > 1:
            layers.append(ChannelShuffle(groups))
        layers.append(pool)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        groups: int = 1,
        shuffle: bool = False,
        num_classes: int = 2,
    ):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(n_mels, 64, groups, shuffle),
            ConvBlock(64, 64, groups, shuffle),
            ConvBlock(64, 32, groups, shuffle, pool=nn.AdaptiveAvgPool1d(1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (batch, n_mels, n_frames)
        x = self.features(x)  # (batch, 32, 1)
        x = x.squeeze(-1)  # (batch, 32)
        return self.classifier(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    n_batches = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
    return running_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        preds = model(features).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_mels", type=int, default=80, choices=[20, 40, 80])
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="enable channel shuffle (ShuffleNet-style)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shuffle_tag = "_shuffle" if args.shuffle else ""
    run_name = f"n_mels_{args.n_mels}_groups_{args.groups}{shuffle_tag}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    mel_transform = LogMelFilterBanks(n_mels=args.n_mels)
    collate_fn = make_collate_fn(mel_transform)

    train_set = YesNoSpeechCommands("training", args.data_dir)
    val_set = YesNoSpeechCommands("validation", args.data_dir)
    test_set = YesNoSpeechCommands("testing", args.data_dir)

    loader_kw = dict(
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kw)

    model = CNN(n_mels=args.n_mels, groups=args.groups, shuffle=args.shuffle).to(device)
    n_params = count_parameters(model)

    # compute number of frames for 1-second signal: (16000 + 400 - 400)/160 + 1 = 101
    n_frames = (
        TARGET_LENGTH + mel_transform.n_fft - mel_transform.n_fft
    ) // mel_transform.hop_length + 1
    dummy = torch.randn(1, args.n_mels, n_frames, device=device)
    flops = count_flops(model, dummy)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        epoch_time = time.time() - t0

        val_acc = evaluate(model, val_loader, device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("train/epoch_time_sec", epoch_time, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{run_name}_best.pt")

    model.load_state_dict(
        torch.load(f"checkpoints/{run_name}_best.pt", weights_only=True)
    )
    test_acc = evaluate(model, test_loader, device)

    writer.add_hparams(
        {
            "n_mels": args.n_mels,
            "groups": args.groups,
            "shuffle": args.shuffle,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
        {
            "hparam/test_accuracy": test_acc,
            "hparam/params": n_params,
            "hparam/flops": flops or 0,
        },
    )
    writer.close()


if __name__ == "__main__":
    main()
