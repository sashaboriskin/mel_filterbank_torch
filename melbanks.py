from typing import Optional

import torch
from torch import nn
import torchaudio
from torchaudio import functional as F


class LogMelFilterBanks(nn.Module):
    def __init__(
            self,
            n_fft: int = 400,
            samplerate: int = 16000,
            hop_length: int = 160,
            n_mels: int = 80,
            pad_mode: str = 'reflect',
            power: float = 2.0,
            normalize_stft: bool = False,
            onesided: bool = True,
            center: bool = True,
            return_complex: bool = True,
            f_min_hz: float = 0.0,
            f_max_hz: Optional[float] = None,
            norm_mel: Optional[str] = None,
            mel_scale: str = 'htk'
        ):
        super(LogMelFilterBanks, self).__init__()
        # general params
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.window_length = n_fft
        self.window = torch.hann_window(self.window_length)
        self.eps = 1e-6
        # stft params
        self.hop_length = hop_length
        self.center = center
        self.return_complex = return_complex
        self.onesided = onesided
        self.normalize_stft = normalize_stft
        self.pad_mode = pad_mode
        self.power = power
        # mel fbanks params
        self.n_mels = n_mels
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz if f_max_hz is not None else samplerate/2
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale
        self.mel_fbanks = self._init_melscale_fbanks() # [n_freqs, n_mels]

    def _init_melscale_fbanks(self):
        return F.melscale_fbanks(
            n_freqs=int(self.n_fft/2 + 1), 
            f_min=self.f_min_hz,
            f_max=self.f_max_hz, 
            n_mels=self.n_mels,
            sample_rate=self.samplerate,
            norm=self.norm_mel,
            mel_scale=self.mel_scale
        )

    def spectrogram(self, x):
        stft = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,
            center=self.center, 
            pad_mode=self.pad_mode,
            normalized=self.normalize_stft,
            onesided=self.onesided,
            return_complex=self.return_complex,
        )
        magnitude = torch.abs(stft)
        spec = torch.pow(magnitude, exponent=2)
        return spec

    def forward(self, x):
        """
        Args:
            x (Torch.Tensor): Tensor of audio of dimension (batch, time), audiosignal
        Returns:
            Torch.Tensor: Tensor of log mel filterbanks of dimension (batch, n_mels, n_frames),
                where n_frames is a function of the window_length, hop_length and length of audio
        """
        # x: (batch, time)
        x = x.squeeze(dim=0) # (time)
        spec = self.spectrogram(x) # (n_freqs, n_frames)
        mel = spec.T @ self.mel_fbanks # (n_freqs, n_mels)
        mel = mel.T # (n_mels, n_frames)
        return torch.log(mel + self.eps).unsqueeze(dim=0) # (batch, n_mels, n_frames)


if __name__ == "__main__":
    signal, sr = torchaudio.load("test.wav")
    logmelbanks = LogMelFilterBanks()(signal)
    print(logmelbanks.shape)

    melspec = torchaudio.transforms.MelSpectrogram(
        hop_length=160,
        n_mels=80
    )(signal)

    assert torch.log(melspec + 1e-6).shape == logmelbanks.shape
    assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)
