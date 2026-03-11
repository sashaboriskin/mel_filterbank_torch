from thop import profile
from torch import nn
import torch


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: nn.Module, dummy_input: torch.Tensor):
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return int(flops)
