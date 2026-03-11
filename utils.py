# from thop import profile
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: torch.nn.Module, dummy: torch.Tensor) -> int:    
    model.eval()
    with FlopCounterMode(display=False) as flop_counter:
        model(dummy)
    return flop_counter.get_total_flops()