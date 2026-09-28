import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from collections.abc import Iterable
from einops import einsum, rearrange

def gradient_clipping(parameters, max_l2_norm, eps = 1e-6):
    g = 0
    for p in parameters:
        if p.grad is not None:
            g += (p.grad**2).sum()
    g = g.sqrt()
    if g<=max_l2_norm:
        return
    g = max_l2_norm / (g+eps)
    for p in parameters:
        if p.grad is not None:
            p.grad *= g


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    return gradient_clipping(parameters, max_l2_norm)