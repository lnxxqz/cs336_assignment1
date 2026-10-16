import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from collections.abc import Iterable
from einops import einsum, rearrange

@torch.no_grad()
def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return

    total_norm = torch.sqrt(
        sum(p.grad.float().square().sum() for p in parameters)
    )

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in parameters:
            p.grad.mul_(scale)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    return gradient_clipping(parameters, max_l2_norm)