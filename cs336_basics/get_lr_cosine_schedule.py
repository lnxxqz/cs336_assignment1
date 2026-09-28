import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from einops import einsum, rearrange

def get_lr_cosine_schedule(it, mx, mn, warmup, cosine_cycle):
    if it<warmup:
        return it / warmup * mx
    if it<=cosine_cycle:
        return mn + 0.5 * (1+ math.cos((it - warmup)/(cosine_cycle - warmup) * math.pi)) * (mx - mn)
    return mn

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    return get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)