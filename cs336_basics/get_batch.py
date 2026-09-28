import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from collections.abc import Iterable
from einops import einsum, rearrange

def get_batch(dataset, batch_size, context_length, device):
    


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch(dataset, batch_size, context_length, device)