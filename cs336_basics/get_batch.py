import numpy.typing as npt
import numpy as np
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from collections.abc import Iterable
from einops import einsum, rearrange

def get_batch(dataset, batch_size, context_length, device):
    starts = np.random.randint(0, len(dataset) - context_length, batch_size)
    rows = [dataset[i : i+context_length] for i in starts]
    inp = torch.tensor(np.stack(rows), dtype=torch.long, device=device)
    rows = [dataset[i+1 : i+1+context_length] for i in starts]
    oup = torch.tensor(np.stack(rows), dtype=torch.long, device=device)
    return (inp, oup)

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch(dataset, batch_size, context_length, device)