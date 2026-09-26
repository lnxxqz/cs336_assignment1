import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from einops import einsum, rearrange

class cross_entropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:Tensor,targets:Tensor):
        mx = x.amax(dim=-1, keepdim=True)
        x = x - mx
        ex = x.exp()
        sum = ex.sum(dim=-1)
        logsum = sum.log()
        t = x[torch.arange(x.shape[0], device=x.device), targets]
        re = logsum - t
        return re.mean()

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    ce = cross_entropy()
    return ce.forward(inputs, targets)