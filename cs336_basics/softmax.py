import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math

class softmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(x:Float[Tensor, " ..."]):
        ex = x.exp()
        return ex / sum(ex)

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    sf = softmax(dim)