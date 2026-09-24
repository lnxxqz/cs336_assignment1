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
    def forward(self,x:Float[Tensor, " ..."]):
        mx = x.amax(dim=self.dim, keepdim=True)
        x = (x-mx).exp()
        return x / x.sum(dim=self.dim, keepdim=True)

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    sf = softmax(dim)
    return sf(in_features)