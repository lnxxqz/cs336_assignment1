import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from einops import einsum, rearrange

class rmsnorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def set_weights(self, weights):
        self.g.data.copy_(weights)

    def forward(self, x: Tensor)->Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        sqmean = x.square().mean(dim=-1,keepdim=True)
        RMS = (sqmean + self.eps).sqrt()
        result = x / RMS * self.g
        return result.to(in_dtype)

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    RMSnorm = rmsnorm(d_model, eps)
    RMSnorm.set_weights(weights)
    return RMSnorm.forward(in_features)