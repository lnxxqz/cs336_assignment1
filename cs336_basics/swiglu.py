import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math

from cs336_basics.linear import Linear

class swiglu(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def set_weights(self, w1, w2, w3):
        self.w1.set_weights(w1)
        self.w2.set_weights(w2)
        self.w3.set_weights(w3)

    def forward(self, x):
        u = self.w1.forward(x)
        v = self.w3.forward(x)
        silu = u*torch.sigmoid(u)
        r = silu*v
        return self.w2.forward(r)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:

    Swi = swiglu(d_model, d_ff)
    Swi.set_weights(w1_weight,w2_weight,w3_weight)
    return Swi.forward(in_features)