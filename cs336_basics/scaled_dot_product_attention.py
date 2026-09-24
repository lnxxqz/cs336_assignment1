import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from einops import einsum, rearrange
import cs336_basics.softmax as sft

class scaled_dot_product_attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        sf = sft.softmax(dim=-1)
        QK = einsum(Q, K, "... q d_k, ... k d_k -> ... q k")
        QK = QK/math.sqrt(Q.shape[-1])
        if mask is not None:
            QK = QK.masked_fill(~mask,-1e20)
        QK = sf.forward(QK)
        result = einsum(QK, V ,"... q k, ... k d_v ->... q d_v")
        return result

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    T = scaled_dot_product_attention()
    return T.forward(Q,K,V,mask)