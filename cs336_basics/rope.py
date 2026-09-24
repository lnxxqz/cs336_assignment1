import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math

class rope(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        i = torch.arange(max_seq_len, device=device)       # (S,)
        k = torch.arange(d_k // 2, device=device)          # (P,)  P = d_k/2
        omega = theta ** (-2 * k / d_k)                    # (P,)  每种 pair 一个频率
        angles = i[:, None] * omega[None, :]
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, inTensor, token_position)->Tensor:
        possin = self.sin[token_position]
        poscos = self.cos[token_position]
        x_even = inTensor[...,0::2]
        x_odd  = inTensor[...,1::2]
        x = x_even*poscos - x_odd*possin
        y = x_even*possin + x_odd*poscos
        result = torch.empty_like(inTensor)
        result[..., 0::2] = x
        result[..., 1::2] = y
        return result
        

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:

    RoPE = rope(theta, d_k,max_seq_len)

    return RoPE.forward(in_query_or_key,token_positions)
