import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from cs336_basics.multihead_self_attention_with_rope import multihead_self_attention_with_rope as mth
from cs336_basics.rmsnorm import rmsnorm as rms
from cs336_basics.swiglu import swiglu as swiglu


class transformer_block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rms1 = rms(d_model, device=device,dtype=dtype)
        self.rms2 = rms(d_model, device=device,dtype=dtype)
        self.mth = mth(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
        self.swiglu = swiglu(d_model, d_ff, device=device,dtype=dtype)

    def set_weights(self, weights: dict[str, Tensor]):
        self.rms1.set_weights(weights['ln1.weight'])
        self.mth.set_weights(weights['attn.q_proj.weight'],weights['attn.k_proj.weight'],weights['attn.v_proj.weight'],weights['attn.output_proj.weight'])
        self.rms2.set_weights(weights['ln2.weight'])
        self.swiglu.set_weights(weights['ffn.w1.weight'],weights['ffn.w2.weight'],weights['ffn.w3.weight'])

    def forward(self, x:Tensor):
        seq = x.shape[-2]
        token_positions = torch.arange(seq, device=x.device)
        y = x + self.mth.forward(self.rms1.forward(x), token_positions)
        z = y + self.swiglu.forward(self.rms2.forward(y))
        return z



def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    block = transformer_block(d_model, num_heads, d_ff, max_seq_len, theta)
    block.set_weights(weights)
    return block.forward(in_features)