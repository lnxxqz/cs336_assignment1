import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from einops import einsum, rearrange
from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.rope import rope

class multihead_self_attention_with_rope(nn.Module):
    def __init__(self, d_model, num_heads, theta, max_seq_len, device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.WQ = Linear(d_model,d_model,device,dtype)
        self.WK = Linear(d_model,d_model,device,dtype)
        self.WV = Linear(d_model,d_model,device,dtype)
        self.WO = Linear(d_model,d_model,device,dtype)
        self.rope = rope(theta, d_model//num_heads, max_seq_len)

    def set_weights(self, WQ, WK, WV, WO):
        self.WQ.set_weights(WQ)
        self.WK.set_weights(WK)
        self.WV.set_weights(WV)
        self.WO.set_weights(WO)

    def forward(self, in_features, token_positions):
        Q = self.WQ.forward(in_features)
        K = self.WK.forward(in_features)
        V = self.WV.forward(in_features)
        Q = rearrange(Q, "... seq (head d) -> ... head seq d", head=self.num_heads)
        K = rearrange(K, "... seq (head d) -> ... head seq d", head=self.num_heads)
        V = rearrange(V, "... seq (head d) -> ... head seq d", head=self.num_heads)
        Q = self.rope.forward(Q,token_positions)
        K = self.rope.forward(K,token_positions)
        attention = scaled_dot_product_attention()
        idx = torch.arange(in_features.shape[-2], device=Q.device)
        mask = idx[None, :] <= idx[:, None]
        T = attention.forward(Q,K,V,mask)
        T = rearrange(T, "... head seq d -> ... seq (head d)")
        result = self.WO.forward(T)
        return result 

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:

    f = multihead_self_attention_with_rope(d_model, num_heads, theta, max_seq_len)
    f.set_weights(q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight)
    return f.forward(in_features, token_positions)