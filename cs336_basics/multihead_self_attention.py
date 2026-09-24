import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from einops import einsum, rearrange
from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention

class multihead_self_attention(nn.Module):
    def __init__(self, d_model, num_heads,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.WQ = Linear(d_model,d_model,device,dtype)
        self.WK = Linear(d_model,d_model,device,dtype)
        self.WV = Linear(d_model,d_model,device,dtype)
        self.WO = Linear(d_model,d_model,device,dtype)

    def set_weights(self, WQ, WK, WV, WO):
        self.WQ.set_weights(WQ)
        self.WK.set_weights(WK)
        self.WV.set_weights(WV)
        self.WO.set_weights(WO)

    def forward(self, in_features):
        Q = self.WQ.forward(in_features)
        K = self.WK.forward(in_features)
        V = self.WV.forward(in_features)
        Q = rearrange(Q, "... seq (head d) -> ... head seq d", head=self.num_heads)
        K = rearrange(K, "... seq (head d) -> ... head seq d", head=self.num_heads)
        V = rearrange(V, "... seq (head d) -> ... head seq d", head=self.num_heads)
        attention = scaled_dot_product_attention()
        idx = torch.arange(in_features.shape[-2], device=Q.device)
        mask = idx[None, :] <= idx[:, None]
        T = attention.forward(Q,K,V,mask)
        T = rearrange(T, "... head seq d -> ... seq (head d)")
        result = self.WO.forward(T)
        return result 

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:

    f = multihead_self_attention(d_model, num_heads)
    f.set_weights(q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight)
    return f.forward(in_features)