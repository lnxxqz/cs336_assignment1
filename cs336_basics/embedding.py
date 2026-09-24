import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.W = nn.Parameter(torch.empty(vocab_size, d_model, device=device, dtype=dtype))
        sigma = math.sqrt(2/(vocab_size+d_model))
        torch.nn.init.trunc_normal_(self.W,mean=0.0,std=sigma,a=-3*sigma,b=3*sigma,)
    def set_weights(self, weights):
        self.W.data.copy_(weights)
        
    def forward(self, token_ids) -> Tensor:
        return self.W[token_ids]

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    Embedding_v = Embedding(vocab_size=vocab_size, d_model=d_model)
    Embedding_v.set_weights(weights)
    return Embedding_v.forward(token_ids)