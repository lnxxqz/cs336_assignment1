import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from cs336_basics.embedding import Embedding as embedding
from cs336_basics.transformer_block import transformer_block as transformer_block
from cs336_basics.linear import Linear as linear
from cs336_basics.rmsnorm import rmsnorm as rms

class transformer(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.embedding = embedding(vocab_size, d_model,device=device, dtype=dtype)
        self.layers = nn.ModuleList([
        transformer_block(d_model, num_heads, d_ff, context_length, rope_theta)
        for _ in range(num_layers)])
        self.final_rms = rms(d_model=d_model, device=device, dtype=dtype)
        self.LM_head = linear(d_model, vocab_size)

    def set_weights(self, w):
        self.embedding.set_weights(w['token_embeddings.weight'])
        self.final_rms.set_weights(w['ln_final.weight'])
        self.LM_head.set_weights(w['lm_head.weight'])
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}."
            layer_weights = {
                key[len(prefix):]: value
                for key, value in w.items()
                if key.startswith(prefix)
            }
            layer.set_weights(layer_weights)
            
    def forward(self, x):
        x = self.embedding.forward(x)
        for i in self.layers:
            x = i.forward(x)
        x = self.final_rms.forward(x)
        x = self.LM_head.forward(x)
        return x

        
def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    block = transformer(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    block.set_weights(weights)
    return block.forward(in_indices)