import numpy.typing as npt
import torch.nn as nn
import torch
import numpy as np
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from collections.abc import Iterable
from einops import einsum, rearrange
import os
from typing import IO, Any, BinaryIO
import pickle
import softmax

def decoding(model, ini_context, tokeniz, context_length, max_length, tau=1, top_p=1):
    ids = tokeniz.encode(ini_context)
    end_id = tokeniz.special_token_to_id["<|endoftext|>"]
    for i in range(max_length):
        x = torch.tensor([ids[-context_length:]], dtype=torch.long)
        logits = model.forward(x)
        logits = logits[0, -1]
        if tau == 0:
            x = torch.argmax(logits)
        else:
            logits = logits /  tau
            sft = softmax.softmax(-1)
            p = sft.forward(logits)
            sorted_p, index = torch.sort(p, descending=True)
            su = 0
            for k in range(len(sorted_p)):
                su += sorted_p[k]
                if su >= top_p:
                    p[index[k + 1:]] = 0
                    break
            x = torch.multinomial(p, 1)
        if x.item() == end_id:
            break
        ids.append(x.item())
    return ids

