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

import get_lr_cosine_schedule
import get_batch
import gradient_clipping
import transformer_lm
import cross_entropy
import train_bpe
import tokenizer
import adamw
import save_checkpoint

datapath       = "D:/code/cs336/cs336_assignment1/TinyStoriesV2-GPT4-valid.txt"
modelpath       = "D:/code/cs336/cs336_assignment1/train_model/model"
rulepath       = "D:/code/cs336/cs336_assignment1/train_model/rule"

vocab_size     = 1000
special_tokens = ['<|endoftext|>']
context_length   = 256
d_model        = 512
d_ff           = 1344
num_layers     = 4
num_heads      = 16
rope_theta     = 10000
lr             = 1e-3
weight_decay   = 0.1
betas          = (0.9, 0.95)
eps            = 1e-8
batch_size     = 50
max_norm       = 1.0

epoch          = 100

def bpe():
    if os.path.exists(rulepath + "_vocab.pkl"):
        return
    rule = train_bpe.run_train_bpe(datapath, vocab_size, special_tokens)
    with open(rulepath + "_vocab.pkl", "wb") as f:
        pickle.dump(rule[0], f)
    with open(rulepath + "_merges.pkl", "wb") as f:
        pickle.dump(rule[1], f)

def getdata():
    if not os.path.exists("tokens.npy"):
        tokeniz = tokenizer.tokenizer.from_files(
            rulepath + "_vocab.pkl",
            rulepath + "_merges.pkl",
            special_tokens=special_tokens,
        )
        with open(datapath, "r", encoding="utf-8") as f:
            text = f.read()
        ids = tokeniz.encode(text)
        arr = np.array(ids, dtype = np.uint16)
        np.save("tokens.npy", arr)
    data = np.load("tokens.npy", mmap_mode="r")
    return data

def train():
    dataset = getdata()
    model = transformer_lm.transformer(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    optimizer = adamw.adamw(model.parameters(), lr, weight_decay, betas, eps)
    print('start train')
    for i in range(epoch):
        now_lr = get_lr_cosine_schedule.get_lr_cosine_schedule(i, lr, lr*0.1, epoch*0.1, epoch)
        for group in optimizer.param_groups:
            group["lr"] = now_lr
        optimizer.zero_grad()
        (x, y) = get_batch.get_batch(dataset, batch_size, context_length, 'cpu')
        logits = model.forward(x)
        new_log = rearrange(logits, '... context_length vocab -> (... context_length) vocab')
        new_y   = rearrange(y, '... context_length -> (... context_length)')
        cross = cross_entropy.cross_entropy()
        entropy = cross.forward(new_log, new_y)
        entropy.backward()
        gradient_clipping.gradient_clipping(model.parameters(), max_norm)
        optimizer.step()
        if (i+1)%10==0:
            save_checkpoint.save_checkpoint(model, optimizer, i, modelpath+str(i))
        print(f'epoch {i} OK, loss = {entropy.item()}')

bpe()
train()