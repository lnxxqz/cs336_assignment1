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
import time

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
PATH           = "D:/code/cs336/cs336_assignment1/train_model/lr_1e-3/"

valid_datapath = "D:/code/cs336/cs336_assignment1/TinyStoriesV2-GPT4-valid.txt"
valid_PATH     = PATH + "valid/"
valid_tokenpath= valid_PATH + "tokens.npy"

rulepath       = PATH + "rule"
tokenpath      = PATH + "tokens.npy"
modelpath      = PATH + "model"
rulepath       = PATH + "rule"
tokenpath      = PATH + "tokens.npy"
logpath        = PATH + "log.txt"

vocab_size     = 10000
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

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'device: {device}')

def bpe(datapath, rulepath):
    if os.path.exists(rulepath + "_vocab.pkl"):
        return
    rule = train_bpe.run_train_bpe(datapath, vocab_size, special_tokens)
    with open(rulepath + "_vocab.pkl", "wb") as f:
        pickle.dump(rule[0], f)
    with open(rulepath + "_merges.pkl", "wb") as f:
        pickle.dump(rule[1], f)

def getdata(datapath, rulepath, tokenpath):
    print('start getdata...')
    if not os.path.exists(tokenpath):
        tokeniz = tokenizer.tokenizer.from_files(
            rulepath + "_vocab.pkl",
            rulepath + "_merges.pkl",
            special_tokens=special_tokens,
        )
        with open(datapath, "r", encoding="utf-8") as f:
            text = f.read()
        ids = tokeniz.encode(text)
        arr = np.array(ids, dtype = np.uint16)
        np.save(tokenpath, arr)
    data = np.load(tokenpath, mmap_mode="r")
    print('getdata ok')
    return data

def getbatch(dataset, batch_size, context_length, device):
    return get_batch.get_batch(dataset, batch_size, context_length, device)
def get_loss(model, x, label, loss):
    logits = model.forward(x)
    new_log = rearrange(logits, '... context_length vocab -> (... context_length) vocab')
    new_y   = rearrange(label, '... context_length -> (... context_length)')
    entropy = loss.forward(new_log, new_y)
    return entropy


def train():
    model = transformer_lm.transformer(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device)
    model.to(device)
    optimizer = adamw.adamw(model.parameters(), lr, weight_decay, betas, eps)
    print('start train')
    start_time = time.perf_counter()
    cross = cross_entropy.cross_entropy()
    for i in range(epoch):
        now_lr = get_lr_cosine_schedule.get_lr_cosine_schedule(i, lr, lr*0.1, epoch*0.1, epoch)
        for group in optimizer.param_groups:
            group["lr"] = now_lr
        optimizer.zero_grad()
        (x, y) = getbatch(dataset, batch_size, context_length, device)
        entropy = get_loss(model, x, y, cross)
        entropy.backward()
        gradient_clipping.gradient_clipping(model.parameters(), max_norm)
        optimizer.step()
        if (i+1)%10==0:
            save_checkpoint.save_checkpoint(model, optimizer, i, modelpath+str(i))
            with torch.no_grad():
                (vx, vy) = getbatch(valid_dataset, batch_size, context_length, device)
                valid_entropy = get_loss(model, vx, vy, cross)
            with open(logpath, "a", encoding="utf-8") as f:
                f.write(f"step {i}\t loss{entropy.item()}\t valid_loss {valid_entropy.item()}\t time {time.perf_counter()-start_time}\n")
        print(f'epoch {i} OK, loss = {entropy.item()}')

bpe(datapath, rulepath)

dataset = getdata(datapath, rulepath, tokenpath)
valid_dataset = getdata(valid_datapath, rulepath, valid_tokenpath)

train()