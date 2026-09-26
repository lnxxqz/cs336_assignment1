import numpy.typing as npt
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from einops import einsum, rearrange

class adamw(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:continue

                t = self.state[p].get('t',0)+1
                m = self.state[p].get('m',torch.zeros_like(p))
                v = self.state[p].get('v',torch.zeros_like(p))
                (beta1, beta2) = group['betas']
                weight_decay   = group['weight_decay']
                lr   = group['lr']
                eps  = group['eps']
                alpha_t = lr * math.sqrt(1 - beta2**t)/(1 - beta1**t)
                p.data = p.data - weight_decay * lr * p.data
                m = beta1*m + (1-beta1)*p.grad
                v = beta2*v + (1-beta2)*(p.grad**2)
                p.data = p.data - alpha_t * m / (v.sqrt() + eps)
                self.state[p]['t'] = t
                self.state[p]['m'] = m
                self.state[p]['v'] = v



def get_adamw_cls():
    return adamw