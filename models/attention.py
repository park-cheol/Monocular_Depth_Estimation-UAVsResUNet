import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils.utils import *
from models.modules import *

import sys
import numpy as np


class Attention(nn.Module):
    def __init__(self, dim, proj_kernel=3, kv_proj_stride=2, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.laynorm = LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = DepthwiseSeparableConv(dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False)
        self.to_kv = DepthwiseSeparableConv(dim, inner_dim * 2, proj_kernel, padding=padding, stride=kv_proj_stride, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        x = self.laynorm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)


if __name__ == "__main__":
    q = torch.randn(2, 512, 32, 32).cuda()
    kv = torch.randn(2, 512, 32, 32).cuda()
    m = Attention(dim=512).cuda()
    print(m(q).size())