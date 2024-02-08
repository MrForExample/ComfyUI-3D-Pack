# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import os
import warnings

from torch import Tensor
from torch import nn

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_q: int,
        dim_k: int,
        dim_v: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(dim_q, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim_k, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim_v, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # q: [B, N, Cq]
        # k: [B, M, Ck]
        # v: [B, M, Cv]
        # return: [B, N, C]

        B, N, _ = q.shape
        M = k.shape[1]
        
        q = self.scale * self.to_q(q).reshape(B, N, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3) # [B, nh, N, C/nh]
        k = self.to_k(k).reshape(B, M, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3) # [B, nh, M, C/nh]
        v = self.to_v(v).reshape(B, M, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3) # [B, nh, M, C/nh]

        attn = q @ k.transpose(-2, -1) # [B, nh, N, M]

        attn = attn.softmax(dim=-1) # [B, nh, N, M]
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1) # [B, nh, N, M] @ [B, nh, M, C/nh] --> [B, nh, N, C/nh] --> [B, N, nh, C/nh] --> [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffCrossAttention(CrossAttention):
    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, _ = q.shape
        M = k.shape[1]

        q = self.scale * self.to_q(q).reshape(B, N, self.num_heads, self.dim // self.num_heads) # [B, N, nh, C/nh]
        k = self.to_k(k).reshape(B, M, self.num_heads, self.dim // self.num_heads) # [B, M, nh, C/nh]
        v = self.to_v(v).reshape(B, M, self.num_heads, self.dim // self.num_heads) # [B, M, nh, C/nh]

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
