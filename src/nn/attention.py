"""Self-attention with optional rotary (temporal) or no rotary (spatial)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .rotary_embedding import apply_rotary_pos_emb, invert_rotary_pos_emb


class RotarySelfAttention(nn.Module):
    """Self-attention with rotary position embedding on q, k, v (for temporal dimension)."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        dropout: float = 0.0,
        rotate_value: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.rotate_value = rotate_value
        dim_head = dim_head or (dim // heads)
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        rotary: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (N, D) or (B, N, D)
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        need_unsqueeze = q.ndim == 2
        if need_unsqueeze:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        B, N, _ = q.shape
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        if rotary is not None:
            if rotary.ndim == 2:
                rotary = rotary.unsqueeze(0)
            q = apply_rotary_pos_emb(rotary, q)
            k = apply_rotary_pos_emb(rotary, k)
            if self.rotate_value:
                v = apply_rotary_pos_emb(rotary, v)

        scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)
        if rotary is not None and self.rotate_value:
            out = apply_rotary_pos_emb(invert_rotary_pos_emb(rotary), out)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        if need_unsqueeze:
            out = out.squeeze(0)
        return out


class SelfAttention(nn.Module):
    """Plain self-attention (for spatial dimension; add spatial pos embedding before calling)."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        dropout: float = 0.0,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        dim_head = dim_head or (dim // heads)
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        need_unsqueeze = q.ndim == 2
        if need_unsqueeze:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        out = self.to_out(out)
        if need_unsqueeze:
            out = out.squeeze(0)
        return out
