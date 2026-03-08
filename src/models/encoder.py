"""
Encoder: binned patches -> temporal transformer (rotary) -> spatio-temporal transformer
(spatial with distance-based positional encoding) -> mean pool -> target neuron embedding H.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from einops import rearrange

from ..nn import RotaryEmbedding, RotarySelfAttention, SelfAttention, SpatialPosEmbedding


class FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0, pre_norm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.in_proj = nn.Linear(dim, 2 * dim * mult)
        self.out_proj = nn.Linear(dim * mult, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(gate) * x
        x = self.dropout(x)
        return self.out_proj(x)


class NeuronEncoder(nn.Module):
    """
    Encode population activity for one time window: [target, n-1 neighbors] with
    binned patches, temporal rotary attention, then spatio-temporal with distance-based
    spatial positional encoding. Output is the embedding of the target (index 0) only.
    """

    def __init__(
        self,
        bins_per_patch: int,
        dim: int,
        num_heads: int = 8,
        dim_head: Optional[int] = None,
        t_layers: int = 2,
        st_layers: int = 2,
        num_latents: int = 32,
        num_distance_bins: int = 32,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        rot_ratio: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents
        self.bins_per_patch = bins_per_patch
        dim_head = dim_head or (dim // num_heads)

        self.read_in = nn.Linear(bins_per_patch, dim)

        t_min, t_max = 1.0, 8.0 * num_latents
        self.rotary_emb = RotaryEmbedding(
            head_dim=dim_head,
            rotate_dim=int(dim_head * rot_ratio),
            t_min=t_min,
            t_max=t_max,
        )

        self.t_blocks = nn.ModuleList()
        for _ in range(t_layers):
            self.t_blocks.append(
                nn.ModuleList([
                    RotarySelfAttention(dim, num_heads, dim_head, attn_dropout, rotate_value=True),
                    FFN(dim, mult=4, dropout=ff_dropout),
                ])
            )

        self.spatial_pos_emb = SpatialPosEmbedding(num_distance_bins, dim)
        self.st_blocks = nn.ModuleList()
        for _ in range(st_layers):
            self.st_blocks.append(
                nn.ModuleList([
                    SelfAttention(dim, num_heads, dim_head, attn_dropout),
                    FFN(dim, mult=4, dropout=ff_dropout),
                    RotarySelfAttention(dim, num_heads, dim_head, attn_dropout, rotate_value=True),
                    FFN(dim, mult=4, dropout=ff_dropout),
                ])
            )

        self.ff_dropout = nn.Dropout(ff_dropout)

    def forward(
        self,
        bins: torch.Tensor,
        distance_bin_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            bins: (B, n, P, bins_per_patch). n = 1 + n_neighbors (target first).
            distance_bin_ids: (B, n) int64. 0 = target, 1..K = neighbor distance bin.
        Returns:
            (B, dim) embedding of the target neuron only (index 0).
        """
        B, n, P, _ = bins.shape
        device = bins.device

        x = self.read_in(bins.float())
        x = rearrange(x, "b n p d -> (b n) p d")
        t = torch.arange(P, device=device, dtype=torch.float32).unsqueeze(0).expand(B * n, -1)
        x_rot = self.rotary_emb(t)

        for t_attn, t_ffn in self.t_blocks:
            x = x + self.ff_dropout(t_attn(x, rotary=x_rot))
            x = x + self.ff_dropout(t_ffn(x))

        x = rearrange(x, "(b n) p d -> b n p d", b=B, n=n)
        spatial_pos = self.spatial_pos_emb(distance_bin_ids)

        for s_attn, s_ffn, t_attn, t_ffn in self.st_blocks:
            x = rearrange(x, "b n p d -> b p n d")
            x_flat = rearrange(x, "b p n d -> (b p) n d")
            pos_flat = spatial_pos.unsqueeze(1).expand(-1, P, -1, -1)
            pos_flat = rearrange(pos_flat, "b p n d -> (b p) n d")
            x_flat = x_flat + pos_flat
            x_flat = x_flat + self.ff_dropout(s_attn(x_flat))
            x_flat = x_flat + self.ff_dropout(s_ffn(x_flat))
            x = rearrange(x_flat, "(b p) n d -> b n p d", b=B, p=P)
            x = rearrange(x, "b n p d -> (b n) p d")
            x = x + self.ff_dropout(t_attn(x, rotary=x_rot))
            x = x + self.ff_dropout(t_ffn(x))
            x = rearrange(x, "(b n) p d -> b n p d", b=B, n=n)

        x = x.mean(dim=2)
        return x[:, 0]
