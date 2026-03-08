"""VICReg loss: invariance (paired), variance, covariance."""

from typing import Optional

import torch
import torch.nn as nn
from einops import einsum


class VICReg(nn.Module):
    def __init__(
        self,
        dim_in: int,
        inv_factor: float = 25.0,
        var_factor: float = 25.0,
        cov_factor: float = 1.0,
        var_cutoff: float = 1.0,
        eps: float = 1e-4,
        proj_dim: Optional[int] = None,
        use_projector: bool = True,
    ):
        super().__init__()
        self.inv_factor = inv_factor
        self.var_factor = var_factor
        self.cov_factor = cov_factor
        self.var_cutoff = var_cutoff
        self.eps = eps
        proj_dim = proj_dim or (4 * dim_in)
        self.projector = (
            nn.Sequential(
                nn.Linear(dim_in, 4 * dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(4 * dim_in, 4 * dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(4 * dim_in, 4 * dim_in),
            )
            if use_projector
            else nn.Identity()
        )

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        match_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            z1: (N1, D) embeddings from view 1.
            z2: (N2, D) embeddings from view 2.
            match_idx: (2, M) long tensor; match_idx[0] indices into z1, match_idx[1] into z2 for M pairs.
        Returns:
            loss, dict with inv_loss, var_loss, cov_loss.
        """
        p1 = self.projector(z1)
        p2 = self.projector(z2)
        idx1, idx2 = match_idx[0], match_idx[1]
        inv_loss = (p1[idx1] - p2[idx2]).pow(2).mean()
        var_loss = self._var_loss(p1) + self._var_loss(p2)
        cov_loss = self._cov_loss(p1) + self._cov_loss(p2)
        loss = self.inv_factor * inv_loss + self.var_factor * var_loss + self.cov_factor * cov_loss
        return loss, {
            "inv": inv_loss.item(),
            "var": var_loss.item(),
            "cov": cov_loss.item(),
        }

    def _var_loss(self, z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0) + self.eps)
        return (self.var_cutoff - std).clamp(min=0).mean()

    def _cov_loss(self, z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov = einsum(z_centered, z_centered, "n d1, n d2 -> d1 d2") / (n - 1)
        cov.fill_diagonal_(0)
        return cov.pow(2).sum() / d
