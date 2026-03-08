"""Projector: H -> Z for VICReg training (not used at inference)."""

import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, dim_in: int, dim_out: int | None = None, hidden_mult: int = 4):
        super().__init__()
        dim_out = dim_out or dim_in
        hidden = dim_in * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim_out),
        )

    def forward(self, x):
        return self.net(x)
