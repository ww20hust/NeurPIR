"""Sinusoidal period computation for rotary time embedding."""

import torch


def get_periods(num: int, t_min: float, t_max: float) -> torch.Tensor:
    exponents = torch.linspace(0, 1.0, num, dtype=torch.float32)
    t_min_t = torch.tensor(t_min, dtype=torch.float32)
    t_max_t = torch.tensor(t_max, dtype=torch.float32)
    periods = torch.exp(torch.lerp(t_min_t.log(), t_max_t.log(), exponents))
    return periods
