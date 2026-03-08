"""
Inference: extract target-neuron embeddings H from binned activity + distance_bin_ids.
No projector; optional aggregation over multiple windows per neuron.
"""

from typing import Optional

import torch
import torch.nn as nn


def extract_embeddings(
    encoder: nn.Module,
    bins: torch.Tensor,
    distance_bin_ids: torch.Tensor,
    device: torch.device,
    aggregate: str = "none",
) -> torch.Tensor:
    """
    Args:
        encoder: NeuronEncoder (no projector).
        bins: (B, n, P, bins_per_patch) or (n_windows, n, P, bins_per_patch) for one neuron.
        distance_bin_ids: (B, n) or (n_windows, n).
        device: torch device.
        aggregate: "none" | "mean". If "mean", average over first dim (e.g. multiple windows per neuron).
    Returns:
        (B, dim) or (dim,) if aggregate="mean" and single neuron.
    """
    encoder.eval()
    bins = bins.to(device)
    distance_bin_ids = distance_bin_ids.to(device)
    with torch.no_grad():
        h = encoder(bins, distance_bin_ids)
    if aggregate == "mean" and h.dim() == 2 and h.size(0) > 1:
        h = h.mean(dim=0, keepdim=True)
    return h.cpu()
