"""Spatial position embedding: target neuron first, then neighbors by distance (tokenized)."""

import torch
import torch.nn as nn
from torch import Tensor


class SpatialPosEmbedding(nn.Module):
    """
    Embedding for spatial order: index 0 = target neuron, indices 1..K = distance bins
    for n-1 neighbors (distance to target discretized into K bins).
    """

    def __init__(self, num_distance_bins: int, dim: int):
        super().__init__()
        self.num_bins = num_distance_bins
        self.dim = dim
        self.embedding = nn.Embedding(1 + num_distance_bins, dim)  # 0 = target, 1..K = distance bins

    def forward(
        self,
        distance_bin_ids: Tensor,
    ) -> Tensor:
        """
        Args:
            distance_bin_ids: (U,) or (B, U) int64. 0 = target, 1..K = neighbor distance bin.
        Returns:
            (U, D) or (B, U, D) position embeddings.
        """
        return self.embedding(distance_bin_ids)

    @staticmethod
    def distances_to_bin_ids(
        distances: Tensor,
        num_bins: int,
        max_distance: float | None = None,
    ) -> Tensor:
        """
        Convert continuous distances to bin indices in [0, num_bins-1].
        Index 0 is reserved for target (caller should set target positions to 0).
        So neighbors get bins 1..num_bins (we use 1 + bin so embedding has 1+num_bins entries).

        Args:
            distances: (n_neighbors,) or (B, n_neighbors) distances to target.
            num_bins: number of bins (excluding target).
            max_distance: if None, use max of distances per batch or global.
        Returns:
            (n_neighbors,) or (B, n_neighbors) int64 in [1, num_bins]. Caller sets target to 0.
        """
        if max_distance is None:
            max_distance = distances.max().clamp(min=1e-6).item()
        bins = (distances / max_distance).clamp(0, 1) * (num_bins - 1)
        bin_ids = bins.long().clamp(0, num_bins - 1) + 1
        return bin_ids
