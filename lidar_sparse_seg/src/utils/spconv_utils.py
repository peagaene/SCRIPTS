from __future__ import annotations

import torch
import spconv.pytorch as spconv


def build_spconv_tensor(features: torch.Tensor, coordinates: torch.Tensor) -> spconv.SparseConvTensor:
    """Create a SparseConvTensor from batched coords [N,4] and features [N,C]."""
    if coordinates.numel() == 0:
        raise ValueError("Cannot create SparseConvTensor from empty coordinates")

    coords = coordinates.int()
    feats = features.float()

    max_xyz = torch.amax(coords[:, 1:], dim=0)
    spatial_shape = [max(int(v.item()) + 1, 1) for v in max_xyz]
    batch_size = int(torch.amax(coords[:, 0]).item()) + 1

    return spconv.SparseConvTensor(
        features=feats,
        indices=coords,
        spatial_shape=spatial_shape,
        batch_size=batch_size,
    )
