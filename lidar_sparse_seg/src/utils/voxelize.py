from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def _aggregate_mean(features: np.ndarray, inverse: np.ndarray, n_unique: int) -> np.ndarray:
    out = np.zeros((n_unique, features.shape[1]), dtype=np.float32)
    counts = np.bincount(inverse, minlength=n_unique).astype(np.float32)
    for d in range(features.shape[1]):
        out[:, d] = np.bincount(inverse, weights=features[:, d], minlength=n_unique) / np.maximum(counts, 1.0)
    return out


def _aggregate_majority(labels: np.ndarray, inverse: np.ndarray, n_unique: int, num_classes: int) -> np.ndarray:
    votes = np.zeros((n_unique, num_classes), dtype=np.int64)
    np.add.at(votes, (inverse, labels), 1)
    return np.argmax(votes, axis=1).astype(np.int64)


def voxelize_points(
    xyz: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray | None,
    voxel_size: float,
    num_classes: int,
) -> dict[str, np.ndarray]:
    """Quantize points into voxels and aggregate features/labels."""
    if xyz.shape[0] == 0:
        return {
            "coords": np.empty((0, 3), dtype=np.int32),
            "features": np.empty((0, features.shape[1]), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64) if labels is not None else None,
            "inverse": np.empty((0,), dtype=np.int64),
        }

    coords = np.floor(xyz / voxel_size).astype(np.int32)
    # spconv expects non-negative coordinates per sample.
    coords = coords - np.min(coords, axis=0, keepdims=True)
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    n_unique = unique_coords.shape[0]

    vox_features = _aggregate_mean(features, inverse, n_unique)
    vox_labels = None
    if labels is not None:
        vox_labels = _aggregate_majority(labels.astype(np.int64), inverse, n_unique, num_classes)

    return {
        "coords": unique_coords,
        "features": vox_features,
        "labels": vox_labels,
        "inverse": inverse.astype(np.int64),
    }


def sparse_collate_fn(batch: Iterable[dict]) -> dict[str, torch.Tensor]:
    """Collate voxelized samples into spconv batched tensors."""
    coord_list = []
    feat_list = []
    label_list = []
    has_labels = True

    for bidx, sample in enumerate(batch):
        coords = sample["coords"]
        feats = sample["features"]
        labels = sample.get("labels")
        if coords.shape[0] == 0:
            continue

        batch_col = np.full((coords.shape[0], 1), bidx, dtype=np.int32)
        bcoords = np.concatenate([batch_col, coords.astype(np.int32)], axis=1)
        coord_list.append(torch.from_numpy(bcoords).int())
        feat_list.append(torch.from_numpy(feats).float())

        if labels is None:
            has_labels = False
        else:
            label_list.append(torch.from_numpy(labels).long())

    if not coord_list:
        return {
            "coords": torch.zeros((0, 4), dtype=torch.int32),
            "features": torch.zeros((0, 2), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        }

    out = {
        "coords": torch.cat(coord_list, dim=0),
        "features": torch.cat(feat_list, dim=0),
    }
    if has_labels:
        out["labels"] = torch.cat(label_list, dim=0)

    return out
