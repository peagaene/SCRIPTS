from __future__ import annotations

import numpy as np


def compute_local_normal_features(
    xyz: np.ndarray,
    cell_size: float = 1.0,
    min_points: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute approximate local normal_z and roughness by XY grid-cell PCA.

    normal_z: abs(z-component of local surface normal) in [0, 1].
    roughness: std of point-to-plane distances in local cell (meters).
    """
    n = xyz.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    cell = max(float(cell_size), 1e-3)
    x0 = float(np.min(xyz[:, 0]))
    y0 = float(np.min(xyz[:, 1]))
    gx = np.floor((xyz[:, 0] - x0) / cell).astype(np.int32)
    gy = np.floor((xyz[:, 1] - y0) / cell).astype(np.int32)

    keys = np.stack([gx, gy], axis=1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)

    normal_z = np.ones((n,), dtype=np.float32)
    roughness = np.zeros((n,), dtype=np.float32)

    order = np.argsort(inverse)
    inv_sorted = inverse[order]
    counts = np.bincount(inverse)

    ptr = 0
    for c in counts:
        if c <= 0:
            continue

        idx = order[ptr : ptr + c]
        ptr += c

        if c < max(int(min_points), 3):
            continue

        pts = xyz[idx]
        center = pts.mean(axis=0, keepdims=True)
        centered = pts - center

        cov = centered.T @ centered
        cov /= max(c - 1, 1)

        eigvals, eigvecs = np.linalg.eigh(cov)
        # Smallest eigenvector is local normal direction.
        normal = eigvecs[:, 0]
        nz = float(abs(normal[2]))

        dist = centered @ normal
        rough = float(np.std(dist))

        normal_z[idx] = nz
        roughness[idx] = rough

    return normal_z.astype(np.float32), roughness.astype(np.float32)
