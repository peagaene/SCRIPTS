from __future__ import annotations

import numpy as np


def compute_local_geom_features(
    xyz: np.ndarray,
    cell_size: float = 1.0,
    min_points: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute local geometric descriptors by XY grid-cell PCA.

    Returns:
    - normal_z in [0, 1]
    - roughness (std point-to-plane distance, meters)
    - slope in [0, 1] (0=flat/horizontal surface, 1=vertical surface)
    - planarity in [0, 1]
    - linearity in [0, 1]
    """
    n = xyz.shape[0]
    if n == 0:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty, empty, empty, empty

    cell = max(float(cell_size), 1e-3)
    x0 = float(np.min(xyz[:, 0]))
    y0 = float(np.min(xyz[:, 1]))
    gx = np.floor((xyz[:, 0] - x0) / cell).astype(np.int32)
    gy = np.floor((xyz[:, 1] - y0) / cell).astype(np.int32)

    keys = np.stack([gx, gy], axis=1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)

    normal_z = np.ones((n,), dtype=np.float32)
    roughness = np.zeros((n,), dtype=np.float32)
    slope = np.zeros((n,), dtype=np.float32)
    planarity = np.zeros((n,), dtype=np.float32)
    linearity = np.zeros((n,), dtype=np.float32)

    order = np.argsort(inverse)
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

        # eigh returns ascending eigenvalues.
        eigvals_asc, eigvecs = np.linalg.eigh(cov)
        # Largest->smallest for common geometric formulas.
        eigvals = eigvals_asc[::-1]
        normal = eigvecs[:, 0]  # smallest eigenvector (from ascending output)
        nz = float(abs(normal[2]))

        l1 = float(max(eigvals[0], 1e-12))
        l2 = float(max(eigvals[1], 0.0))
        l3 = float(max(eigvals[2], 0.0))

        dist = centered @ normal
        rough = float(np.std(dist))

        lin = (l1 - l2) / l1
        pla = (l2 - l3) / l1
        slp = 1.0 - nz

        normal_z[idx] = nz
        roughness[idx] = rough
        linearity[idx] = np.clip(lin, 0.0, 1.0)
        planarity[idx] = np.clip(pla, 0.0, 1.0)
        slope[idx] = np.clip(slp, 0.0, 1.0)

    return (
        normal_z.astype(np.float32),
        roughness.astype(np.float32),
        slope.astype(np.float32),
        planarity.astype(np.float32),
        linearity.astype(np.float32),
    )


def compute_local_normal_features(
    xyz: np.ndarray,
    cell_size: float = 1.0,
    min_points: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible helper returning only normal_z and roughness."""
    normal_z, roughness, _, _, _ = compute_local_geom_features(
        xyz=xyz,
        cell_size=cell_size,
        min_points=min_points,
    )
    return normal_z, roughness
