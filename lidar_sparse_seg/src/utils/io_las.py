from __future__ import annotations

from pathlib import Path
from typing import Any

import laspy
import numpy as np


def _optional_dimension(las: laspy.LasData, name: str, dtype, default: float = 0.0) -> np.ndarray:
    """Read optional LAS dimension; fallback to constant when missing."""
    if hasattr(las, name):
        return np.asarray(getattr(las, name), dtype=dtype)
    return np.full((len(las.points),), default, dtype=dtype)


def read_las_points(path: str | Path) -> dict[str, Any]:
    """Read LAS/LAZ file and return basic point attributes."""
    path = Path(path)
    las = laspy.read(path)

    if len(las.points) == 0:
        return {
            "x": np.empty((0,), dtype=np.float64),
            "y": np.empty((0,), dtype=np.float64),
            "z": np.empty((0,), dtype=np.float64),
            "intensity": np.empty((0,), dtype=np.float32),
            "classification": np.empty((0,), dtype=np.int32),
            "return_number": np.empty((0,), dtype=np.float32),
            "number_of_returns": np.empty((0,), dtype=np.float32),
            "scan_angle": np.empty((0,), dtype=np.float32),
            "las": las,
        }

    intensity = np.asarray(las.intensity, dtype=np.float32)
    classification = np.asarray(las.classification, dtype=np.int32)
    return_number = _optional_dimension(las, "return_number", np.float32, default=1.0)
    number_of_returns = _optional_dimension(las, "number_of_returns", np.float32, default=1.0)
    # Different LAS point formats may expose either "scan_angle" or "scan_angle_rank".
    if hasattr(las, "scan_angle"):
        scan_angle = np.asarray(las.scan_angle, dtype=np.float32)
    else:
        scan_angle = _optional_dimension(las, "scan_angle_rank", np.float32, default=0.0)

    return {
        "x": np.asarray(las.x, dtype=np.float64),
        "y": np.asarray(las.y, dtype=np.float64),
        "z": np.asarray(las.z, dtype=np.float64),
        "intensity": intensity,
        "classification": classification,
        "return_number": return_number,
        "number_of_returns": number_of_returns,
        "scan_angle": scan_angle,
        "las": las,
    }


def write_las_with_classification(
    input_path: str | Path,
    output_path: str | Path,
    classification: np.ndarray,
) -> None:
    """Write output LAS/LAZ by replacing classification dimension."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    las = laspy.read(input_path)
    if len(classification) != len(las.points):
        raise ValueError(
            f"classification size mismatch: got {len(classification)} expected {len(las.points)}"
        )

    las.classification = classification.astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(output_path)


def resolve_path(line: str, data_root: str | Path) -> Path:
    """Resolve split-file path entry as absolute or relative to data_root."""
    raw = Path(line.strip())
    if raw.is_absolute():
        return raw
    return Path(data_root) / raw
