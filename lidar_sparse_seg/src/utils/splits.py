from __future__ import annotations

from pathlib import Path

import numpy as np


def _collect_las_files(data_root: Path, recursive: bool = True) -> list[Path]:
    patterns = ["*.las", "*.laz", "*.LAS", "*.LAZ"]
    files: list[Path] = []
    for pattern in patterns:
        if recursive:
            files.extend(data_root.rglob(pattern))
        else:
            files.extend(data_root.glob(pattern))

    # Deduplicate and sort for deterministic behavior before RNG shuffle.
    unique = sorted(set(files))
    return [p for p in unique if p.is_file()]


def _write_split(path: Path, items: list[Path], data_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(item.relative_to(data_root)).replace('\\', '/') for item in items]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def ensure_or_generate_splits(cfg: dict) -> dict[str, int]:
    split_cfg = cfg.get("split", {})
    auto = bool(split_cfg.get("auto_generate", False))

    train_path = Path(cfg["paths"]["train_split"])
    val_path = Path(cfg["paths"]["val_split"])
    test_path = Path(cfg["paths"]["test_split"])

    if not auto:
        return {
            "generated": 0,
            "train_count": _count_non_empty_lines(train_path),
            "val_count": _count_non_empty_lines(val_path),
            "test_count": _count_non_empty_lines(test_path),
        }

    regenerate = bool(split_cfg.get("regenerate", False))
    if train_path.exists() and val_path.exists() and test_path.exists() and not regenerate:
        return {
            "generated": 0,
            "train_count": _count_non_empty_lines(train_path),
            "val_count": _count_non_empty_lines(val_path),
            "test_count": _count_non_empty_lines(test_path),
        }

    data_root = Path(cfg["paths"]["data_root"])
    recursive = bool(split_cfg.get("recursive", True))
    seed = int(split_cfg.get("seed", cfg.get("seed", 42)))
    train_ratio = float(split_cfg.get("train_ratio", 0.7))
    val_ratio = float(split_cfg.get("val_ratio", 0.15))
    test_ratio = float(split_cfg.get("test_ratio", 0.15))

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_sum}")

    files = _collect_las_files(data_root=data_root, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files found in data_root={data_root}")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    files = [files[i] for i in idx.tolist()]

    n = len(files)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes from {n} files: train={n_train}, val={n_val}, test={n_test}. "
            "Adjust ratios or add more files."
        )

    train_items = files[:n_train]
    val_items = files[n_train : n_train + n_val]
    test_items = files[n_train + n_val :]

    _write_split(train_path, train_items, data_root)
    _write_split(val_path, val_items, data_root)
    _write_split(test_path, test_items, data_root)

    return {
        "generated": 1,
        "train_count": len(train_items),
        "val_count": len(val_items),
        "test_count": len(test_items),
    }


def _count_non_empty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))
