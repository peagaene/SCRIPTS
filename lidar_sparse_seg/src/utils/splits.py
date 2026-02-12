from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils.io_las import read_las_points


def _collect_las_files(data_root: Path, recursive: bool = True) -> list[Path]:
    patterns = ["*.las", "*.laz", "*.LAS", "*.LAZ"]
    files: list[Path] = []
    for pattern in patterns:
        if recursive:
            files.extend(data_root.rglob(pattern))
        else:
            files.extend(data_root.glob(pattern))

    unique = sorted(set(files))
    return [p for p in unique if p.is_file()]


def _write_split(path: Path, items: list[Path], data_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(item.relative_to(data_root)).replace('\\', '/') for item in items]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _count_non_empty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))


def _compute_file_target_scores(
    files: list[Path],
    las_to_train_map_path: str | Path,
    ignore_las_classes_path: str | Path,
    target_train_class: int,
) -> np.ndarray:
    las_to_train = json.loads(Path(las_to_train_map_path).read_text(encoding="utf-8-sig"))
    las_to_train = {int(k): int(v) for k, v in las_to_train.items()}
    ignore_set = {int(v) for v in json.loads(Path(ignore_las_classes_path).read_text(encoding="utf-8-sig"))}

    valid_las_for_target = {las_c for las_c, tr_c in las_to_train.items() if tr_c == target_train_class}
    if not valid_las_for_target:
        return np.zeros((len(files),), dtype=np.float64)

    scores = np.zeros((len(files),), dtype=np.float64)
    for i, f in enumerate(files):
        try:
            cls = read_las_points(f)["classification"].astype(np.int32)
        except Exception:
            continue

        if cls.size == 0:
            continue

        keep = np.ones_like(cls, dtype=bool)
        for c in ignore_set:
            keep &= cls != c
        cls = cls[keep]
        if cls.size == 0:
            continue

        scores[i] = float(np.isin(cls, list(valid_las_for_target)).sum())

    return scores


def _sample_indices_weighted_without_replacement(
    rng: np.random.Generator,
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    n_pick: int,
    alpha: float,
) -> np.ndarray:
    if n_pick <= 0:
        return np.array([], dtype=np.int64)
    if candidate_indices.size < n_pick:
        raise ValueError(f"Cannot pick {n_pick} from {candidate_indices.size} candidates")

    w = scores[candidate_indices].astype(np.float64)
    w = w + float(alpha)
    if np.all(w <= 0):
        w = np.ones_like(w, dtype=np.float64)
    p = w / w.sum()
    return rng.choice(candidate_indices, size=n_pick, replace=False, p=p).astype(np.int64)


def _build_splits_with_target_balance(
    files: list[Path],
    rng: np.random.Generator,
    n_train: int,
    n_val: int,
    n_test: int,
    scores: np.ndarray,
    min_positive_files_per_eval_split: int,
    weight_alpha: float,
) -> tuple[list[Path], list[Path], list[Path]]:
    n = len(files)
    all_indices = np.arange(n, dtype=np.int64)

    positive = np.where(scores > 0)[0]

    min_pos = int(max(min_positive_files_per_eval_split, 0))
    min_pos = min(min_pos, n_val, n_test)

    if positive.size >= 2 * min_pos and min_pos > 0:
        pos = positive.copy()
        rng.shuffle(pos)
        val_seed = pos[:min_pos]
        test_seed = pos[min_pos : 2 * min_pos]
        seeded = np.concatenate([val_seed, test_seed])
    else:
        val_seed = np.array([], dtype=np.int64)
        test_seed = np.array([], dtype=np.int64)
        seeded = np.array([], dtype=np.int64)

    remaining = np.setdiff1d(all_indices, seeded, assume_unique=False)

    val_extra = _sample_indices_weighted_without_replacement(
        rng=rng,
        candidate_indices=remaining,
        scores=scores,
        n_pick=n_val - val_seed.size,
        alpha=weight_alpha,
    )
    remaining = np.setdiff1d(remaining, val_extra, assume_unique=False)

    test_extra = _sample_indices_weighted_without_replacement(
        rng=rng,
        candidate_indices=remaining,
        scores=scores,
        n_pick=n_test - test_seed.size,
        alpha=weight_alpha,
    )

    val_idx = np.concatenate([val_seed, val_extra]).astype(np.int64)
    test_idx = np.concatenate([test_seed, test_extra]).astype(np.int64)
    train_idx = np.setdiff1d(all_indices, np.concatenate([val_idx, test_idx]), assume_unique=False)

    if train_idx.size != n_train or val_idx.size != n_val or test_idx.size != n_test:
        raise RuntimeError(
            f"Split size mismatch after balancing: train={train_idx.size}, val={val_idx.size}, test={test_idx.size}"
        )

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_items = [files[i] for i in train_idx.tolist()]
    val_items = [files[i] for i in val_idx.tolist()]
    test_items = [files[i] for i in test_idx.tolist()]
    return train_items, val_items, test_items


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

    n = len(files)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes from {n} files: train={n_train}, val={n_val}, test={n_test}. "
            "Adjust ratios or add more files."
        )

    mode = str(split_cfg.get("mode", "random")).lower()
    if mode == "balance_target_class":
        target_train_class = int(split_cfg.get("target_train_class", 1))
        min_positive = int(split_cfg.get("min_positive_files_per_eval_split", 1))
        weight_alpha = float(split_cfg.get("balance_weight_alpha", 1.0))

        scores = _compute_file_target_scores(
            files=files,
            las_to_train_map_path=cfg["paths"]["las_to_train"],
            ignore_las_classes_path=cfg["paths"]["ignore_las_classes"],
            target_train_class=target_train_class,
        )

        train_items, val_items, test_items = _build_splits_with_target_balance(
            files=files,
            rng=rng,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            scores=scores,
            min_positive_files_per_eval_split=min_positive,
            weight_alpha=weight_alpha,
        )
    else:
        idx = np.arange(len(files))
        rng.shuffle(idx)
        files = [files[i] for i in idx.tolist()]

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
