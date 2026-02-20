from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any
from collections import OrderedDict

import numpy as np
from torch.utils.data import Dataset

from src.utils.geom_features import compute_local_geom_features
from src.utils.io_las import read_las_points, resolve_path
from src.utils.voxelize import voxelize_points


class LidarSemanticDataset(Dataset):
    def __init__(
        self,
        split_file: str | Path,
        data_root: str | Path,
        las_to_train_map_path: str | Path,
        ignore_las_classes_path: str | Path,
        voxel_size: float,
        num_classes: int,
        max_intensity: float = 65535.0,
        crop_size: float | None = 64.0,
        max_points_per_crop: int = 250000,
        use_hag: bool = False,
        hag_cell_size: float = 1.0,
        use_return_features: bool = False,
        use_scan_angle: bool = False,
        use_normal_features: bool = False,
        use_roughness_feature: bool = False,
        use_slope_feature: bool = False,
        use_planarity_feature: bool = False,
        use_linearity_feature: bool = False,
        normal_cell_size: float = 1.0,
        normal_min_points: int = 6,
        roughness_scale: float = 1.0,
        tile_cache_size: int = 4,
        cache_full_samples: bool = False,
        mode: str = "train",
    ) -> None:
        self.mode = mode
        self.data_root = Path(data_root)
        self.voxel_size = float(voxel_size)
        self.num_classes = int(num_classes)
        self.max_intensity = float(max_intensity)
        self.crop_size = crop_size
        self.max_points_per_crop = int(max_points_per_crop)
        self.use_hag = bool(use_hag)
        self.hag_cell_size = float(hag_cell_size)
        self.use_return_features = bool(use_return_features)
        self.use_scan_angle = bool(use_scan_angle)
        self.use_normal_features = bool(use_normal_features)
        self.use_roughness_feature = bool(use_roughness_feature)
        self.use_slope_feature = bool(use_slope_feature)
        self.use_planarity_feature = bool(use_planarity_feature)
        self.use_linearity_feature = bool(use_linearity_feature)
        self.normal_cell_size = float(normal_cell_size)
        self.normal_min_points = int(normal_min_points)
        self.roughness_scale = float(roughness_scale)
        self.tile_cache_size = max(int(tile_cache_size), 0)
        self.cache_full_samples = bool(cache_full_samples)

        self.paths = self._load_split(split_file, self.data_root)
        self.las_to_train = self._load_map(las_to_train_map_path)
        self.ignore_set = self._load_ignore(ignore_las_classes_path)
        self._tile_cache: "OrderedDict[str, dict[str, np.ndarray]]" = OrderedDict()
        self._sample_cache: dict[int, dict[str, Any]] = {}

    @property
    def feature_dim(self) -> int:
        return get_input_channels(
            use_hag=self.use_hag,
            use_return_features=self.use_return_features,
            use_scan_angle=self.use_scan_angle,
            use_normal_features=self.use_normal_features,
            use_roughness_feature=self.use_roughness_feature,
            use_slope_feature=self.use_slope_feature,
            use_planarity_feature=self.use_planarity_feature,
            use_linearity_feature=self.use_linearity_feature,
        )

    @staticmethod
    def _load_split(split_file: str | Path, data_root: Path) -> list[Path]:
        lines = Path(split_file).read_text(encoding="utf-8").splitlines()
        files = []
        for line in lines:
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue
            files.append(resolve_path(clean, data_root))
        return files

    @staticmethod
    def _load_map(path: str | Path) -> dict[int, int]:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return {int(k): int(v) for k, v in raw.items()}

    @staticmethod
    def _load_ignore(path: str | Path) -> set[int]:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return {int(v) for v in raw}

    def __len__(self) -> int:
        return len(self.paths)

    def _cache_put(self, key: str, value: dict[str, np.ndarray]) -> None:
        if self.tile_cache_size <= 0:
            return
        self._tile_cache[key] = value
        self._tile_cache.move_to_end(key)
        while len(self._tile_cache) > self.tile_cache_size:
            self._tile_cache.popitem(last=False)

    def _get_tile_data(self, path: Path) -> dict[str, np.ndarray]:
        key = str(path)
        if key in self._tile_cache:
            self._tile_cache.move_to_end(key)
            return self._tile_cache[key]

        pts = read_las_points(path)
        data = {
            "xyz": np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float64),
            "intensity": pts["intensity"].astype(np.float32),
            "classification": pts["classification"].astype(np.int32),
            "return_number": pts["return_number"].astype(np.float32),
            "number_of_returns": pts["number_of_returns"].astype(np.float32),
            "scan_angle": pts["scan_angle"].astype(np.float32),
        }
        if self.use_hag and data["xyz"].shape[0] > 0:
            data["hag"] = self._compute_hag(data["xyz"], data["classification"])
        self._cache_put(key, data)
        return data

    def _filter_and_map_labels(self, las_cls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = np.ones_like(las_cls, dtype=bool)
        for c in self.ignore_set:
            keep &= las_cls != c

        mapped = np.full_like(las_cls, fill_value=-1, dtype=np.int64)
        for las_c, train_c in self.las_to_train.items():
            mapped[las_cls == las_c] = train_c

        keep &= mapped >= 0
        return keep, mapped

    def _crop_points(self, xyz: np.ndarray) -> np.ndarray:
        n = xyz.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=bool)

        if self.crop_size is None or self.crop_size <= 0:
            mask = np.ones((n,), dtype=bool)
        else:
            half = self.crop_size / 2.0
            if self.mode == "train":
                idx = np.random.randint(0, n)
                cx, cy = xyz[idx, 0], xyz[idx, 1]
            else:
                cx, cy = np.median(xyz[:, 0]), np.median(xyz[:, 1])
            mask = (
                (xyz[:, 0] >= cx - half)
                & (xyz[:, 0] <= cx + half)
                & (xyz[:, 1] >= cy - half)
                & (xyz[:, 1] <= cy + half)
            )
            if mask.sum() == 0:
                mask = np.ones((n,), dtype=bool)

        if mask.sum() > self.max_points_per_crop:
            idx = np.where(mask)[0]
            if self.mode == "train":
                chosen = np.random.choice(idx, size=self.max_points_per_crop, replace=False)
            else:
                chosen = idx[: self.max_points_per_crop]
            out = np.zeros((n,), dtype=bool)
            out[chosen] = True
            return out

        return mask

    def _compute_hag(self, xyz: np.ndarray, las_cls: np.ndarray) -> np.ndarray:
        if xyz.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)

        ground_mask = las_cls == 2
        if not np.any(ground_mask):
            return (xyz[:, 2] - np.min(xyz[:, 2])).astype(np.float32)

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        x0 = np.min(x)
        y0 = np.min(y)
        gx = np.floor((x - x0) / self.hag_cell_size).astype(np.int32)
        gy = np.floor((y - y0) / self.hag_cell_size).astype(np.int32)

        key_all = np.stack([gx, gy], axis=1)
        key_ground = key_all[ground_mask]
        z_ground = z[ground_mask]

        unique_cells, inverse = np.unique(key_ground, axis=0, return_inverse=True)
        ground_min = np.full((unique_cells.shape[0],), np.inf, dtype=np.float64)
        np.minimum.at(ground_min, inverse, z_ground)

        cell_to_min = {tuple(cell.tolist()): float(ground_min[i]) for i, cell in enumerate(unique_cells)}
        global_ground = float(np.min(z_ground))

        z_ref = np.empty_like(z, dtype=np.float64)
        for i, cell in enumerate(key_all):
            z_ref[i] = cell_to_min.get((int(cell[0]), int(cell[1])), global_ground)

        hag = z - z_ref
        hag[hag < 0] = 0
        return hag.astype(np.float32)

    @staticmethod
    def _normalize_return_features(return_number: np.ndarray, number_of_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        denom = np.maximum(number_of_returns, 1.0)
        return_ratio = np.clip(return_number / denom, 0.0, 1.0).astype(np.float32)
        num_returns_norm = np.clip(number_of_returns / 8.0, 0.0, 1.0).astype(np.float32)
        return return_ratio, num_returns_norm

    @staticmethod
    def _normalize_scan_angle(scan_angle: np.ndarray) -> np.ndarray:
        # Typical scan angle range is around [-90, 90]. Normalize to [0, 1].
        return np.clip((scan_angle + 90.0) / 180.0, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        if self.cache_full_samples and self.mode != "train" and idx in self._sample_cache:
            return self._sample_cache[idx]

        tile = self._get_tile_data(path)
        xyz = tile["xyz"].copy()
        intensity = tile["intensity"].copy()
        las_cls = tile["classification"].copy()
        return_number = tile["return_number"].copy()
        number_of_returns = tile["number_of_returns"].copy()
        scan_angle = tile["scan_angle"].copy()
        hag_all = tile.get("hag", None)

        if xyz.shape[0] == 0:
            return {
                "coords": np.empty((0, 3), dtype=np.int32),
                "features": np.empty((0, self.feature_dim), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int64),
                "path": str(path),
            }
        if self.use_hag and hag_all is None:
            # Fallback when tile cache is disabled.
            hag_all = self._compute_hag(xyz.copy(), las_cls.copy())

        keep_mask, mapped = self._filter_and_map_labels(las_cls)
        if not np.any(keep_mask):
            sample = {
                "coords": np.empty((0, 3), dtype=np.int32),
                "features": np.empty((0, self.feature_dim), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int64),
                "path": str(path),
            }
            if self.cache_full_samples and self.mode != "train":
                self._sample_cache[idx] = sample
            return sample

        xyz = xyz[keep_mask]
        intensity = intensity[keep_mask]
        labels = mapped[keep_mask]
        las_cls = las_cls[keep_mask]
        return_number = return_number[keep_mask]
        number_of_returns = number_of_returns[keep_mask]
        scan_angle = scan_angle[keep_mask]
        if hag_all is not None:
            hag_all = hag_all[keep_mask]

        crop_mask = self._crop_points(xyz)
        xyz = xyz[crop_mask]
        intensity = intensity[crop_mask]
        labels = labels[crop_mask]
        las_cls = las_cls[crop_mask]
        return_number = return_number[crop_mask]
        number_of_returns = number_of_returns[crop_mask]
        scan_angle = scan_angle[crop_mask]
        if hag_all is not None:
            hag_all = hag_all[crop_mask]

        if xyz.shape[0] == 0:
            sample = {
                "coords": np.empty((0, 3), dtype=np.int32),
                "features": np.empty((0, self.feature_dim), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int64),
                "path": str(path),
            }
            if self.cache_full_samples and self.mode != "train":
                self._sample_cache[idx] = sample
            return sample

        xy_mean = np.mean(xyz[:, :2], axis=0, keepdims=True)
        xyz[:, :2] = xyz[:, :2] - xy_mean
        z_min = np.min(xyz[:, 2])
        z_rel = (xyz[:, 2] - z_min).astype(np.float32)

        intensity_norm = np.clip(intensity / max(self.max_intensity, 1.0), 0.0, 1.0).astype(np.float32)

        feat_list = [z_rel, intensity_norm]
        if self.use_hag:
            # Use precomputed HAG from original tile references.
            feat_list.append(hag_all.astype(np.float32))
        if self.use_return_features:
            return_ratio, num_returns_norm = self._normalize_return_features(return_number, number_of_returns)
            feat_list.append(return_ratio)
            feat_list.append(num_returns_norm)
        if self.use_scan_angle:
            feat_list.append(self._normalize_scan_angle(scan_angle))
        if (
            self.use_normal_features
            or self.use_roughness_feature
            or self.use_slope_feature
            or self.use_planarity_feature
            or self.use_linearity_feature
        ):
            normal_z, roughness, slope, planarity, linearity = compute_local_geom_features(
                xyz=xyz,
                cell_size=self.normal_cell_size,
                min_points=self.normal_min_points,
            )
            if self.use_normal_features:
                feat_list.append(normal_z)
            if self.use_roughness_feature:
                roughness_norm = np.clip(roughness / max(self.roughness_scale, 1e-6), 0.0, 1.0).astype(np.float32)
                feat_list.append(roughness_norm)
            if self.use_slope_feature:
                feat_list.append(slope.astype(np.float32))
            if self.use_planarity_feature:
                feat_list.append(planarity.astype(np.float32))
            if self.use_linearity_feature:
                feat_list.append(linearity.astype(np.float32))

        features = np.stack(feat_list, axis=1).astype(np.float32)
        vox = voxelize_points(xyz=xyz, features=features, labels=labels, voxel_size=self.voxel_size, num_classes=self.num_classes)

        sample = {
            "coords": vox["coords"],
            "features": vox["features"],
            "labels": vox["labels"],
            "path": str(path),
        }
        if self.cache_full_samples and self.mode != "train":
            self._sample_cache[idx] = sample
        return sample


def compute_class_histogram(
    split_file: str | Path,
    data_root: str | Path,
    las_to_train_map_path: str | Path,
    ignore_las_classes_path: str | Path,
    num_classes: int,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    """Compute class frequencies from LAS labels for weighting loss."""
    signature_src = (
        Path(split_file).read_text(encoding="utf-8")
        + Path(las_to_train_map_path).read_text(encoding="utf-8-sig")
        + Path(ignore_las_classes_path).read_text(encoding="utf-8-sig")
        + str(num_classes)
    )
    signature = hashlib.md5(signature_src.encode("utf-8")).hexdigest()
    if cache_path is not None:
        cpath = Path(cache_path)
        if cpath.exists():
            try:
                payload = json.loads(cpath.read_text(encoding="utf-8"))
                if payload.get("signature") == signature:
                    vals = np.asarray(payload.get("hist", []), dtype=np.int64)
                    if vals.shape[0] == num_classes:
                        return vals
            except Exception:
                pass

    lines = Path(split_file).read_text(encoding="utf-8").splitlines()
    las_to_train = json.loads(Path(las_to_train_map_path).read_text(encoding="utf-8-sig"))
    las_to_train = {int(k): int(v) for k, v in las_to_train.items()}
    ignore_set = {int(v) for v in json.loads(Path(ignore_las_classes_path).read_text(encoding="utf-8-sig"))}

    hist = np.zeros((num_classes,), dtype=np.int64)

    for line in lines:
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue

        path = resolve_path(clean, data_root)
        try:
            cls = read_las_points(path)["classification"].astype(np.int32)
        except Exception:
            continue

        keep = np.ones_like(cls, dtype=bool)
        for c in ignore_set:
            keep &= cls != c
        cls = cls[keep]

        mapped = np.full_like(cls, -1, dtype=np.int64)
        for las_c, train_c in las_to_train.items():
            mapped[cls == las_c] = train_c
        mapped = mapped[mapped >= 0]

        if mapped.size:
            hist += np.bincount(mapped, minlength=num_classes)

    if cache_path is not None:
        cpath = Path(cache_path)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(
            json.dumps({"signature": signature, "hist": hist.tolist()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return hist


def get_input_channels(
    use_hag: bool,
    use_return_features: bool,
    use_scan_angle: bool,
    use_normal_features: bool = False,
    use_roughness_feature: bool = False,
    use_slope_feature: bool = False,
    use_planarity_feature: bool = False,
    use_linearity_feature: bool = False,
) -> int:
    channels = 2  # z_rel, intensity_norm
    if use_hag:
        channels += 1
    if use_return_features:
        channels += 2  # return_ratio, num_returns_norm
    if use_scan_angle:
        channels += 1
    if use_normal_features:
        channels += 1
    if use_roughness_feature:
        channels += 1
    if use_slope_feature:
        channels += 1
    if use_planarity_feature:
        channels += 1
    if use_linearity_feature:
        channels += 1
    return channels
