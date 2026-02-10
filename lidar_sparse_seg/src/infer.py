from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.datasets.lidar_dataset import get_input_channels
from src.models.minkunet import SparseUNet
from src.utils.geom_features import compute_local_normal_features
from src.utils.io_las import read_las_points, write_las_with_classification
from src.utils.spconv_utils import build_spconv_tensor
from src.utils.voxelize import voxelize_points


PREFERRED_LAS_FOR_OUTPUT = [2, 6, 3, 4, 5, 13]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for LiDAR semantic segmentation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="LAS/LAZ file or folder")
    parser.add_argument("--output", type=str, required=True, help="Output LAS/LAZ file or folder")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--block_size", type=float, default=128.0, help="XY block size in meters")
    parser.add_argument("--max_points_per_block", type=int, default=400000)
    return parser.parse_args()


def build_train_to_las_map(las_to_train: dict[int, int], num_classes: int) -> dict[int, int]:
    grouped: dict[int, list[int]] = {i: [] for i in range(num_classes)}
    for las_cls, train_cls in las_to_train.items():
        if train_cls in grouped:
            grouped[train_cls].append(las_cls)

    train_to_las: dict[int, int] = {}
    for train_cls in range(num_classes):
        candidates = grouped.get(train_cls, [])
        if not candidates:
            continue

        choice = None
        for pref in PREFERRED_LAS_FOR_OUTPUT:
            if pref in candidates:
                choice = pref
                break
        if choice is None:
            choice = min(candidates)
        train_to_las[train_cls] = int(choice)
    return train_to_las


def compute_hag(xyz: np.ndarray, las_cls: np.ndarray, cell_size: float) -> np.ndarray:
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

    gx = np.floor((x - x0) / cell_size).astype(np.int32)
    gy = np.floor((y - y0) / cell_size).astype(np.int32)

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


def normalize_return_features(return_number: np.ndarray, number_of_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    denom = np.maximum(number_of_returns, 1.0)
    return_ratio = np.clip(return_number / denom, 0.0, 1.0).astype(np.float32)
    num_returns_norm = np.clip(number_of_returns / 8.0, 0.0, 1.0).astype(np.float32)
    return return_ratio, num_returns_norm


def normalize_scan_angle(scan_angle: np.ndarray) -> np.ndarray:
    return np.clip((scan_angle + 90.0) / 180.0, 0.0, 1.0).astype(np.float32)


def iter_blocks(xy: np.ndarray, block_size: float) -> list[np.ndarray]:
    if xy.shape[0] == 0:
        return []

    x_min, y_min = np.min(xy, axis=0)
    x_max, y_max = np.max(xy, axis=0)

    xs = np.arange(x_min, x_max + block_size, block_size)
    ys = np.arange(y_min, y_max + block_size, block_size)

    blocks: list[np.ndarray] = []
    for x0 in xs:
        for y0 in ys:
            mask = (
                (xy[:, 0] >= x0)
                & (xy[:, 0] < x0 + block_size)
                & (xy[:, 1] >= y0)
                & (xy[:, 1] < y0 + block_size)
            )
            if np.any(mask):
                blocks.append(mask)
    return blocks


def infer_file(
    model: SparseUNet,
    path_in: Path,
    path_out: Path,
    cfg: dict,
    device: torch.device,
    block_size: float,
    max_points_per_block: int,
) -> None:
    pts = read_las_points(path_in)
    n_total = pts["classification"].shape[0]
    if n_total == 0:
        write_las_with_classification(path_in, path_out, pts["classification"])
        return

    xyz_all = np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float64)
    intensity_all = pts["intensity"].astype(np.float32)
    cls_all = pts["classification"].astype(np.int32)
    return_number_all = pts["return_number"].astype(np.float32)
    number_of_returns_all = pts["number_of_returns"].astype(np.float32)
    scan_angle_all = pts["scan_angle"].astype(np.float32)

    ignore_set = {int(v) for v in json.loads(Path(cfg["paths"]["ignore_las_classes"]).read_text(encoding="utf-8-sig"))}
    las_to_train = {
        int(k): int(v)
        for k, v in json.loads(Path(cfg["paths"]["las_to_train"]).read_text(encoding="utf-8-sig")).items()
    }

    valid = np.ones((n_total,), dtype=bool)
    for c in ignore_set:
        valid &= cls_all != c
    valid &= np.isin(cls_all, list(las_to_train.keys()))
    non_ignored = np.ones((n_total,), dtype=bool)
    for c in ignore_set:
        non_ignored &= cls_all != c

    pred_cls_las = cls_all.copy().astype(np.uint8)
    if not np.any(valid):
        write_las_with_classification(path_in, path_out, pred_cls_las)
        return

    valid_idx = np.where(valid)[0]
    non_ignored_idx = np.where(non_ignored)[0]
    xyz_non_ignored = xyz_all[non_ignored]
    cls_non_ignored = cls_all[non_ignored]
    blocks = iter_blocks(xyz_non_ignored[:, :2], block_size)
    train_pred = np.full((valid_idx.shape[0],), -1, dtype=np.int64)
    valid_global_to_local = {int(g): i for i, g in enumerate(valid_idx.tolist())}

    model.eval()
    with torch.no_grad():
        for mask in tqdm(blocks, desc=f"infer {path_in.name}", leave=False, ncols=100):
            idx_block_non_ignored = np.where(mask)[0]
            if idx_block_non_ignored.size == 0:
                continue

            idx_block_global = non_ignored_idx[idx_block_non_ignored]
            block_valid_mask = valid[idx_block_global]
            if not np.any(block_valid_mask):
                continue

            valid_block_global = idx_block_global[block_valid_mask]
            if valid_block_global.size > max_points_per_block:
                valid_block_global = valid_block_global[:max_points_per_block]

            xyz_valid_block = xyz_all[valid_block_global].copy()
            intensity = intensity_all[valid_block_global]
            cls_valid_block = cls_all[valid_block_global]
            return_number = return_number_all[valid_block_global]
            number_of_returns = number_of_returns_all[valid_block_global]
            scan_angle = scan_angle_all[valid_block_global]

            # Keep HAG references from all non-ignored points in this spatial block.
            xyz_block_all = xyz_all[idx_block_global].copy()
            cls_block_all = cls_all[idx_block_global]

            xy_mean = np.mean(xyz_valid_block[:, :2], axis=0, keepdims=True)
            xyz_valid_block[:, :2] = xyz_valid_block[:, :2] - xy_mean
            xyz_block_all[:, :2] = xyz_block_all[:, :2] - xy_mean
            z_min = np.min(xyz_valid_block[:, 2])
            z_rel = (xyz_valid_block[:, 2] - z_min).astype(np.float32)
            xyz_block_all[:, 2] = xyz_block_all[:, 2] - z_min
            intensity_norm = np.clip(intensity / max(float(cfg["max_intensity"]), 1.0), 0.0, 1.0).astype(np.float32)

            feat_list = [z_rel, intensity_norm]
            if bool(cfg["use_hag"]):
                hag_all = compute_hag(xyz_block_all, cls_block_all, float(cfg["hag_cell_size"]))
                valid_set = set(valid_block_global.tolist())
                valid_pos = np.array([i for i, g in enumerate(idx_block_global.tolist()) if g in valid_set], dtype=np.int64)
                feat_list.append(hag_all[valid_pos].astype(np.float32))
            if bool(cfg.get("use_return_features", False)):
                return_ratio, num_returns_norm = normalize_return_features(return_number, number_of_returns)
                feat_list.append(return_ratio)
                feat_list.append(num_returns_norm)
            if bool(cfg.get("use_scan_angle", False)):
                feat_list.append(normalize_scan_angle(scan_angle))
            if bool(cfg.get("use_normal_features", False)) or bool(cfg.get("use_roughness_feature", False)):
                normal_z, roughness = compute_local_normal_features(
                    xyz=xyz_valid_block,
                    cell_size=float(cfg.get("normal_cell_size", 1.0)),
                    min_points=int(cfg.get("normal_min_points", 6)),
                )
                if bool(cfg.get("use_normal_features", False)):
                    feat_list.append(normal_z)
                if bool(cfg.get("use_roughness_feature", False)):
                    roughness_norm = np.clip(
                        roughness / max(float(cfg.get("roughness_scale", 1.0)), 1e-6),
                        0.0,
                        1.0,
                    ).astype(np.float32)
                    feat_list.append(roughness_norm)

            features = np.stack(feat_list, axis=1).astype(np.float32)
            vox = voxelize_points(
                xyz=xyz,
                features=features,
                labels=None,
                voxel_size=float(cfg["voxel_size"]),
                num_classes=int(cfg["num_classes"]),
            )

            if vox["coords"].shape[0] == 0:
                continue

            bcol = np.zeros((vox["coords"].shape[0], 1), dtype=np.int32)
            coords = np.concatenate([bcol, vox["coords"]], axis=1)

            stensor = build_spconv_tensor(
                features=torch.from_numpy(vox["features"]).float().to(device),
                coordinates=torch.from_numpy(coords).int().to(device),
            )
            logits = model(stensor)
            pred_vox = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)

            pred_pts = pred_vox[vox["inverse"]]
            local_valid_idx = np.array([valid_global_to_local[int(g)] for g in valid_block_global], dtype=np.int64)
            train_pred[local_valid_idx] = pred_pts

    train_to_las = build_train_to_las_map(las_to_train, int(cfg["num_classes"]))

    unresolved = train_pred < 0
    if np.any(unresolved):
        mapped_train_labels = np.array(sorted(train_to_las.keys()), dtype=np.int64)
        fallback = np.random.choice(mapped_train_labels, size=unresolved.sum())
        train_pred[unresolved] = fallback

    pred_las_valid = np.array([train_to_las[int(v)] for v in train_pred], dtype=np.uint8)
    pred_cls_las[valid_idx] = pred_las_valid

    write_las_with_classification(path_in, path_out, pred_cls_las)


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.las")) + sorted(path.glob("*.laz"))
    return files


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise RuntimeError(
            "spconv requer CUDA. Torch CPU-only detectado. "
            "Instale uma build CUDA do PyTorch e execute em GPU."
        )
    in_channels = get_input_channels(
        use_hag=bool(cfg["use_hag"]),
        use_return_features=bool(cfg.get("use_return_features", False)),
        use_scan_angle=bool(cfg.get("use_scan_angle", False)),
        use_normal_features=bool(cfg.get("use_normal_features", False)),
        use_roughness_feature=bool(cfg.get("use_roughness_feature", False)),
    )

    model = SparseUNet(
        in_channels=in_channels,
        num_classes=int(cfg["num_classes"]),
        base_channels=int(cfg["model"]["base_channels"]),
        depth=int(cfg["model"]["depth"]),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    input_path = Path(args.input)
    output_path = Path(args.output)

    in_files = collect_inputs(input_path)
    if not in_files:
        raise FileNotFoundError(f"No LAS/LAZ found in {input_path}")

    for f in tqdm(in_files, desc="files", ncols=100):
        if input_path.is_file():
            out_f = output_path
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            out_f = output_path / f.name

        infer_file(
            model=model,
            path_in=f,
            path_out=out_f,
            cfg=cfg,
            device=device,
            block_size=float(args.block_size),
            max_points_per_block=int(args.max_points_per_block),
        )


if __name__ == "__main__":
    main()
