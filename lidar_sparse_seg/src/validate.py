from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.lidar_dataset import LidarSemanticDataset, get_input_channels
from src.models.spconv_unet import SparseUNet
from src.utils.metrics import (
    compute_metrics_from_confusion,
    confusion_matrix_np,
    save_confusion_csv,
    save_metrics_json,
)
from src.utils.splits import ensure_or_generate_splits
from src.utils.spconv_utils import build_spconv_tensor
from src.utils.voxelize import sparse_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate sparse LiDAR semantic segmentation model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None, help="Evaluate only first N batches (benchmark mode)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _find_class_idx(class_names: list[str], key: str) -> int | None:
    key = key.lower()
    for i, name in enumerate(class_names):
        if key in name.lower():
            return i
    return None


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    max_batches = int(args.max_batches) if args.max_batches is not None else int(cfg.get("validate_max_batches", 0))
    split_info = ensure_or_generate_splits(cfg)
    if split_info["generated"]:
        print(
            "Auto split generated: "
            f"train={split_info['train_count']} val={split_info['val_count']} test={split_info['test_count']}"
        )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise RuntimeError(
            "spconv requer CUDA. Torch CPU-only detectado. "
            "Instale uma build CUDA do PyTorch e execute em GPU."
        )
    num_classes = int(cfg["num_classes"])
    use_hag = bool(cfg["use_hag"])
    use_return_features = bool(cfg.get("use_return_features", False))
    use_scan_angle = bool(cfg.get("use_scan_angle", False))
    use_normal_features = bool(cfg.get("use_normal_features", False))
    use_roughness_feature = bool(cfg.get("use_roughness_feature", False))
    use_slope_feature = bool(cfg.get("use_slope_feature", False))
    use_planarity_feature = bool(cfg.get("use_planarity_feature", False))
    use_linearity_feature = bool(cfg.get("use_linearity_feature", False))

    split_key = "val_split" if args.split == "val" else "test_split"
    ds = LidarSemanticDataset(
        split_file=cfg["paths"][split_key],
        data_root=cfg["paths"]["data_root"],
        las_to_train_map_path=cfg["paths"]["las_to_train"],
        ignore_las_classes_path=cfg["paths"]["ignore_las_classes"],
        voxel_size=cfg["voxel_size"],
        num_classes=num_classes,
        max_intensity=cfg["max_intensity"],
        crop_size=cfg["crop_size"],
        max_points_per_crop=cfg["max_points_per_crop"],
        use_hag=use_hag,
        hag_cell_size=cfg["hag_cell_size"],
        use_return_features=use_return_features,
        use_scan_angle=use_scan_angle,
        use_normal_features=use_normal_features,
        use_roughness_feature=use_roughness_feature,
        use_slope_feature=use_slope_feature,
        use_planarity_feature=use_planarity_feature,
        use_linearity_feature=use_linearity_feature,
        normal_cell_size=float(cfg.get("normal_cell_size", 1.0)),
        normal_min_points=int(cfg.get("normal_min_points", 6)),
        roughness_scale=float(cfg.get("roughness_scale", 1.0)),
        tile_cache_size=int(cfg.get("tile_cache_size", 4)),
        cache_full_samples=bool(cfg.get("cache_full_samples_val", True)),
        mode=args.split,
    )

    persistent_workers = int(cfg["num_workers"]) > 0
    loader = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers,
        collate_fn=sparse_collate_fn,
    )

    in_channels = get_input_channels(
        use_hag=use_hag,
        use_return_features=use_return_features,
        use_scan_angle=use_scan_angle,
        use_normal_features=use_normal_features,
        use_roughness_feature=use_roughness_feature,
        use_slope_feature=use_slope_feature,
        use_planarity_feature=use_planarity_feature,
        use_linearity_feature=use_linearity_feature,
    )
    model = SparseUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=int(cfg["model"]["base_channels"]),
        depth=int(cfg["model"]["depth"]),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("=" * 90)
    print(f"Validation start | split={args.split} | device={device} | amp={bool(cfg['amp']) and device.type == 'cuda'}")
    print(f"checkpoint={args.checkpoint}")
    print(
        f"files={len(ds)} batch_size={cfg['batch_size']} voxel_size={cfg['voxel_size']} "
        f"use_hag={cfg['use_hag']} use_return_features={use_return_features} use_scan_angle={use_scan_angle} "
        f"use_normal_features={use_normal_features} use_roughness_feature={use_roughness_feature} "
        f"use_slope_feature={use_slope_feature} use_planarity_feature={use_planarity_feature} "
        f"use_linearity_feature={use_linearity_feature}"
    )
    print("=" * 90)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    amp_enabled = bool(cfg["amp"]) and device.type == "cuda"
    evaluated_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"validating({args.split})", ncols=110), start=1):
            if batch["coords"].shape[0] == 0:
                continue

            coords = batch["coords"].to(device)
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)

            x = build_spconv_tensor(features=feats, coordinates=coords)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(x)

            pred = torch.argmax(logits, dim=1)
            cm += confusion_matrix_np(pred.cpu().numpy(), labels.cpu().numpy(), num_classes)
            evaluated_batches += 1
            if max_batches > 0 and batch_idx >= max_batches:
                break

    metrics = compute_metrics_from_confusion(cm)

    class_map = json.loads(Path(cfg["paths"]["classes"]).read_text(encoding="utf-8-sig"))
    class_names = [class_map[str(i)] for i in range(num_classes)]
    metrics["class_names"] = class_names
    metrics["confusion_matrix"] = cm.tolist()
    metrics["evaluated_batches"] = int(evaluated_batches)
    edif_idx = _find_class_idx(class_names, "edific")
    veg_idx = _find_class_idx(class_names, "veget")
    if edif_idx is not None and veg_idx is not None:
        gt_edif = float(cm[edif_idx, :].sum())
        gt_veg = float(cm[veg_idx, :].sum())
        metrics["confusion_edificacao_para_vegetacao_rate"] = (
            float(cm[edif_idx, veg_idx] / gt_edif) if gt_edif > 0 else 0.0
        )
        metrics["confusion_vegetacao_para_edificacao_rate"] = (
            float(cm[veg_idx, edif_idx] / gt_veg) if gt_veg > 0 else 0.0
        )

    default_output_dir = Path(cfg["paths"]["checkpoints_dir"]) / "reports"
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    save_metrics_json(output_dir / f"metrics_{args.split}.json", metrics)
    save_confusion_csv(output_dir / f"confusion_{args.split}.csv", cm, class_names)

    summary_csv = output_dir / f"summary_{args.split}.csv"
    lines = ["metric,value", f"overall_accuracy,{metrics['overall_accuracy']}", f"mIoU,{metrics['mIoU']}"]
    lines.append(f"evaluated_batches,{metrics['evaluated_batches']}")
    for i, iou in enumerate(metrics["iou_per_class"]):
        lines.append(f"IoU_{class_names[i]},{iou}")
    for i, f1 in enumerate(metrics["f1_per_class"]):
        lines.append(f"F1_{class_names[i]},{f1}")
    if "confusion_edificacao_para_vegetacao_rate" in metrics:
        lines.append(
            f"confusion_edificacao_para_vegetacao_rate,{metrics['confusion_edificacao_para_vegetacao_rate']}"
        )
    if "confusion_vegetacao_para_edificacao_rate" in metrics:
        lines.append(
            f"confusion_vegetacao_para_edificacao_rate,{metrics['confusion_vegetacao_para_edificacao_rate']}"
        )
    summary_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"overall_accuracy={metrics['overall_accuracy']:.6f}")
    print(f"mIoU={metrics['mIoU']:.6f}")
    print(f"evaluated_batches={metrics['evaluated_batches']}")
    for i, name in enumerate(class_names):
        print(
            f"IoU[{i}:{name}]={metrics['iou_per_class'][i]:.6f} "
            f"F1={metrics['f1_per_class'][i]:.6f}"
        )
    if "confusion_edificacao_para_vegetacao_rate" in metrics:
        print(
            "conf_edif->veg="
            f"{metrics['confusion_edificacao_para_vegetacao_rate']:.6f} "
            "conf_veg->edif="
            f"{metrics['confusion_vegetacao_para_edificacao_rate']:.6f}"
        )
    print(f"reports_dir={output_dir}")


if __name__ == "__main__":
    main()
