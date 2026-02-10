from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets.lidar_dataset import LidarSemanticDataset, compute_class_histogram, get_input_channels
from src.models.minkunet import SparseUNet
from src.utils.metrics import compute_metrics_from_confusion, confusion_matrix_np, save_epoch_history_csv
from src.utils.repro import set_seed
from src.utils.splits import ensure_or_generate_splits
from src.utils.spconv_utils import build_spconv_tensor
from src.utils.voxelize import sparse_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sparse LiDAR semantic segmentation with spconv")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--use_hag", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="Load only model weights from checkpoint and reset optimizer/scheduler/best metric.",
    )
    return parser.parse_args()


def load_config(path: str | Path, args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.voxel_size is not None:
        cfg["voxel_size"] = args.voxel_size
    if args.use_hag:
        cfg["use_hag"] = True
    return cfg


def compute_class_weights(hist: np.ndarray) -> np.ndarray:
    hist = hist.astype(np.float64)
    hist = np.maximum(hist, 1.0)
    inv = 1.0 / hist
    w = inv / np.mean(inv)
    return w.astype(np.float32)


def print_run_header(cfg: dict, device: torch.device, class_names: list[str], train_count: int, val_count: int) -> None:
    print("=" * 90)
    print("Sparse LiDAR Segmentation - Training")
    print(f"device={device} | amp={bool(cfg['amp']) and device.type == 'cuda'} | seed={cfg['seed']}")
    print(
        f"epochs={cfg['epochs']} batch_size={cfg['batch_size']} lr={cfg['lr']} "
        f"voxel_size={cfg['voxel_size']} scheduler={cfg['scheduler']}"
    )
    print(
        f"crop_size={cfg['crop_size']} max_points_per_crop={cfg['max_points_per_crop']} "
        f"use_hag={cfg['use_hag']}"
    )
    print(
        f"use_return_features={bool(cfg.get('use_return_features', False))} "
        f"use_scan_angle={bool(cfg.get('use_scan_angle', False))} "
        f"use_normal_features={bool(cfg.get('use_normal_features', False))} "
        f"use_roughness_feature={bool(cfg.get('use_roughness_feature', False))}"
    )
    print(f"train_files={train_count} val_files={val_count}")
    print(f"classes={class_names}")
    print("=" * 90)


def save_class_stats(path: Path, class_names: list[str], hist: np.ndarray, weights: np.ndarray) -> None:
    lines = ["class_id,class_name,count,weight"]
    for i, (name, count, weight) in enumerate(zip(class_names, hist.tolist(), weights.tolist())):
        lines.append(f"{i},{name},{count},{weight}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(model, loader, device, num_classes: int, amp: bool) -> tuple[float, dict]:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    val_loss_sum = 0.0
    val_loss_count = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            if batch["coords"].shape[0] == 0:
                continue

            coords = batch["coords"].to(device)
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)

            x = build_spconv_tensor(features=feats, coordinates=coords)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, labels)

            pred = torch.argmax(logits, dim=1)
            cm += confusion_matrix_np(pred.cpu().numpy(), labels.cpu().numpy(), num_classes)
            val_loss_sum += float(loss.item())
            val_loss_count += 1

    metrics = compute_metrics_from_confusion(cm)
    val_loss = val_loss_sum / max(val_loss_count, 1)
    return val_loss, metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args)

    set_seed(int(cfg["seed"]))
    split_info = ensure_or_generate_splits(cfg)
    if split_info["generated"]:
        print(
            "Auto split generated: "
            f"train={split_info['train_count']} val={split_info['val_count']} test={split_info['test_count']}"
        )
    else:
        print(
            "Using existing split files: "
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

    checkpoints_dir = Path(cfg["paths"]["checkpoints_dir"])
    logs_dir = Path(cfg["paths"]["logs_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = checkpoints_dir / "run_artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    with Path(cfg["paths"]["classes"]).open("r", encoding="utf-8-sig") as f:
        class_map = json.load(f)
    class_names = [class_map[str(i)] for i in range(num_classes)]

    train_ds = LidarSemanticDataset(
        split_file=cfg["paths"]["train_split"],
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
        normal_cell_size=float(cfg.get("normal_cell_size", 1.0)),
        normal_min_points=int(cfg.get("normal_min_points", 6)),
        roughness_scale=float(cfg.get("roughness_scale", 1.0)),
        mode="train",
    )

    val_ds = LidarSemanticDataset(
        split_file=cfg["paths"]["val_split"],
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
        normal_cell_size=float(cfg.get("normal_cell_size", 1.0)),
        normal_min_points=int(cfg.get("normal_min_points", 6)),
        roughness_scale=float(cfg.get("roughness_scale", 1.0)),
        mode="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=sparse_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=sparse_collate_fn,
    )

    hist = compute_class_histogram(
        split_file=cfg["paths"]["train_split"],
        data_root=cfg["paths"]["data_root"],
        las_to_train_map_path=cfg["paths"]["las_to_train"],
        ignore_las_classes_path=cfg["paths"]["ignore_las_classes"],
        num_classes=num_classes,
    )
    class_weights = compute_class_weights(hist)
    save_class_stats(run_dir / "class_stats.csv", class_names, hist, class_weights)

    cfg_snapshot = run_dir / "config_resolved.yaml"
    cfg_snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    in_channels = get_input_channels(
        use_hag=use_hag,
        use_return_features=use_return_features,
        use_scan_angle=use_scan_angle,
        use_normal_features=use_normal_features,
        use_roughness_feature=use_roughness_feature,
    )
    model = SparseUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=int(cfg["model"]["base_channels"]),
        depth=int(cfg["model"]["depth"]),
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    total_epochs = int(cfg["epochs"])
    if str(cfg["scheduler"]).lower() == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=float(cfg["lr"]),
            epochs=total_epochs,
            steps_per_epoch=max(len(train_loader), 1),
        )
        scheduler_step_per_batch = True
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
        scheduler_step_per_batch = False

    amp_enabled = bool(cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    writer = SummaryWriter(log_dir=str(logs_dir / "tb"))
    history: list[dict] = []

    print_run_header(cfg, device, class_names, len(train_ds), len(val_ds))
    print("Class frequency and loss weights:")
    for idx, name in enumerate(class_names):
        print(f"  class={idx} name={name:<18} count={int(hist[idx]):>10} weight={float(class_weights[idx]):.6f}")
    print(f"Artifacts saved to: {run_dir}")

    best_miou = -1.0
    start_epoch = 0

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

        print(f"Resuming training from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

        if args.resume_weights_only:
            best_miou = -1.0
            start_epoch = 0
            print("Resume mode: weights-only (optimizer/scheduler reset).")
        else:
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            if "scaler_state" in ckpt and amp_enabled:
                scaler.load_state_dict(ckpt["scaler_state"])

            best_miou = float(ckpt.get("best_miou", -1.0))
            start_epoch = int(ckpt.get("epoch", 0))
            print(f"Resume mode: full-state. start_epoch={start_epoch + 1}, best_mIoU={best_miou:.5f}")

    for epoch in range(start_epoch, total_epochs):
        epoch_start = perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{total_epochs}", ncols=110)
        for batch in pbar:
            if batch["coords"].shape[0] == 0:
                continue

            coords = batch["coords"].to(device)
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            x = build_spconv_tensor(features=feats, coordinates=coords)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler_step_per_batch:
                scheduler.step()

            train_loss_sum += float(loss.item())
            train_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if not scheduler_step_per_batch:
            scheduler.step()

        train_loss = train_loss_sum / max(train_steps, 1)
        val_loss, val_metrics = evaluate(model, val_loader, device, num_classes, amp_enabled)
        epoch_time = perf_counter() - epoch_start

        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_metrics["overall_accuracy"],
            "val_miou": val_metrics["mIoU"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        for cidx, iou in enumerate(val_metrics["iou_per_class"]):
            epoch_row[f"iou_{cidx}"] = iou

        history.append(epoch_row)

        writer.add_scalar("loss/train", train_loss, epoch + 1)
        writer.add_scalar("loss/val", val_loss, epoch + 1)
        writer.add_scalar("metrics/overall_acc", val_metrics["overall_accuracy"], epoch + 1)
        writer.add_scalar("metrics/mIoU", val_metrics["mIoU"], epoch + 1)
        for cidx, cname in enumerate(class_names):
            writer.add_scalar(f"iou/{cname}", val_metrics["iou_per_class"][cidx], epoch + 1)

        last_ckpt = checkpoints_dir / "last.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_miou": best_miou,
                "config": cfg,
            },
            last_ckpt,
        )

        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            best_ckpt = checkpoints_dir / "best_mIoU.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_miou": best_miou,
                    "config": cfg,
                },
                best_ckpt,
            )

        save_epoch_history_csv(logs_dir / "history.csv", history)

        print(
            f"[epoch {epoch + 1:03d}/{total_epochs:03d}] "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"acc={val_metrics['overall_accuracy']:.5f} mIoU={val_metrics['mIoU']:.5f} "
            f"lr={optimizer.param_groups[0]['lr']:.7f} time={epoch_time:.1f}s"
        )
        for cidx, cname in enumerate(class_names):
            print(f"    IoU[{cidx}:{cname}]={val_metrics['iou_per_class'][cidx]:.5f}")
        print(f"    best_mIoU={best_miou:.5f} | last={last_ckpt} | best={checkpoints_dir / 'best_mIoU.pth'}")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
