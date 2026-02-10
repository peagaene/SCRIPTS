from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def confusion_matrix_np(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    mask = (target >= 0) & (target < num_classes)
    pred = pred[mask]
    target = target[mask]
    cm = np.bincount(num_classes * target + pred, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def compute_metrics_from_confusion(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(np.float64)
    gt = cm.sum(axis=1).astype(np.float64)
    pred = cm.sum(axis=0).astype(np.float64)

    union = gt + pred - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, gt, out=np.zeros_like(tp), where=gt > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    acc = np.divide(tp.sum(), cm.sum(), out=np.array(0.0), where=cm.sum() > 0).item()
    miou = iou.mean().item() if iou.size else 0.0

    return {
        "overall_accuracy": float(acc),
        "iou_per_class": iou.tolist(),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "mIoU": float(miou),
    }


def save_metrics_json(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_confusion_csv(path: str | Path, cm: np.ndarray, class_names: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = ",".join(["gt/pred"] + class_names)
    lines = [header]
    for i, row in enumerate(cm):
        name = class_names[i] if i < len(class_names) else str(i)
        lines.append(",".join([name] + [str(int(v)) for v in row.tolist()]))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_epoch_history_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row.get(k, "")) for k in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
