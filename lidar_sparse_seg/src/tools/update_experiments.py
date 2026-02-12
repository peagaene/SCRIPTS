from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append one experiment row to TEMPLATE_EXPERIMENTOS.csv")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--template", type=str, default="TEMPLATE_EXPERIMENTOS.csv")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--checkpoints_dir", type=str, default=None)
    parser.add_argument("--experiment_id", type=str, default=None)
    parser.add_argument("--dataset_tag", type=str, default="")
    parser.add_argument("--treino_tipo", type=str, default="")
    parser.add_argument("--checkpoint_base", type=str, default="best_mIoU.pth")
    parser.add_argument("--responsavel", type=str, default="")
    parser.add_argument("--observacoes", type=str, default="")
    parser.add_argument("--decisao", type=str, default="")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _class_idx(class_names: list[str], key: str) -> int | None:
    key = key.lower()
    for i, name in enumerate(class_names):
        if key in str(name).lower():
            return i
    return None


def _safe_list_get(values: list, idx: int | None):
    if idx is None:
        return ""
    if idx < 0 or idx >= len(values):
        return ""
    return values[idx]


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    checkpoints_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else Path(cfg["paths"]["checkpoints_dir"])
    reports_dir = checkpoints_dir / "reports"
    logs_dir = Path(cfg["paths"]["logs_dir"])

    metrics_val = _read_json(reports_dir / "metrics_val.json")
    metrics_test = _read_json(reports_dir / "metrics_test.json")
    history = _read_history(logs_dir / "history.csv")

    class_names = metrics_test.get("class_names") or metrics_val.get("class_names") or []
    edif_idx = _class_idx(class_names, "edific")
    terr_idx = _class_idx(class_names, "terreno")
    veg_idx = _class_idx(class_names, "veget")

    now = dt.datetime.now().isoformat(timespec="seconds")
    experiment_id = args.experiment_id or f"{checkpoints_dir.name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    train_time_h = 0.0
    if history:
        train_time_h = sum(_to_float(r.get("epoch_time_sec", 0.0)) for r in history) / 3600.0

    val_miou = metrics_val.get("mIoU", "")
    test_miou = metrics_test.get("mIoU", "")

    val_iou = metrics_val.get("iou_per_class", [])
    test_iou = metrics_test.get("iou_per_class", [])
    val_f1 = metrics_val.get("f1_per_class", [])
    test_f1 = metrics_test.get("f1_per_class", [])

    row = {
        "experimento_id": experiment_id,
        "data_hora": now,
        "dataset_tag": args.dataset_tag,
        "config_path": str(config_path).replace("\\", "/"),
        "checkpoint_base": args.checkpoint_base,
        "treino_tipo": args.treino_tipo,
        "split_seed": cfg.get("split", {}).get("seed", ""),
        "split_regenerate": cfg.get("split", {}).get("regenerate", ""),
        "voxel_size": cfg.get("voxel_size", ""),
        "crop_size": cfg.get("crop_size", ""),
        "batch_size": cfg.get("batch_size", ""),
        "epochs": cfg.get("epochs", ""),
        "lr": cfg.get("lr", ""),
        "scheduler": cfg.get("scheduler", ""),
        "use_hag": cfg.get("use_hag", ""),
        "use_return_features": cfg.get("use_return_features", ""),
        "use_scan_angle": cfg.get("use_scan_angle", ""),
        "use_normal_features": cfg.get("use_normal_features", ""),
        "use_roughness_feature": cfg.get("use_roughness_feature", ""),
        "normal_cell_size": cfg.get("normal_cell_size", ""),
        "roughness_scale": cfg.get("roughness_scale", ""),
        "base_channels": cfg.get("model", {}).get("base_channels", ""),
        "depth": cfg.get("model", {}).get("depth", ""),
        "val_mIoU": val_miou,
        "test_mIoU": test_miou,
        "val_IoU_edificacao": _safe_list_get(val_iou, edif_idx),
        "test_IoU_edificacao": _safe_list_get(test_iou, edif_idx),
        "val_IoU_terreno": _safe_list_get(val_iou, terr_idx),
        "test_IoU_terreno": _safe_list_get(test_iou, terr_idx),
        "val_IoU_vegetacao": _safe_list_get(val_iou, veg_idx),
        "test_IoU_vegetacao": _safe_list_get(test_iou, veg_idx),
        "val_F1_edificacao": _safe_list_get(val_f1, edif_idx),
        "test_F1_edificacao": _safe_list_get(test_f1, edif_idx),
        "conf_edif_para_veg_val": metrics_val.get("confusion_edificacao_para_vegetacao_rate", ""),
        "conf_edif_para_veg_test": metrics_test.get("confusion_edificacao_para_vegetacao_rate", ""),
        "fragmentacao_edificacao_score": "",
        "boundary_score_proxy": "",
        "tempo_treino_h": round(train_time_h, 4),
        "tempo_validacao_min": "",
        "qualidade_visual_geral": "",
        "qualidade_visual_edificacao": "",
        "qualidade_visual_terreno": "",
        "qualidade_visual_vegetacao": "",
        "tempo_revisao_estimado_min_por_tile": "",
        "principais_erros_observados": "",
        "decisao": args.decisao,
        "responsavel": args.responsavel,
        "observacoes": args.observacoes,
    }

    template_path = Path(args.template)
    output_path = Path(args.output) if args.output else template_path

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with template_path.open("r", encoding="utf-8") as f:
        header_line = f.readline().strip()
    fieldnames = [h.strip() for h in header_line.split(";") if h.strip()]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Row appended to {output_path}")


if __name__ == "__main__":
    main()
