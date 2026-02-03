from __future__ import annotations

import os
import sys
import math
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import laspy
except Exception as exc:  # pragma: no cover - import guard for local env
    laspy = None
    _LASPY_IMPORT_ERROR = exc
else:
    _LASPY_IMPORT_ERROR = None

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception as exc:  # pragma: no cover - UI guard for local env
    tk = None
    filedialog = None
    messagebox = None
    _TK_IMPORT_ERROR = exc
else:
    _TK_IMPORT_ERROR = None


LAS_CLASS_NAMES = {
    0: "created_never_classified",
    1: "unclassified",
    2: "ground",
    3: "low_vegetation",
    4: "medium_vegetation",
    5: "high_vegetation",
    6: "building",
    7: "low_point",
    8: "model_key_point",
    9: "water",
    10: "rail",
    11: "road_surface",
    12: "overlap",
    13: "wire_guard",
    14: "wire_conductor",
    15: "transmission_tower",
    16: "wire_structure_connector",
    17: "bridge_deck",
    18: "high_noise",
}

VEG_CLASSES = (3, 4, 5)
BUILDING_CLASS = 6


@dataclass
class RunningStats:
    count: int = 0
    sum: float = 0.0
    sumsq: float = 0.0
    min: Optional[float] = None
    max: Optional[float] = None

    def update(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        arr = arr.astype(np.float64, copy=False)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.sum += float(arr.sum())
        self.sumsq += float((arr * arr).sum())
        mn = float(arr.min())
        mx = float(arr.max())
        self.min = mn if self.min is None else min(self.min, mn)
        self.max = mx if self.max is None else max(self.max, mx)

    @property
    def mean(self) -> Optional[float]:
        if self.count <= 0:
            return None
        return self.sum / self.count

    @property
    def std(self) -> Optional[float]:
        if self.count <= 1:
            return None
        mean = self.sum / self.count
        var = max(0.0, (self.sumsq / self.count) - (mean * mean))
        return math.sqrt(var)


def _safe_percentiles(arr: Optional[np.ndarray], qs: List[float]) -> Dict[str, Optional[float]]:
    if arr is None or arr.size == 0:
        return {f"p{int(q)}": None for q in qs}
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{int(q)}": None for q in qs}
    vals = np.percentile(arr, qs)
    return {f"p{int(q)}": float(v) for q, v in zip(qs, vals)}


def _merge_stats(stats_list: List[RunningStats]) -> RunningStats:
    out = RunningStats()
    for st in stats_list:
        if st.count <= 0:
            continue
        out.count += st.count
        out.sum += st.sum
        out.sumsq += st.sumsq
        if st.min is not None:
            out.min = st.min if out.min is None else min(out.min, st.min)
        if st.max is not None:
            out.max = st.max if out.max is None else max(out.max, st.max)
    return out


def _reservoir_concat(
    existing: Optional[np.ndarray], new: np.ndarray, max_n: int, rng: np.random.Generator
) -> np.ndarray:
    if existing is None or existing.size == 0:
        if new.size <= max_n:
            return new
        idx = rng.choice(np.arange(new.size), size=max_n, replace=False)
        return new[idx]
    total = existing.size + new.size
    if total <= max_n:
        return np.concatenate([existing, new])
    combined = np.concatenate([existing, new])
    idx = rng.choice(np.arange(combined.size), size=max_n, replace=False)
    return combined[idx]


def _best_threshold(sample_build: np.ndarray, sample_veg: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if sample_build.size < 100 or sample_veg.size < 100:
        return None, None, None
    sample_build = sample_build[np.isfinite(sample_build)]
    sample_veg = sample_veg[np.isfinite(sample_veg)]
    if sample_build.size < 100 or sample_veg.size < 100:
        return None, None, None

    x = np.concatenate([sample_build, sample_veg])
    if x.size < 200:
        return None, None, None

    direction = "gte" if np.nanmean(sample_build) >= np.nanmean(sample_veg) else "lte"
    qs = np.linspace(0.05, 0.95, 200)
    cand = np.quantile(x, qs)

    best_t = None
    best_score = -1.0
    for t in cand:
        if direction == "gte":
            pred_build = x >= t
        else:
            pred_build = x <= t
        tp = int(np.sum(pred_build[: sample_build.size]))
        fn = sample_build.size - tp
        tn = int(np.sum(~pred_build[sample_build.size :]))
        fp = sample_veg.size - tn
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        bal_acc = 0.5 * (tpr + tnr)
        if bal_acc > best_score:
            best_score = bal_acc
            best_t = float(t)
    return best_t, best_score if best_score >= 0.0 else None, direction


def _compute_grid_min_z(reader, grid_size: float, use_ground_only: bool) -> Dict[Tuple[int, int], float]:
    grid_min: Dict[Tuple[int, int], float] = {}
    for chunk in reader.chunk_iterator(2_000_000):
        cls = chunk.classification
        if use_ground_only:
            mask = cls == 2
            if not np.any(mask):
                continue
            xs = np.asarray(chunk.x[mask], dtype=np.float64)
            ys = np.asarray(chunk.y[mask], dtype=np.float64)
            zs = np.asarray(chunk.z[mask], dtype=np.float64)
        else:
            xs = np.asarray(chunk.x, dtype=np.float64)
            ys = np.asarray(chunk.y, dtype=np.float64)
            zs = np.asarray(chunk.z, dtype=np.float64)
        ix = np.floor(xs / grid_size).astype(np.int64)
        iy = np.floor(ys / grid_size).astype(np.int64)
        for kx, ky, z in zip(ix, iy, zs):
            key = (int(kx), int(ky))
            cur = grid_min.get(key)
            if cur is None or z < cur:
                grid_min[key] = float(z)
    return grid_min


def _grid_min_for_points(xs: np.ndarray, ys: np.ndarray, grid_min: Dict[Tuple[int, int], float], grid_size: float) -> np.ndarray:
    ix = np.floor(xs / grid_size).astype(np.int64)
    iy = np.floor(ys / grid_size).astype(np.int64)
    out = np.empty(xs.size, dtype=np.float64)
    for i, (kx, ky) in enumerate(zip(ix, iy)):
        out[i] = grid_min.get((int(kx), int(ky)), np.nan)
    return out


def _detect_dims(reader) -> List[str]:
    return list(reader.header.point_format.dimension_names)


def _build_feature_arrays(
    chunk,
    dims: List[str],
    grid_min: Optional[Dict[Tuple[int, int], float]],
    grid_size: float,
) -> Dict[str, np.ndarray]:
    feats: Dict[str, np.ndarray] = {}
    feats["z"] = np.asarray(chunk.z, dtype=np.float64)

    if "intensity" in dims:
        feats["intensity"] = np.asarray(chunk.intensity, dtype=np.float64)

    if "return_number" in dims and "number_of_returns" in dims:
        rn = np.asarray(chunk.return_number, dtype=np.float64)
        nr = np.asarray(chunk.number_of_returns, dtype=np.float64)
        feats["return_number"] = rn
        feats["number_of_returns"] = nr
        with np.errstate(divide="ignore", invalid="ignore"):
            feats["return_ratio"] = rn / nr
        feats["is_first"] = (rn == 1).astype(np.float64)
        feats["is_last"] = (rn == nr).astype(np.float64)

    if "scan_angle" in dims:
        ang = np.asarray(chunk.scan_angle, dtype=np.float64)
        feats["scan_angle"] = ang
        feats["scan_angle_abs"] = np.abs(ang)

    if "gps_time" in dims:
        feats["gps_time"] = np.asarray(chunk.gps_time, dtype=np.float64)

    if "user_data" in dims:
        feats["user_data"] = np.asarray(chunk.user_data, dtype=np.float64)

    if "point_source_id" in dims:
        feats["point_source_id"] = np.asarray(chunk.point_source_id, dtype=np.float64)

    has_rgb = all(d in dims for d in ("red", "green", "blue"))
    if has_rgb:
        r = np.asarray(chunk.red, dtype=np.float64)
        g = np.asarray(chunk.green, dtype=np.float64)
        b = np.asarray(chunk.blue, dtype=np.float64)
        feats["red"] = r
        feats["green"] = g
        feats["blue"] = b
        feats["brightness"] = (r + g + b) / 3.0

    if grid_min is not None:
        xs = np.asarray(chunk.x, dtype=np.float64)
        ys = np.asarray(chunk.y, dtype=np.float64)
        zs = np.asarray(chunk.z, dtype=np.float64)
        min_z = _grid_min_for_points(xs, ys, grid_min, grid_size)
        feats["height_norm"] = zs - min_z

    return feats


def _ensure_ui() -> None:
    if _TK_IMPORT_ERROR is not None:
        raise RuntimeError(f"Tkinter nao disponivel: {_TK_IMPORT_ERROR}")


def _ensure_laspy() -> None:
    if _LASPY_IMPORT_ERROR is not None:
        raise RuntimeError(f"laspy nao disponivel: {_LASPY_IMPORT_ERROR}")


def _pick_las_files() -> List[str]:
    _ensure_ui()
    root = tk.Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(
        title="Selecione arquivos LAS/LAZ",
        filetypes=[("LAS/LAZ", "*.las *.laz"), ("Todos", "*.*")],
    )
    root.destroy()
    return [str(p) for p in paths] if paths else []


def _pick_output_dir(initial_dir: str) -> str:
    _ensure_ui()
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title="Selecione pasta de saida", initialdir=initial_dir)
    root.destroy()
    return str(path) if path else ""


def _log(msg: str) -> None:
    print(msg, flush=True)


def _format_class_name(cls: int) -> str:
    return LAS_CLASS_NAMES.get(int(cls), "class_%d" % int(cls))


def _build_report(
    out_dir: str,
    report_base: str,
    header_rows: List[Dict[str, object]],
    class_rows: List[Dict[str, object]],
    compare_rows: List[Dict[str, object]],
    per_file_rows: List[Dict[str, object]],
    notes: List[str],
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, report_base + ".xlsx")

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        pd.DataFrame(header_rows).to_excel(writer, index=False, sheet_name="resumo")
        pd.DataFrame(class_rows).to_excel(writer, index=False, sheet_name="stats_por_classe")
        pd.DataFrame(compare_rows).to_excel(writer, index=False, sheet_name="comparacao_c6_vs_veg")
        pd.DataFrame(per_file_rows).to_excel(writer, index=False, sheet_name="contagem_por_arquivo")
        pd.DataFrame({"nota": notes}).to_excel(writer, index=False, sheet_name="observacoes")

    out_json = os.path.join(out_dir, report_base + ".json")
    payload = {
        "resumo": header_rows,
        "stats_por_classe": class_rows,
        "comparacao_c6_vs_veg": compare_rows,
        "contagem_por_arquivo": per_file_rows,
        "observacoes": notes,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_xlsx


def _analisar_arquivos(
    files: List[str],
    out_dir: str,
    grid_size: float = 5.0,
    sample_max_per_class: int = 200000,
    use_ground_mode: str = "auto",
) -> str:
    _ensure_laspy()
    if not files:
        raise ValueError("Nenhum arquivo selecionado.")

    rng = np.random.default_rng(42)
    t0 = time.time()

    global_stats: Dict[int, Dict[str, RunningStats]] = {}
    global_samples: Dict[int, Dict[str, np.ndarray]] = {}
    global_class_counts: Dict[int, int] = {}
    per_file_counts: List[Dict[str, object]] = []
    header_rows: List[Dict[str, object]] = []

    sample_features = {"z", "intensity", "height_norm", "scan_angle_abs", "return_ratio", "brightness"}

    for fpath in files:
        if not os.path.isfile(fpath):
            _log(f"[WARN] Arquivo nao encontrado: {fpath}")
            continue
        _log(f"[INFO] Abrindo: {fpath}")

        with laspy.open(fpath) as reader:
            dims = _detect_dims(reader)
            total_pts = int(reader.header.point_count)
            header = reader.header
            header_rows.append(
                {
                    "arquivo": fpath,
                    "pontos": total_pts,
                    "point_format": str(header.point_format),
                    "version": str(header.version),
                    "xmin": float(header.mins[0]),
                    "ymin": float(header.mins[1]),
                    "zmin": float(header.mins[2]),
                    "xmax": float(header.maxs[0]),
                    "ymax": float(header.maxs[1]),
                    "zmax": float(header.maxs[2]),
                    "dims": ",".join(dims),
                }
            )

            _log(f"[INFO] Dimensoes: {', '.join(dims)}")

        # Passo 1: contagem de classes para decidir modo do grid
        class_counts: Dict[int, int] = {}
        with laspy.open(fpath) as reader:
            for chunk in reader.chunk_iterator(2_000_000):
                cls = chunk.classification
                uniq, cnt = np.unique(cls, return_counts=True)
                for c, n in zip(uniq, cnt):
                    class_counts[int(c)] = class_counts.get(int(c), 0) + int(n)

        per_file_counts.append(
            {
                "arquivo": fpath,
                "classes": ",".join(str(c) for c in sorted(class_counts.keys())),
                "total_pontos": int(sum(class_counts.values())),
                "contagem_json": json.dumps(class_counts, ensure_ascii=False),
            }
        )
        for c, n in class_counts.items():
            global_class_counts[c] = global_class_counts.get(c, 0) + n

        use_ground_only = False
        if use_ground_mode == "always":
            use_ground_only = True
        elif use_ground_mode == "auto":
            use_ground_only = class_counts.get(2, 0) > 0

        grid_min: Optional[Dict[Tuple[int, int], float]] = None
        if grid_size > 0.0:
            _log(
                f"[INFO] Construindo grade min Z (grid={grid_size}m, ground_only={use_ground_only})"
            )
            with laspy.open(fpath) as reader:
                grid_min = _compute_grid_min_z(reader, grid_size, use_ground_only)
            _log(f"[INFO] Grade pronta: {len(grid_min)} celulas")

        _log("[INFO] Calculando estatisticas por classe...")
        with laspy.open(fpath) as reader:
            dims = _detect_dims(reader)
            for chunk in reader.chunk_iterator(2_000_000):
                cls = chunk.classification
                feats = _build_feature_arrays(chunk, dims, grid_min, grid_size)
                uniq = np.unique(cls)
                for c in uniq:
                    c_int = int(c)
                    mask = cls == c
                    if not np.any(mask):
                        continue
                    if c_int not in global_stats:
                        global_stats[c_int] = {}
                    if c_int not in global_samples:
                        global_samples[c_int] = {}
                    for feat_name, feat_arr in feats.items():
                        if feat_name not in global_stats[c_int]:
                            global_stats[c_int][feat_name] = RunningStats()
                        global_stats[c_int][feat_name].update(feat_arr[mask])

                        if feat_name in sample_features:
                            sample = global_samples[c_int].get(feat_name)
                            new_vals = feat_arr[mask]
                            new_vals = new_vals[np.isfinite(new_vals)]
                            if new_vals.size:
                                global_samples[c_int][feat_name] = _reservoir_concat(
                                    sample, new_vals, sample_max_per_class, rng
                                )

        _log(f"[INFO] Arquivo concluido: {fpath}")

    # Montar tabela de stats por classe
    class_rows: List[Dict[str, object]] = []
    pqs = [1, 5, 25, 50, 75, 95, 99]
    for cls, feats in sorted(global_stats.items(), key=lambda kv: kv[0]):
        cls_name = _format_class_name(cls)
        for feat_name, st in sorted(feats.items(), key=lambda kv: kv[0]):
            sample = global_samples.get(cls, {}).get(feat_name)
            pvals = _safe_percentiles(sample, pqs)
            class_rows.append(
                {
                    "classe": cls,
                    "classe_nome": cls_name,
                    "feature": feat_name,
                    "count": st.count,
                    "mean": st.mean,
                    "std": st.std,
                    "min": st.min,
                    "max": st.max,
                    **pvals,
                }
            )

    # Comparacao classe 6 vs vegetacao (3,4,5)
    compare_rows: List[Dict[str, object]] = []
    notes: List[str] = []
    build_stats = global_stats.get(BUILDING_CLASS, {})
    veg_stats = {}
    for feat_name in set().union(*(global_stats.get(c, {}).keys() for c in VEG_CLASSES)):
        veg_stats[feat_name] = _merge_stats([global_stats.get(c, {}).get(feat_name, RunningStats()) for c in VEG_CLASSES])

    for feat_name in sorted(set(build_stats.keys()) | set(veg_stats.keys())):
        st_b = build_stats.get(feat_name, RunningStats())
        st_v = veg_stats.get(feat_name, RunningStats())
        if st_b.count <= 0 or st_v.count <= 0:
            continue
        mean_b = st_b.mean
        mean_v = st_v.mean
        std_b = st_b.std
        std_v = st_v.std
        cohen_d = None
        if std_b is not None and std_v is not None and st_b.count > 1 and st_v.count > 1:
            pooled = math.sqrt(
                ((st_b.count - 1) * (std_b ** 2) + (st_v.count - 1) * (std_v ** 2))
                / (st_b.count + st_v.count - 2)
            )
            cohen_d = (mean_b - mean_v) / pooled if pooled > 0 else None

        sample_b = global_samples.get(BUILDING_CLASS, {}).get(feat_name)
        sample_v = None
        for c in VEG_CLASSES:
            s = global_samples.get(c, {}).get(feat_name)
            if s is None:
                continue
            sample_v = _reservoir_concat(sample_v, s, sample_max_per_class, rng)

        p_b = _safe_percentiles(sample_b, [5, 95])
        p_v = _safe_percentiles(sample_v, [5, 95])
        overlap_ratio = None
        if p_b["p5"] is not None and p_b["p95"] is not None and p_v["p5"] is not None and p_v["p95"] is not None:
            lo = max(p_b["p5"], p_v["p5"])
            hi = min(p_b["p95"], p_v["p95"])
            union = max(p_b["p95"], p_v["p95"]) - min(p_b["p5"], p_v["p5"])
            overlap_ratio = float(max(0.0, hi - lo) / union) if union > 0 else None

        best_t, best_bal, direction = _best_threshold(
            sample_b if sample_b is not None else np.array([]),
            sample_v if sample_v is not None else np.array([]),
        )

        compare_rows.append(
            {
                "feature": feat_name,
                "build_count": st_b.count,
                "veg_count": st_v.count,
                "build_mean": mean_b,
                "veg_mean": mean_v,
                "build_std": std_b,
                "veg_std": std_v,
                "cohen_d": cohen_d,
                "p5_build": p_b["p5"],
                "p95_build": p_b["p95"],
                "p5_veg": p_v["p5"],
                "p95_veg": p_v["p95"],
                "overlap_p5_p95": overlap_ratio,
                "best_threshold": best_t,
                "best_bal_acc": best_bal,
                "threshold_dir": direction,
            }
        )

    # Ordenar comparacao por separacao
    compare_rows.sort(key=lambda r: (abs(r["cohen_d"]) if r.get("cohen_d") is not None else 0.0), reverse=True)

    notes.append("Comparacao focada: classe 6 (building) vs classes 3/4/5 (vegetacao).")
    notes.append("Threshold_dir: gte = manter classe 6 quando feature >= threshold; lte = quando <=.")
    notes.append("Overlaps calculados com faixa p5-p95; valores menores indicam maior separacao.")
    if grid_size > 0.0:
        notes.append("height_norm usa grade de min Z; auto usa classe 2 (ground) se existir.")

    out_base = time.strftime("analise_las_classes_%Y%m%d_%H%M%S")
    report_path = _build_report(
        out_dir=out_dir,
        report_base=out_base,
        header_rows=header_rows,
        class_rows=class_rows,
        compare_rows=compare_rows,
        per_file_rows=per_file_counts,
        notes=notes,
    )

    t1 = time.time()
    _log(f"[OK] Relatorio gerado: {report_path}")
    _log(f"[INFO] Tempo total: {t1 - t0:.1f}s")
    return report_path


def main() -> None:
    try:
        files = _pick_las_files()
        if not files:
            _log("[INFO] Nenhum arquivo selecionado.")
            return
        out_dir = _pick_output_dir(os.path.dirname(files[0]))
        if not out_dir:
            _log("[INFO] Pasta de saida nao selecionada.")
            return
        _analisar_arquivos(files, out_dir)
    except Exception as exc:
        msg = f"Erro: {exc}"
        _log(msg)
        if messagebox is not None:
            messagebox.showerror("Erro", msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
