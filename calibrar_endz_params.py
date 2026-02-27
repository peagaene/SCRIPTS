from __future__ import annotations

import argparse
import csv
import itertools
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def run_cmd(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("\n[ERRO] Comando falhou:", " ".join(cmd), file=sys.stderr)
        if p.stdout:
            print(p.stdout, file=sys.stderr)
        if p.stderr:
            print(p.stderr, file=sys.stderr)
        raise SystemExit(p.returncode)


def parse_floats_list(s: str, allow_none: bool = False) -> list[float | None]:
    vals: list[float | None] = []
    for raw in [x.strip() for x in s.split(",") if x.strip()]:
        if allow_none and raw.lower() in {"none", "off", "no"}:
            vals.append(None)
        else:
            vals.append(float(raw))
    return vals


def parse_bool_list(s: str) -> list[bool]:
    out: list[bool] = []
    for raw in [x.strip().lower() for x in s.split(",") if x.strip()]:
        if raw in {"on", "true", "1", "yes"}:
            out.append(True)
        elif raw in {"off", "false", "0", "no"}:
            out.append(False)
        else:
            raise ValueError(f"Valor booleano invalido: {raw}")
    return out


def read_endz_txt(path: Path, dz_col: int) -> np.ndarray:
    pts: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= dz_col or len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                dz = float(parts[dz_col])
            except ValueError:
                continue
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(dz):
                pts.append((x, y, dz))
    if not pts:
        raise SystemExit(f"Sem pontos validos: {path}")
    return np.asarray(pts, dtype=np.float64)


@dataclass
class Metrics:
    n_ref: int
    n_gen: int
    matched: int
    tol: float
    mean: float
    median: float
    mae: float
    rmse: float
    p95_abs: float
    dist_median: float
    dist_p95: float


def compare_nn(ref_pts: np.ndarray, gen_pts: np.ndarray, tol: float) -> Metrics:
    if cKDTree is None:
        raise SystemExit("scipy nao encontrado. Instale scipy para comparacao NN eficiente.")
    tree = cKDTree(gen_pts[:, :2])
    d, idx = tree.query(ref_pts[:, :2], k=1)
    m = d <= tol
    if not np.any(m):
        return Metrics(ref_pts.shape[0], gen_pts.shape[0], 0, tol, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    err = gen_pts[idx[m], 2] - ref_pts[m, 2]
    ae = np.abs(err)
    return Metrics(
        n_ref=ref_pts.shape[0],
        n_gen=gen_pts.shape[0],
        matched=int(m.sum()),
        tol=tol,
        mean=float(np.mean(err)),
        median=float(np.median(err)),
        mae=float(np.mean(ae)),
        rmse=float(np.sqrt(np.mean(err**2))),
        p95_abs=float(np.percentile(ae, 95)),
        dist_median=float(np.median(d)),
        dist_p95=float(np.percentile(d, 95)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibra parametros do gerar_endz.py comparando contra TXT de referencia (ex. TerraScan)."
    )
    ap.add_argument("--gerar-script", type=Path, default=Path(r"D:\SCRIPTS\gerar_endz.py"))
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument("--mms-file", required=True, type=Path, help="Faixa MMS unica para calibracao (.las/.laz).")
    ap.add_argument("--reference-txt", required=True, type=Path, help="TXT de referencia (TerraScan).")
    ap.add_argument("--reference-dz-col", type=int, default=4, help="Coluna DZ no TXT referencia (base 0). Default=4.")
    ap.add_argument("--dtm-aerial", type=Path, help="Raster unico DTM.")
    ap.add_argument("--dtm-aerial-dir", type=Path, help="Pasta com tiles DTM.")
    ap.add_argument("--dtm-glob", default="*.tif")
    ap.add_argument("--work-root", required=True, type=Path, help="Pasta raiz de trabalho para as rodadas.")
    ap.add_argument("--report-csv", required=True, type=Path, help="CSV de resultados das combinacoes.")
    ap.add_argument("--match-tol", type=float, default=1.0, help="Tolerancia XY (m) para comparacao NN. Default=1.0.")

    ap.add_argument("--step-lowest-values", default="0.5,1.0")
    ap.add_argument("--ts-iteration-angle-values", default="4,6,8")
    ap.add_argument("--ts-iteration-distance-values", default="0.8,1.1,1.4")
    ap.add_argument("--ground-thin-cell-values", default="none,1.0,1.5,2.0")
    ap.add_argument("--h-trim-mad-k-values", default="none,1.8,2.5")
    ap.add_argument("--noise-clean-values", default="on,off")
    ap.add_argument("--clip-h-min", type=float, default=-10.0)
    ap.add_argument("--clip-h-max", type=float, default=10.0)
    ap.add_argument("--h-trim-min-band", type=float, default=0.03)
    ap.add_argument("--max-runs", type=int, default=120)
    args = ap.parse_args()

    if not args.mms_file.exists():
        raise SystemExit(f"MMS nao encontrado: {args.mms_file}")
    if not args.reference_txt.exists():
        raise SystemExit(f"Referencia nao encontrada: {args.reference_txt}")

    ref = read_endz_txt(args.reference_txt, args.reference_dz_col)

    steps = [float(x) for x in parse_floats_list(args.step_lowest_values)]
    it_angles = [float(x) for x in parse_floats_list(args.ts_iteration_angle_values)]
    it_dists = [float(x) for x in parse_floats_list(args.ts_iteration_distance_values)]
    thin_cells = parse_floats_list(args.ground_thin_cell_values, allow_none=True)
    trim_ks = parse_floats_list(args.h_trim_mad_k_values, allow_none=True)
    noise_flags = parse_bool_list(args.noise_clean_values)

    combos = list(itertools.product(steps, it_angles, it_dists, thin_cells, trim_ks, noise_flags))
    if len(combos) > args.max_runs:
        combos = combos[: args.max_runs]

    run_in_dir = args.work_root / "_calib_input"
    run_in_dir.mkdir(parents=True, exist_ok=True)
    mms_copy = run_in_dir / args.mms_file.name
    if not mms_copy.exists():
        shutil.copy2(args.mms_file, mms_copy)

    rows: list[dict[str, object]] = []
    print(f"Total de combinacoes: {len(combos)}")
    for i, (step_lowest, it_angle, it_dist, thin_cell, trim_k, noise_clean_on) in enumerate(combos, start=1):
        run_id = f"r{i:03d}"
        work_dir = args.work_root / f"{run_id}_work"
        out_dir = args.work_root / f"{run_id}_out"
        work_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python_exe,
            str(args.gerar_script),
            "--mms-dir",
            str(run_in_dir),
            "--work-dir",
            str(work_dir),
            "--out-dir",
            str(out_dir),
            "--ground-algorithm",
            "terrascan_like",
            "--ts-max-building-size",
            "60",
            "--ts-iteration-angle",
            f"{it_angle}",
            "--ts-iteration-distance",
            f"{it_dist}",
            "--step-lowest",
            f"{step_lowest}",
            "--clip-h-min",
            f"{args.clip_h_min}",
            "--clip-h-max",
            f"{args.clip_h_max}",
            "--h-trim-min-band",
            f"{args.h_trim_min_band}",
            "--decimals",
            "3",
        ]
        if args.dtm_aerial:
            cmd.extend(["--dtm-aerial", str(args.dtm_aerial)])
        else:
            if not args.dtm_aerial_dir:
                raise SystemExit("Passe --dtm-aerial ou --dtm-aerial-dir.")
            cmd.extend(["--dtm-aerial-dir", str(args.dtm_aerial_dir), "--dtm-glob", args.dtm_glob])

        if thin_cell is None:
            cmd.append("--no-ground-thin")
        else:
            cmd.extend(["--ground-thin-cell", f"{thin_cell}"])

        if trim_k is None:
            cmd.append("--no-h-trim")
        else:
            cmd.extend(["--h-trim-mad-k", f"{trim_k}"])

        if not noise_clean_on:
            cmd.append("--no-noise-clean")

        run_cmd(cmd)

        gen_txt = out_dir / f"{args.mms_file.stem}_EN_DZ.txt"
        if not gen_txt.exists():
            print(f"[WARN] {run_id} sem EN_DZ: {gen_txt}")
            continue
        gen = read_endz_txt(gen_txt, 2)
        met = compare_nn(ref, gen, args.match_tol)
        row = {
            "run_id": run_id,
            "step_lowest": step_lowest,
            "ts_iteration_angle": it_angle,
            "ts_iteration_distance": it_dist,
            "ground_thin_cell": thin_cell if thin_cell is not None else "none",
            "h_trim_mad_k": trim_k if trim_k is not None else "none",
            "noise_clean": "on" if noise_clean_on else "off",
            "n_ref": met.n_ref,
            "n_gen": met.n_gen,
            "matched": met.matched,
            "match_tol": met.tol,
            "mean": met.mean,
            "median": met.median,
            "mae": met.mae,
            "rmse": met.rmse,
            "p95_abs": met.p95_abs,
            "dist_median": met.dist_median,
            "dist_p95": met.dist_p95,
            "out_dir": str(out_dir),
        }
        rows.append(row)
        print(
            f"[{run_id}] mae={met.mae:.3f} rmse={met.rmse:.3f} "
            f"mean={met.mean:.3f} matched={met.matched}/{met.n_ref}"
        )

    if not rows:
        raise SystemExit("Nenhuma rodada valida para comparar.")

    rows.sort(key=lambda r: (float(r["mae"]) if not math.isnan(float(r["mae"])) else 1e9))
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.report_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nTop 10 por MAE:")
    for r in rows[:10]:
        print(
            f"{r['run_id']}: mae={r['mae']:.3f} rmse={r['rmse']:.3f} mean={r['mean']:.3f} "
            f"step={r['step_lowest']} it_ang={r['ts_iteration_angle']} it_dist={r['ts_iteration_distance']} "
            f"thin={r['ground_thin_cell']} trim={r['h_trim_mad_k']} noise={r['noise_clean']}"
        )
    print(f"\nRelatorio: {args.report_csv}")


if __name__ == "__main__":
    main()
