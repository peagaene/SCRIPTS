from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import uuid
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("\n[ERRO] Comando falhou:", " ".join(cmd), file=sys.stderr)
        if p.stdout:
            print(p.stdout, file=sys.stderr)
        if p.stderr:
            print(p.stderr, file=sys.stderr)
        raise SystemExit(p.returncode)
    if p.stdout.strip():
        print(p.stdout)
    if p.stderr.strip():
        print(p.stderr, file=sys.stderr)


def run_pdal_pipeline(stages: list[dict], work_dir: Path, tag: str) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"_pdal_{tag}_{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps({"pipeline": stages}, indent=2), encoding="utf-8")
    try:
        run_cmd(["pdal", "pipeline", str(path)])
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def smrf_settings(preset: str, cell_size: float) -> dict:
    if preset == "town":
        return {"type": "filters.smrf", "cell": cell_size, "slope": 0.20, "scalar": 1.20, "threshold": 0.45, "window": 18.0}
    return {"type": "filters.smrf", "cell": cell_size, "slope": 0.15, "scalar": 1.25, "threshold": 0.50, "window": 16.0}


def pmf_settings(
    cell_size: float,
    max_building_size: float,
    iteration_angle_deg: float,
    iteration_distance: float,
    follow_surface_trend: bool,
) -> dict:
    max_window_cells = max(3, int(round(max_building_size / max(cell_size, 0.01))))
    if max_window_cells % 2 == 0:
        max_window_cells += 1
    slope = max(0.01, math.tan(math.radians(max(0.1, iteration_angle_deg))))
    return {
        "type": "filters.pmf",
        "cell_size": cell_size,
        "slope": slope,
        "initial_distance": max(0.05, iteration_distance * 0.2),
        "max_distance": max(0.1, iteration_distance),
        "max_window_size": max_window_cells,
        "exponential": bool(follow_surface_trend),
    }


def process_one(
    mms_file: Path,
    work_dir: Path,
    out_dir: Path,
    out_ext: str,
    algorithm: str,
    smrf_cell: float,
    preset: str,
    ts_max_building_size: float,
    ts_iteration_angle: float,
    ts_iteration_distance: float,
    ts_follow_surface_trend: bool,
    stable_returns_only: bool,
    enable_noise_clean: bool,
    noise_cell: float,
    noise_threshold: float,
    outlier_mean_k: int,
    outlier_multiplier: float,
    ground_class: int,
) -> Path:
    base = mms_file.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_ground = out_dir / f"{base}_GROUND.{out_ext}"

    stages: list[dict] = [{"type": "readers.las", "filename": str(mms_file)}]
    if stable_returns_only:
        stages.append({"type": "filters.returns", "groups": "last,only"})
    if enable_noise_clean:
        stages.extend(
            [
                {"type": "filters.elm", "cell": noise_cell, "threshold": noise_threshold, "class": 7},
                {"type": "filters.outlier", "method": "statistical", "mean_k": outlier_mean_k, "multiplier": outlier_multiplier, "class": 7},
                {"type": "filters.range", "limits": "Classification![7:7]"},
            ]
        )

    if algorithm == "smrf":
        stages.append(smrf_settings(preset, smrf_cell))
    elif algorithm == "pmf":
        stages.append(pmf_settings(smrf_cell, 33.0 * smrf_cell, 8.0, 1.0, True))
    else:
        stages.append(
            pmf_settings(
                cell_size=smrf_cell,
                max_building_size=ts_max_building_size,
                iteration_angle_deg=ts_iteration_angle,
                iteration_distance=ts_iteration_distance,
                follow_surface_trend=ts_follow_surface_trend,
            )
        )
    stages.extend(
        [
            {"type": "filters.range", "limits": f"Classification[{ground_class}:{ground_class}]"},
            {"type": "writers.las", "filename": str(out_ground)},
        ]
    )
    run_pdal_pipeline(stages, work_dir, f"ground_{base}")
    return out_ground


def main() -> None:
    ap = argparse.ArgumentParser(description="Classifica/extrai ground de nuvens MMS para processamento posterior.")
    ap.add_argument("--mms-dir", required=True, type=Path, help="Pasta com .las/.laz MMS.")
    ap.add_argument("--work-dir", required=True, type=Path, help="Pasta temporaria de trabalho.")
    ap.add_argument("--out-dir", required=True, type=Path, help="Pasta de saida do ground.")
    ap.add_argument("--out-ext", choices=["las", "laz"], default="laz")

    ap.add_argument("--ground-algorithm", choices=["smrf", "pmf", "terrascan_like"], default="terrascan_like")
    ap.add_argument("--step-lowest", type=float, default=0.5, help="Cell size base para classificador.")
    ap.add_argument("--ground-preset", choices=["town", "nature"], default="town")
    ap.add_argument("--ts-max-building-size", type=float, default=60.0)
    ap.add_argument("--ts-iteration-angle", type=float, default=4.0)
    ap.add_argument("--ts-iteration-distance", type=float, default=0.8)
    ap.add_argument("--ts-no-follow-surface-trend", action="store_true")

    ap.add_argument("--all-returns", action="store_true")
    ap.add_argument("--no-noise-clean", action="store_true")
    ap.add_argument("--noise-cell", type=float, default=10.0)
    ap.add_argument("--noise-threshold", type=float, default=1.0)
    ap.add_argument("--outlier-mean-k", type=int, default=12)
    ap.add_argument("--outlier-multiplier", type=float, default=2.5)
    ap.add_argument("--ground-class", type=int, default=2)

    args = ap.parse_args()
    files = sorted([*args.mms_dir.glob("*.laz"), *args.mms_dir.glob("*.las")])
    if not files:
        raise SystemExit(f"Nenhum .las/.laz encontrado em: {args.mms_dir}")

    ts_follow = not args.ts_no_follow_surface_trend
    print(f"Total de faixas: {len(files)}")
    for f in files:
        out_ground = process_one(
            mms_file=f,
            work_dir=args.work_dir,
            out_dir=args.out_dir,
            out_ext=args.out_ext,
            algorithm=args.ground_algorithm,
            smrf_cell=args.step_lowest,
            preset=args.ground_preset,
            ts_max_building_size=args.ts_max_building_size,
            ts_iteration_angle=args.ts_iteration_angle,
            ts_iteration_distance=args.ts_iteration_distance,
            ts_follow_surface_trend=ts_follow,
            stable_returns_only=not args.all_returns,
            enable_noise_clean=not args.no_noise_clean,
            noise_cell=args.noise_cell,
            noise_threshold=args.noise_threshold,
            outlier_mean_k=args.outlier_mean_k,
            outlier_multiplier=args.outlier_multiplier,
            ground_class=args.ground_class,
        )
        print(f"[OK] {out_ground.name}")


if __name__ == "__main__":
    main()
