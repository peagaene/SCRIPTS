from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import laspy
import numpy as np
import rasterio


def run_cmd(cmd: list[str]) -> None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if p.stdout.strip():
            print(p.stdout)
        if p.stderr.strip():
            print(p.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print("\n[ERRO] Comando falhou:", " ".join(cmd), file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        raise


def txt_endz_to_csv(endz_txt: Path, out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with endz_txt.open("r", encoding="utf-8", errors="ignore") as f_in, out_csv.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        w = csv.writer(f_out)
        w.writerow(["X", "Y", "DZ"])
        for line in f_in:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                dz = float(parts[2])
            except ValueError:
                continue
            w.writerow([x, y, dz])
            count += 1
    return count


def read_endz_points(endz_txt: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    dzs: list[float] = []
    with endz_txt.open("r", encoding="utf-8", errors="ignore") as f_in:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                dz = float(parts[2])
            except ValueError:
                continue
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(dz)):
                continue
            xs.append(x)
            ys.append(y)
            dzs.append(dz)
    if not xs:
        raise SystemExit(f"Sem pontos validos em: {endz_txt}")
    return np.asarray(xs), np.asarray(ys), np.asarray(dzs)


def fit_robust_plane(
    x: np.ndarray,
    y: np.ndarray,
    dz: np.ndarray,
    max_iter: int = 6,
    sigma_k: float = 2.5,
    min_keep_ratio: float = 0.25,
) -> tuple[np.ndarray, float, float, np.ndarray, float]:
    """
    Fit DZ = a + b*(X-x0) + c*(Y-y0) with iterative robust clipping.
    Returns (coef[a,b,c], x0, y0, keep_mask, rmse).
    """
    x0 = float(np.median(x))
    y0 = float(np.median(y))
    dx = x - x0
    dy = y - y0
    A = np.column_stack((np.ones_like(dx), dx, dy))
    keep = np.ones(x.shape[0], dtype=bool)

    coef = np.zeros(3, dtype=np.float64)
    for _ in range(max_iter):
        if keep.sum() < max(30, int(min_keep_ratio * x.shape[0])):
            break
        coef, *_ = np.linalg.lstsq(A[keep], dz[keep], rcond=None)
        resid = dz - (A @ coef)
        r_keep = resid[keep]
        med = float(np.median(r_keep))
        mad = float(np.median(np.abs(r_keep - med)))
        sigma = 1.4826 * mad
        if sigma < 1e-6:
            break
        new_keep = np.abs(resid - med) <= (sigma_k * sigma)
        if new_keep.sum() == keep.sum():
            keep = new_keep
            break
        keep = new_keep

    coef, *_ = np.linalg.lstsq(A[keep], dz[keep], rcond=None)
    resid = dz - (A @ coef)
    rmse = float(np.sqrt(np.mean((resid[keep]) ** 2))) if keep.any() else float(np.sqrt(np.mean(resid**2)))
    return coef, x0, y0, keep, rmse


def write_vrt(csv_path: Path, vrt_path: Path) -> None:
    vrt = f"""<OGRVRTDataSource>
  <OGRVRTLayer name="endz">
    <SrcDataSource>{csv_path}</SrcDataSource>
    <GeometryType>wkbPoint</GeometryType>
    <GeometryField encoding="PointFromColumns" x="X" y="Y"/>
  </OGRVRTLayer>
</OGRVRTDataSource>
"""
    vrt_path.write_text(vrt, encoding="utf-8")


def build_dz_raster(
    endz_txt: Path,
    las_path: Path,
    dz_tif: Path,
    tmp_dir: Path,
    grid_res: float,
    power: float,
    radius: float,
    max_points: int,
    min_points: int,
    nodata: float,
) -> int:
    las = laspy.read(las_path)
    min_x, min_y, _ = las.header.mins
    max_x, max_y, _ = las.header.maxs

    csv_path = tmp_dir / f"{las_path.stem}_endz.csv"
    vrt_path = tmp_dir / f"{las_path.stem}_endz.vrt"
    n = txt_endz_to_csv(endz_txt, csv_path)
    if n == 0:
        raise SystemExit(f"Sem pontos validos em: {endz_txt}")
    write_vrt(csv_path, vrt_path)

    cmd = [
        "gdal_grid",
        "-a",
        f"invdistnn:power={power}:radius={radius}:max_points={max_points}:min_points={min_points}",
        "-zfield",
        "DZ",
        "-txe",
        str(min_x),
        str(max_x),
        "-tye",
        str(min_y),
        str(max_y),
        "-tr",
        str(grid_res),
        str(grid_res),
        "-a_nodata",
        str(nodata),
        "-ot",
        "Float32",
        "-of",
        "GTiff",
        str(vrt_path),
        str(dz_tif),
    ]
    run_cmd(cmd)
    return n


def apply_dz_to_las(
    in_las: Path,
    dz_tif: Path,
    out_las: Path,
    nodata: float,
    chunk_size: int = 500_000,
) -> tuple[int, int]:
    las = laspy.read(in_las)
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)

    dz = np.zeros_like(z, dtype=np.float64)
    valid_mask = np.zeros(z.shape[0], dtype=bool)

    with rasterio.open(dz_tif) as ds:
        ds_nodata = ds.nodata if ds.nodata is not None else nodata
        total = z.shape[0]
        for i in range(0, total, chunk_size):
            j = min(i + chunk_size, total)
            coords = np.column_stack((x[i:j], y[i:j]))
            vals = np.fromiter((v[0] for v in ds.sample(coords)), dtype=np.float64, count=j - i)
            ok = np.isfinite(vals) & (vals != ds_nodata)
            valid_mask[i:j] = ok
            dz[i:j] = np.where(ok, vals, 0.0)

    z_corr = z + dz
    out = laspy.LasData(las.header)
    out.points = las.points.copy()
    out.z = z_corr
    out_las.parent.mkdir(parents=True, exist_ok=True)
    out.write(out_las)

    return int(valid_mask.sum()), int(z.shape[0])


def apply_smooth_surface_to_las(
    in_las: Path,
    out_las: Path,
    coef: np.ndarray,
    x0: float,
    y0: float,
    dz_clip: float | None = None,
) -> tuple[int, int]:
    las = laspy.read(in_las)
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)

    dz = coef[0] + coef[1] * (x - x0) + coef[2] * (y - y0)
    if dz_clip is not None and dz_clip > 0:
        dz = np.clip(dz, -dz_clip, dz_clip)

    out = laspy.LasData(las.header)
    out.points = las.points.copy()
    out.z = z + dz
    out_las.parent.mkdir(parents=True, exist_ok=True)
    out.write(out_las)
    return int(z.shape[0]), int(z.shape[0])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aplica correcao vertical na MMS usando EN_DZ (E N DZ), replicando logica de transformacao por superficie."
    )
    ap.add_argument("--mms-dir", required=True, type=Path, help="Pasta de entrada com .las/.laz MMS.")
    ap.add_argument("--endz-dir", required=True, type=Path, help="Pasta com arquivos *_EN_DZ.txt.")
    ap.add_argument("--out-dir", required=True, type=Path, help="Pasta de saida dos LAS/LAZ corrigidos.")
    ap.add_argument("--work-dir", required=True, type=Path, help="Pasta temporaria para CSV/VRT/TIF.")
    ap.add_argument("--grid-res", type=float, default=1.0, help="Resolucao do raster DZ (m). Default=1.0.")
    ap.add_argument("--power", type=float, default=2.0, help="Potencia do IDW. Default=2.0.")
    ap.add_argument("--radius", type=float, default=3.0, help="Raio de busca (m). Default=3.0.")
    ap.add_argument("--max-points", type=int, default=12, help="Max pontos por celula no IDW. Default=12.")
    ap.add_argument("--min-points", type=int, default=1, help="Min pontos por celula no IDW. Default=1.")
    ap.add_argument("--nodata", type=float, default=-9999.0, help="NoData do raster DZ. Default=-9999.")
    ap.add_argument(
        "--correction-model",
        choices=["idw", "smooth_surface"],
        default="idw",
        help="Modelo de correcao vertical: idw (local) ou smooth_surface (plano robusto).",
    )
    ap.add_argument("--robust-sigma-k", type=float, default=2.5, help="Rejeicao robusta na superficie suave. Default=2.5.")
    ap.add_argument("--robust-iters", type=int, default=6, help="Iteracoes robustas na superficie suave. Default=6.")
    ap.add_argument(
        "--smooth-dz-clip",
        type=float,
        default=None,
        help="Limite absoluto de DZ aplicado no modelo smooth_surface (m). Ex: 6.0",
    )
    ap.add_argument(
        "--endz-suffix",
        default="_EN_DZ.txt",
        help="Sufixo do TXT EN_DZ para casar por nome com a faixa MMS. Default=_EN_DZ.txt.",
    )
    ap.add_argument(
        "--out-suffix",
        default="_AJUSTADO",
        help="Sufixo no nome de saida (antes da extensao). Default=_AJUSTADO.",
    )
    args = ap.parse_args()

    mms_files = sorted([*args.mms_dir.glob("*.laz"), *args.mms_dir.glob("*.las")])
    if not mms_files:
        raise SystemExit(f"Nenhum .las/.laz encontrado em: {args.mms_dir}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Total de faixas MMS: {len(mms_files)}")
    for mms in mms_files:
        endz = args.endz_dir / f"{mms.stem}{args.endz_suffix}"
        if not endz.exists():
            print(f"[SKIP] EN_DZ nao encontrado para {mms.name}: {endz.name}")
            continue

        dz_tif = args.work_dir / f"{mms.stem}_DZ_surface.tif"
        out_las = args.out_dir / f"{mms.stem}{args.out_suffix}{mms.suffix}"
        if args.correction_model == "idw":
            n_ctrl = build_dz_raster(
                endz_txt=endz,
                las_path=mms,
                dz_tif=dz_tif,
                tmp_dir=args.work_dir,
                grid_res=args.grid_res,
                power=args.power,
                radius=args.radius,
                max_points=args.max_points,
                min_points=args.min_points,
                nodata=args.nodata,
            )
            n_valid, n_total = apply_dz_to_las(
                in_las=mms,
                dz_tif=dz_tif,
                out_las=out_las,
                nodata=args.nodata,
            )
            pct = 100.0 * n_valid / n_total if n_total else 0.0
            print(
                f"[OK] {mms.name} -> {out_las.name} | model=idw | ctrl={n_ctrl} | "
                f"pontos com DZ={n_valid}/{n_total} ({pct:.1f}%)"
            )
        else:
            x, y, dz = read_endz_points(endz)
            coef, x0, y0, keep, rmse = fit_robust_plane(
                x=x,
                y=y,
                dz=dz,
                max_iter=args.robust_iters,
                sigma_k=args.robust_sigma_k,
            )
            n_valid, n_total = apply_smooth_surface_to_las(
                in_las=mms,
                out_las=out_las,
                coef=coef,
                x0=x0,
                y0=y0,
                dz_clip=args.smooth_dz_clip,
            )
            print(
                f"[OK] {mms.name} -> {out_las.name} | model=smooth_surface | "
                f"ctrl_keep={int(keep.sum())}/{dz.size} | rmse_ctrl={rmse:.3f} m | "
                f"a={coef[0]:.6f}, b={coef[1]:.9f}, c={coef[2]:.9f} | pontos={n_valid}/{n_total}"
            )


if __name__ == "__main__":
    main()
