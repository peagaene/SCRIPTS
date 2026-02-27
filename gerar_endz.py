from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path
from statistics import median

import laspy
import numpy as np
import rasterio


def run_cmd(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"Comando falhou: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")


def build_vrt_from_tiles(tiles: list[Path], vrt_out: Path) -> Path:
    if not tiles:
        raise SystemExit("Nenhum raster informado para mosaico DTM.")
    vrt_out.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["gdalbuildvrt", "-overwrite", str(vrt_out), *[str(t) for t in tiles]])
    if not vrt_out.exists():
        raise SystemExit(f"Falha ao criar VRT: {vrt_out}")
    return vrt_out


def resolve_dtm(args: argparse.Namespace, work_dir: Path) -> Path:
    if args.dtm_aerial:
        if not args.dtm_aerial.exists():
            raise SystemExit(f"DTM nao encontrado: {args.dtm_aerial}")
        return args.dtm_aerial

    tiles: list[Path] = []
    if args.dtm_aerial_list:
        tiles.extend(args.dtm_aerial_list)
    if args.dtm_aerial_dir:
        found = sorted(args.dtm_aerial_dir.glob(args.dtm_glob))
        if not found:
            raise SystemExit(f"Nenhum raster em {args.dtm_aerial_dir} com padrao {args.dtm_glob}")
        tiles.extend(found)
    if not tiles:
        raise SystemExit("Passe --dtm-aerial, ou --dtm-aerial-dir, ou --dtm-aerial-list.")

    uniq: list[Path] = []
    seen: set[str] = set()
    for p in tiles:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    for p in uniq:
        if not p.exists():
            raise SystemExit(f"DTM nao encontrado: {p}")
    if len(uniq) == 1:
        return uniq[0]
    return build_vrt_from_tiles(uniq, work_dir / "_dtm_aerial_mosaic.vrt")


def thin_ground_2d_keep_existing(
    las: laspy.LasData,
    ground_class: int,
    cell: float | None,
) -> tuple[np.ndarray, int]:
    cls = np.asarray(las.classification)
    g_idx = np.flatnonzero(cls == ground_class)
    if g_idx.size == 0:
        return g_idx, 0
    if cell is None or cell <= 0:
        return g_idx, int(g_idx.size)

    x = np.asarray(las.x)[g_idx]
    y = np.asarray(las.y)[g_idx]
    z = np.asarray(las.z)[g_idx]

    buckets: dict[tuple[int, int], list[int]] = {}
    for local_i, (xx, yy) in enumerate(zip(x, y)):
        key = (int(math.floor(xx / cell)), int(math.floor(yy / cell)))
        buckets.setdefault(key, []).append(local_i)

    chosen_global: list[int] = []
    for local_ids in buckets.values():
        zz = z[local_ids]
        z_med = float(np.median(zz))
        best_local = local_ids[int(np.argmin(np.abs(zz - z_med)))]
        chosen_global.append(int(g_idx[best_local]))

    chosen_global.sort()
    return np.asarray(chosen_global, dtype=np.int64), int(g_idx.size)


def compute_dz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    dtm_path: Path,
    chunk_size: int = 300_000,
) -> tuple[np.ndarray, np.ndarray]:
    dz = np.zeros_like(z, dtype=np.float64)
    valid = np.zeros(z.shape[0], dtype=bool)
    with rasterio.open(dtm_path) as ds:
        nodata = ds.nodata
        total = z.shape[0]
        for i in range(0, total, chunk_size):
            j = min(i + chunk_size, total)
            coords = np.column_stack((x[i:j], y[i:j]))
            vals = np.fromiter((v[0] for v in ds.sample(coords)), dtype=np.float64, count=j - i)
            ok = np.isfinite(vals)
            if nodata is not None:
                ok &= vals != nodata
            valid[i:j] = ok
            # DZ = Z_ref - Z_mms = -(Z_mms - Z_ref)
            dz[i:j] = np.where(ok, vals - z[i:j], np.nan)
    return dz, valid


def write_endz_txt(
    out_txt: Path,
    x: np.ndarray,
    y: np.ndarray,
    dz: np.ndarray,
    valid_mask: np.ndarray,
    median_factor: float | None,
    extra_col_1: float,
    extra_col_2: float,
    decimals: int,
) -> tuple[int, int]:
    xv = x[valid_mask]
    yv = y[valid_mask]
    dzv = dz[valid_mask]
    total = dzv.size
    if total == 0:
        out_txt.write_text("", encoding="utf-8")
        return (0, 0)

    median_limit: float | None = None
    if median_factor is not None:
        med = median(dzv.tolist())
        median_limit = median_factor * abs(med)

    fmt = f"{{:.{decimals}f}}"
    out_lines: list[str] = []
    for xx, yy, dzz in zip(xv, yv, dzv):
        dzz_val = float(dzz)
        if median_limit is not None and abs(dzz_val) > median_limit:
            continue
        out_lines.append(
            f"{fmt.format(float(xx))} {fmt.format(float(yy))} "
            f"{fmt.format(extra_col_1)} {fmt.format(extra_col_2)} {fmt.format(dzz_val)}\n"
        )

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(out_lines)
    return (len(out_lines), total)


def process_file(
    in_file: Path,
    out_dir: Path,
    dtm_path: Path,
    ground_class: int,
    thin_cell: float | None,
    laz_suffix: str,
    txt_suffix: str,
    overwrite: bool,
    median_factor: float | None,
    extra_col_1: float,
    extra_col_2: float,
    decimals: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_laz = out_dir / f"{in_file.stem}{laz_suffix}.laz"
    out_txt = out_dir / f"{in_file.stem}{txt_suffix}.txt"
    if out_laz.exists() and out_txt.exists() and not overwrite:
        print(f"[SKIP] {in_file.name}: saidas ja existem")
        return

    las = laspy.read(in_file)
    keep_idx, original_ground_count = thin_ground_2d_keep_existing(las, ground_class, thin_cell)
    if keep_idx.size == 0:
        print(f"[SKIP] {in_file.name}: nenhum ponto classe {ground_class}")
        return

    # Use source arrays for math to avoid compatibility issues with laspy
    # attribute access after creating a fresh LasData.
    x = np.asarray(las.x)[keep_idx]
    y = np.asarray(las.y)[keep_idx]
    z = np.asarray(las.z)[keep_idx]

    out = laspy.LasData(las.header)
    out.points = las.points[keep_idx].copy()
    out.write(out_laz)
    dz, valid = compute_dz(x, y, z, dtm_path)

    kept_txt, total_valid = write_endz_txt(
        out_txt=out_txt,
        x=x,
        y=y,
        dz=dz,
        valid_mask=valid,
        median_factor=median_factor,
        extra_col_1=extra_col_1,
        extra_col_2=extra_col_2,
        decimals=decimals,
    )

    print(
        f"[OK] {in_file.name} -> {out_laz.name}, {out_txt.name} | "
        f"ground={keep_idx.size}/{original_ground_count} | "
        f"txt={kept_txt}/{total_valid} | thin_cell={thin_cell if thin_cell is not None else 'off'}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extrai ground classificado, aplica thin 2D, calcula DZ no DTM e gera TXT final filtrado (X Y C1 C2 DZ)."
    )
    ap.add_argument("--mms-dir", required=True, type=Path, help="Pasta com .las/.laz.")
    ap.add_argument("--out-dir", required=True, type=Path, help="Pasta de saida (.laz thin + .txt EN_DZ).")
    ap.add_argument("--work-dir", required=False, type=Path, default=Path("."), help="Pasta de trabalho para VRT temporario.")

    ap.add_argument("--dtm-aerial", type=Path, help="Raster DTM unico.")
    ap.add_argument("--dtm-aerial-dir", type=Path, help="Pasta com tiles DTM.")
    ap.add_argument("--dtm-aerial-list", nargs="+", type=Path, help="Lista explicita de rasters DTM.")
    ap.add_argument("--dtm-glob", default="*.tif")

    ap.add_argument("--ground-class", type=int, default=2)
    ap.add_argument("--ground-thin-cell", type=float, default=1.0, help="Thin 2D XY (m). Use 0 para desligar.")
    ap.add_argument("--no-ground-thin", action="store_true")
    ap.add_argument("--laz-suffix", default="_GROUND_THIN")
    ap.add_argument("--txt-suffix", default="_EN_DZ")
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--median-factor", type=float, default=1.8, help="Remove |DZ| > fator*|mediana(DZ)|. Use negativo para desligar.")
    ap.add_argument("--extra-col-1", type=float, default=0.0)
    ap.add_argument("--extra-col-2", type=float, default=0.0)
    ap.add_argument("--decimals", type=int, default=3)

    args = ap.parse_args()
    dtm = resolve_dtm(args, args.work_dir)
    print(f"DTM: {dtm}")

    files = sorted([*args.mms_dir.glob("*.laz"), *args.mms_dir.glob("*.las")])
    if not files:
        raise SystemExit(f"Nenhum .las/.laz encontrado em: {args.mms_dir}")

    thin_cell = None if args.no_ground_thin else args.ground_thin_cell
    median_factor = None if args.median_factor < 0 else args.median_factor
    print(f"Total de faixas: {len(files)}")
    for f in files:
        process_file(
            in_file=f,
            out_dir=args.out_dir,
            dtm_path=dtm,
            ground_class=args.ground_class,
            thin_cell=thin_cell,
            laz_suffix=args.laz_suffix,
            txt_suffix=args.txt_suffix,
            overwrite=args.overwrite,
            median_factor=median_factor,
            extra_col_1=args.extra_col_1,
            extra_col_2=args.extra_col_2,
            decimals=args.decimals,
        )


if __name__ == "__main__":
    main()
