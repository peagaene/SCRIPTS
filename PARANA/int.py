from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

import laspy
import pdal


# -----------------------------
# Resolução automática por ppm²
# -----------------------------
def pick_resolution_from_ppm2(ppm2: float, target_ppp: float = 8.0,
                              res_min: float = 0.10, res_max: float = 2.00) -> float:
    if not np.isfinite(ppm2) or ppm2 <= 0:
        return 1.0
    res = math.sqrt(target_ppp / ppm2)
    return float(min(max(res, res_min), res_max))


def header_density_ppm2(las_path: Path) -> tuple[float, float, dict]:
    """
    Lê header (rápido) e estima densidade: ppm² = count / ((maxx-minx)*(maxy-miny)).
    Retorna (ppm2, area_m2, info_dict).
    """
    with laspy.open(str(las_path)) as f:
        h = f.header
        count = int(h.point_count)
        minx, miny, _ = h.mins
        maxx, maxy, _ = h.maxs

    width = maxx - minx
    height = maxy - miny
    area = float(width * height) if width > 0 and height > 0 else float("nan")
    ppm2 = float(count / area) if np.isfinite(area) and area > 0 else float("nan")

    info = {
        "count": count,
        "bounds": (minx, miny, maxx, maxy),
        "area_m2": area,
        "ppm2": ppm2,
    }
    return ppm2, area, info


# -----------------------------
# PDAL rasterização (DSM + Int)
# -----------------------------
def pdal_write_grid(las_path: Path, out_tif: Path, resolution: float,
                    dimension: str, output_type: str, data_type: str = "float32",
                    exclude_class7: bool = True) -> None:
    pipeline = [
        {"type": "readers.las", "filename": str(las_path)},
    ]

    if exclude_class7:
        pipeline.append({
            "type": "filters.range",
            "limits": "Classification![7:7]"
        })

    pipeline.append({
        "type": "writers.gdal",
        "filename": str(out_tif),
        "resolution": float(resolution),
        "output_type": output_type,     # "max" p/ DSM Z, "mean" p/ intensidade etc.
        "dimension": dimension,         # "Z" ou "Intensity"
        "data_type": data_type,
        "gdaldriver": "GTiff",
        "nodata": -9999
    })

    pl = pdal.Pipeline(json.dumps(pipeline))
    pl.execute()


# -----------------------------
# Render "ray-traced-like" 2D
# -----------------------------
def hillshade(z: np.ndarray, cellsize: float, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    dzdx = np.gradient(z, axis=1) / cellsize
    dzdy = np.gradient(z, axis=0) / cellsize

    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect = np.arctan2(dzdy, -dzdx)

    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)

    shaded = (
        np.sin(alt) * np.cos(slope) +
        np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    )
    return np.clip(shaded, 0, 1)


def normalize_to_uint8(x: np.ndarray, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    lo = np.nanpercentile(x, pmin)
    hi = np.nanpercentile(x, pmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x, dtype=np.uint8)

    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    return (y * 255.0).astype(np.uint8)


def apply_colormap_height(z: np.ndarray) -> np.ndarray:
    zn = normalize_to_uint8(z).astype(np.float32) / 255.0  # 0..1
    # azul->verde->amarelo->vermelho (simples e bom)
    r = np.clip(2.0 * (zn - 0.5), 0, 1)
    g = np.clip(2.0 * (0.5 - np.abs(zn - 0.5)), 0, 1)
    b = np.clip(2.0 * (0.5 - zn), 0, 1)
    rgb = np.stack([r, g, b], axis=0)
    return (rgb * 255.0).astype(np.uint8)


def intensity_to_rgb(intensity: np.ndarray) -> np.ndarray:
    i8 = normalize_to_uint8(intensity)
    return np.stack([i8, i8, i8], axis=0).astype(np.uint8)


def shade_rgb(rgb: np.ndarray, shade: np.ndarray, ambient: float = 0.35) -> np.ndarray:
    shade2 = ambient + (1 - ambient) * shade
    shade2 = np.clip(shade2, 0, 1)
    out = rgb.astype(np.float32) * shade2[None, :, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def read_match_grids(z_path: Path, i_path: Path):
    with rasterio.open(z_path) as zsrc:
        z = zsrc.read(1, masked=True).astype(np.float32)
        profile = zsrc.profile
        transform = zsrc.transform
        cellsize = float(transform.a)

    with rasterio.open(i_path) as isrc:
        if (isrc.width != profile["width"]) or (isrc.height != profile["height"]) or (isrc.transform != transform):
            intensity = isrc.read(
                1,
                out_shape=(profile["height"], profile["width"]),
                resampling=Resampling.bilinear
            ).astype(np.float32)
        else:
            intensity = isrc.read(1).astype(np.float32)

    z = np.where(np.ma.getmaskarray(z), np.nan, z)
    # nodata do PDAL (-9999) vira NaN
    z = np.where(z <= -9000, np.nan, z)
    intensity = np.where(intensity <= -9000, np.nan, intensity)

    return z, intensity, profile, cellsize


def write_rgb_geotiff(out_path: Path, rgb: np.ndarray, base_profile: dict):
    profile = base_profile.copy()
    profile.update(
        dtype="uint8",
        count=3,
        nodata=None,
        compress="deflate",
        predictor=2
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(rgb[0], 1)
        dst.write(rgb[1], 2)
        dst.write(rgb[2], 3)


# -----------------------------
# Processo por arquivo
# -----------------------------
def process_one(las_path: Path, out_dir: Path,
                target_ppp: float = 8.0,
                azimuth_deg: float = 315.0,
                altitude_deg: float = 45.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ppm2, area_m2, info = header_density_ppm2(las_path)
    res = pick_resolution_from_ppm2(ppm2, target_ppp=target_ppp)

    stem = las_path.stem
    z_tif = out_dir / f"{stem}_DSM_Z.tif"
    i_tif = out_dir / f"{stem}_INT.tif"

    # DSM (Z max) e Intensidade (mean), excluindo classe 7
    pdal_write_grid(las_path, z_tif, res, dimension="Z", output_type="max", data_type="float32", exclude_class7=True)
    pdal_write_grid(las_path, i_tif, res, dimension="Intensity", output_type="mean", data_type="float32", exclude_class7=True)

    # Render 24-bit RGB "ray-traced-like"
    z, intensity, profile, cellsize = read_match_grids(z_tif, i_tif)
    shade = hillshade(z, cellsize=cellsize, azimuth_deg=azimuth_deg, altitude_deg=altitude_deg)

    rgb_height = apply_colormap_height(z)
    rgb_height_rt = shade_rgb(rgb_height, shade, ambient=0.35)
    write_rgb_geotiff(out_dir / f"{stem}_height_colored_raytraced.tif", rgb_height_rt, profile)

    rgb_int = intensity_to_rgb(intensity)
    rgb_int_rt = shade_rgb(rgb_int, shade, ambient=0.35)
    write_rgb_geotiff(out_dir / f"{stem}_intensity_raytraced.tif", rgb_int_rt, profile)

    # Log simples
    print(f"[OK] {las_path.name}")
    print(f"     pontos={info['count']:,}  área={area_m2:,.1f} m²  ppm²={ppm2:.2f}  res={res:.3f} m")


def main():
    in_path = Path(r"F:\TESTE_PARANA\5_NUVEM_PONTOS\LOTE_09\BLOCO_B\2_NPC_COMPLETO\2_1_LAS")   # <- ajuste
    out_dir = Path(r"F:\TESTE_PARANA\5_NUVEM_PONTOS")  # <- ajuste

    files = []
    if in_path.is_dir():
        files = sorted([*in_path.glob("*.las"), *in_path.glob("*.laz")])
    else:
        files = [in_path]

    if not files:
        raise SystemExit("Nenhum .las/.laz encontrado.")

    for f in files:
        process_one(f, out_dir, target_ppp=8.0, azimuth_deg=315.0, altitude_deg=45.0)


if __name__ == "__main__":
    main()
