#!/usr/bin/env python3
"""
Recorta apenas GeoTIFFs que nao estao 100% dentro de um shapefile.

Requisitos:
  pip install rasterio geopandas shapely numpy
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

import geopandas as gpd
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.mask import mask
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def load_shape_union(shape_path: Path, target_crs) -> BaseGeometry:
    gdf = gpd.read_file(shape_path)
    if gdf.empty:
        raise ValueError(f"Shapefile vazio: {shape_path}")
    if gdf.crs is None:
        raise ValueError(f"Shapefile sem CRS definido: {shape_path}")
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
    if gdf.empty:
        raise ValueError(f"Shapefile sem geometrias validas: {shape_path}")
    return unary_union(gdf.geometry.values)


def find_tifs(input_dir: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for p in input_dir.glob(pattern):
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}:
            yield p


def is_fully_inside_shape(src: rasterio.DatasetReader, shape_union: BaseGeometry) -> bool:
    raster_geom = box(*src.bounds)

    # Filtro rapido: se o bounding box nao cabe, nao esta completamente dentro.
    if not shape_union.contains(raster_geom):
        return False

    # Teste preciso no footprint valido (considera nodata interno e rotacao).
    valid_mask = src.dataset_mask() > 0
    if not valid_mask.any():
        return True
    # Extrai poligonos de pixels validos sem carregar bandas.
    from rasterio.features import shapes  # import local para reduzir custo inicial

    geoms = [shape(g) for g, v in shapes(valid_mask.astype("uint8"), mask=valid_mask, transform=src.transform) if v == 1]
    if not geoms:
        return True
    footprint = unary_union(geoms)
    return shape_union.contains(footprint)


def crop_tif(
    src_path: Path,
    out_path: Path,
    shape_union: BaseGeometry,
    nodata_value: int,
) -> None:
    with rasterio.open(src_path) as src:
        out_img, out_transform = mask(
            src,
            [shape_union.__geo_interface__],
            crop=True,
            filled=False,  # retorna masked array para preservar transparencia por mascara
        )

        # Mantem no maximo 3 bandas.
        # Se tiver 1 ou 2 bandas, preserva sem falhar.
        out_band_count = min(3, src.count)
        out_img = out_img[:out_band_count]
        combined_valid = (~out_img.mask).all(axis=0)
        filled_data = out_img.filled(nodata_value)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=filled_data.shape[1],
            width=filled_data.shape[2],
            transform=out_transform,
            count=out_band_count,
            nodata=nodata_value,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(filled_data.astype(src.dtypes[0]))
            # Mascara interna do GeoTIFF: 255=visivel, 0=transparente.
            dst.write_mask((combined_valid * 255).astype("uint8"))


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Varredura inteligente de GeoTIFFs para recorte por shapefile."
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Pasta com TIF/TIFF.")
    parser.add_argument("--shape", required=True, type=Path, help="Caminho do shapefile (.shp).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Pasta de saida para recortes.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Busca TIFs recursivamente em subpastas.",
    )
    parser.add_argument(
        "--nodata",
        type=int,
        default=0,
        help="Valor nodata para pixels fora da area do shape (padrao: 0).",
    )
    parser.add_argument(
        "--copy-inside",
        action="store_true",
        help="Copia para saida os TIFs ja 100% dentro do shape sem recortar.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir
    shape_path = args.shape
    output_dir = args.output_dir

    if not input_dir.exists():
        print(f"ERRO: pasta de entrada nao existe: {input_dir}")
        return 1
    if not shape_path.exists():
        print(f"ERRO: shapefile nao existe: {shape_path}")
        return 1

    tifs = list(find_tifs(input_dir, args.recursive))
    if not tifs:
        print("Nenhum TIF/TIFF encontrado.")
        return 0

    inside_count = 0
    cropped_count = 0
    skipped_no_overlap = 0
    failed_count = 0
    shape_union_by_crs = {}

    print(f"TIFs encontrados: {len(tifs)}")
    for tif in tifs:
        rel = tif.relative_to(input_dir)
        out_tif = output_dir / rel
        try:
            with rasterio.open(tif) as src:
                crs_key = src.crs.to_wkt() if src.crs else "NONE"
                if crs_key == "NONE":
                    raise ValueError(f"{tif.name}: raster sem CRS.")
                if crs_key not in shape_union_by_crs:
                    shape_union_by_crs[crs_key] = load_shape_union(shape_path, src.crs)
                shape_union = shape_union_by_crs[crs_key]

                raster_bounds_geom = box(*src.bounds)
                if not shape_union.intersects(raster_bounds_geom):
                    skipped_no_overlap += 1
                    print(f"[SKIP-SEM-INTERSECCAO] {rel}")
                    continue

                if is_fully_inside_shape(src, shape_union):
                    inside_count += 1
                    print(f"[OK-JA-DENTRO] {rel}")
                    if args.copy_inside:
                        out_tif.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(tif, out_tif)
                    continue

            crop_tif(tif, out_tif, shape_union, args.nodata)
            cropped_count += 1
            print(f"[RECORTADO] {rel}")

        except (RasterioIOError, ValueError, OSError) as exc:
            failed_count += 1
            print(f"[ERRO] {rel}: {exc}")

    print("\nResumo:")
    print(f"  Ja dentro do shape (sem recorte): {inside_count}")
    print(f"  Recortados: {cropped_count}")
    print(f"  Sem interseccao (ignorados): {skipped_no_overlap}")
    print(f"  Falhas: {failed_count}")
    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
