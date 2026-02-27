#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Renomeia e organiza produtos (NP/MDS/MDT/HC/INT) a partir de um caminho base.
Regras:
- NP/NPc/MDS/MDT: corrige nomes conforme pasta (5_NUVEM_PONTOS / 6_MDS / 7_MDT)
- HC/INT: renomeia IMG_HC / IMG_INTENS e move para subpastas corretas
- GeoTIFF: se nao tiver EPSG, define 31982
- Metadados HC/INT: gera TXT a partir do LAS correspondente
- Itens fora do padrao de pastas/extensoes: move para ../apagar/<NOME_BASE>/...
"""

from __future__ import annotations
import argparse
import os
import queue
import re
import shutil
import tempfile
import time
import datetime
import threading
from pathlib import Path
from typing import Optional, Tuple, Iterable

try:
    from osgeo import gdal, osr
    _HAS_GDAL = True
except Exception:
    gdal = None
    osr = None
    _HAS_GDAL = False

try:
    import rasterio
    from rasterio.crs import CRS
    _HAS_RASTERIO = True
except Exception:
    rasterio = None
    CRS = None
    _HAS_RASTERIO = False

if _HAS_GDAL:
    gdal.UseExceptions()

try:
    from metadados import build_metadata as build_las_metadata
    from metadados import metadata_to_text as las_metadata_to_text
    _HAS_LAS_METADATA = True
except Exception:
    build_las_metadata = None
    las_metadata_to_text = None
    _HAS_LAS_METADATA = False

try:
    import PySimpleGUI as sg
    _HAS_SG = True
except Exception:
    sg = None
    _HAS_SG = False


IGNORED_DIRS = {
    "System Volume Information",
    "$RECYCLE.BIN",
    "Recycle.Bin",
    "$RECYLCE.BIN",
    "RECYLCE.BIN",
    "apagar",
    "FOUND.000",
    "Config.Msi",
}

IGNORED_FILES = {
    "thumbs.db",
}

EXPECTED_STRUCTURE = {
    "5_NUVEM_PONTOS": {"1_NP", "2_NPC_COMPLETO"},
    "6_MDS": {"1_GEOTIFF"},
    "7_MDT": {"1_LAS", "2_LASDATASET", "3_ASCII", "4_GEOTIFF"},
    "8_IMG_HIPSOMETRICA_COMPOSTA": {"1_GEOTIFF", "2_METADADOS"},
    "9_IMG_INTENSIDADE": {"1_GEOTIFF", "2_METADADOS"},
}

EXPECTED_NUVEM_SUBSTRUCTURE = {
    "1_NP": {"1_1_LAS"},
    "2_NPC_COMPLETO": {"2_1_LAS", "2_2_LASDATASET"},
}

EXPECTED_FILE_COUNTS = {
    ("LOTE_09", "BLOCO_A"): 216,
    ("LOTE_09", "BLOCO_B"): 192,
    ("LOTE_09", "BLOCO_C"): 216,
    ("LOTE_09", "BLOCO_D"): 192,
    ("LOTE_09", "BLOCO_E"): 216,
    ("LOTE_09", "BLOCO_F"): 216,
    ("LOTE_09", "BLOCO_G"): 216,
    ("LOTE_09", "BLOCO_H"): 192,
    ("LOTE_09", "BLOCO_I"): 255,
    ("LOTE_09", "BLOCO_J"): 192,
    ("LOTE_09", "BLOCO_K"): 128,
    ("LOTE_10", "BLOCO_I"): 240,
    ("LOTE_10", "BLOCO_J"): 199,
    ("LOTE_10", "BLOCO_M"): 143,
    ("LOTE_10", "BLOCO_N"): 149,
}


def dec_to_dms_str(value: float, is_lon: bool = True) -> str:
    hemi = "E" if is_lon and value >= 0 else "W" if is_lon else "N" if value >= 0 else "S"
    v = abs(value)
    d = int(v)
    m_float = (v - d) * 60.0
    m = int(m_float)
    s = (m_float - m) * 60.0
    return f'{d}° {m:02d}\' {s:07.4f}" {hemi}'


def guess_sample_type(gdal_dtype) -> str:
    if gdal_dtype == gdal.GDT_Byte:
        return "Unsigned 8-bit Integer"
    if gdal_dtype == gdal.GDT_UInt16:
        return "Unsigned 16-bit Integer"
    if gdal_dtype == gdal.GDT_Int16:
        return "Signed 16-bit Integer"
    if gdal_dtype == gdal.GDT_UInt32:
        return "Unsigned 32-bit Integer"
    if gdal_dtype == gdal.GDT_Int32:
        return "Signed 32-bit Integer"
    if gdal_dtype in (gdal.GDT_Float32, gdal.GDT_Float64):
        return "Floating Point"
    return "Unknown Format (0)"


def get_proj_desc_safe(srs) -> str:
    try:
        zone = srs.GetUTMZone()
    except Exception:
        zone = 0
    try:
        units_name = srs.GetLinearUnitsName() or "meters"
    except Exception:
        units_name = "meters"
    try:
        datum = (srs.GetAttrValue("DATUM") or "").replace("_", " ").strip()
    except Exception:
        datum = "Unknown"
    if zone != 0:
        return f"UTM Zone {zone if zone < 0 else '-' + str(zone)} / {datum.split()[0] or datum} / {units_name}"
    try:
        projcs = srs.GetAttrValue("PROJCS") or "Unknown"
    except Exception:
        projcs = "Unknown"
    return f"{projcs} / {units_name}"


def detect_photometric(info_json: dict, band_count: int) -> str:
    md_image = info_json.get("metadata", {}).get("IMAGE_STRUCTURE", {})
    photometric = md_image.get("PHOTOMETRIC")
    if photometric:
        up = photometric.upper()
        if "RGB" in up:
            return "RGB Full-Color"
        if "PALETTE" in up:
            return "Palette Color"
        if "MINISBLACK" in up or "MINISWHITE" in up:
            return "Grayscale"
        return photometric
    if band_count == 3:
        return "RGB Full-Color"
    if band_count == 1:
        return "Grayscale"
    return "Unknown"


def flatten_metadata(md: dict, prefix: str = "META") -> list[str]:
    lines: list[str] = []

    def _walk(obj, path):
        if isinstance(obj, dict):
            for k, v in sorted(obj.items()):
                _walk(v, path + [str(k)])
        elif isinstance(obj, list):
            for idx, v in enumerate(obj, start=1):
                _walk(v, path + [str(idx)])
        else:
            key = "_".join(path)
            lines.append(f"{prefix}_{key}={obj}")

    _walk(md, [])
    return lines


def build_metadata_text(tif_path: str) -> str:
    start_time = time.time()
    ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError("GDAL retornou None ao abrir o arquivo.")
    load_time = time.time() - start_time

    try:
        info_json = gdal.Info(tif_path, options=gdal.InfoOptions(format="json")) or {}
    except Exception:
        info_json = {}

    driver_short = ds.GetDriver().ShortName if ds.GetDriver() else "Unknown"
    driver_long = ds.GetDriver().LongName if ds.GetDriver() else "Unknown"

    cols = ds.RasterXSize
    rows = ds.RasterYSize
    bands = ds.RasterCount
    gt = ds.GetGeoTransform()

    origin_x = gt[0]
    pixel_w = gt[1]
    origin_y = gt[3]
    pixel_h = gt[5]

    ulx = origin_x
    uly = origin_y
    lrx = origin_x + cols * pixel_w
    lry = origin_y + rows * pixel_h
    if lry > uly:
        uly, lry = lry, uly

    area_m2 = abs(pixel_w * pixel_h * cols * rows)
    area_km2 = area_m2 / 1e6

    proj_desc = "Unknown"
    proj_datum = "Unknown"
    proj_units = "Unknown"
    epsg_str = "EPSG:Unknown"
    pcs_citation = "Unknown"
    geog_citation = "Unknown"
    srs = None

    proj_wkt = ds.GetProjection()
    if proj_wkt and proj_wkt.strip():
        try:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(proj_wkt)
            proj_desc = get_proj_desc_safe(srs)
            try:
                proj_datum = (srs.GetAttrValue("DATUM") or "Unknown").replace("_", " ")
            except Exception:
                proj_datum = "Unknown"
            try:
                proj_units = srs.GetLinearUnitsName() or "Unknown"
            except Exception:
                proj_units = "Unknown"
            try:
                srs_clone = srs.Clone()
                srs_clone.AutoIdentifyEPSG()
                epsg = srs_clone.GetAuthorityCode("PROJCS") or srs_clone.GetAuthorityCode(None)
                if epsg:
                    epsg_str = f"EPSG:{epsg}"
            except Exception:
                pass
            pcs_name = srs.GetAttrValue("PROJCS") or ""
            if pcs_name:
                pcs_citation = pcs_name
            try:
                srs_geo = srs.CloneGeogCS()
                geog_name = srs_geo.GetAttrValue("GEOGCS") or srs_geo.GetAttrValue("DATUM") or ""
                if geog_name:
                    geog_citation = geog_name.replace("_", " ")
            except Exception:
                pass
        except Exception:
            srs = None

    west_lon = east_lon = north_lat = south_lat = 0.0
    ul_lon = ul_lat = ur_lon = ur_lat = lr_lon = lr_lat = ll_lon = ll_lat = 0.0

    if srs is not None:
        try:
            srs_geo = srs.CloneGeogCS()
            ct = osr.CoordinateTransformation(srs, srs_geo)

            def proj_to_geo(x, y):
                lon, lat, _ = ct.TransformPoint(x, y)
                return lon, lat

            urx = lrx
            ury = uly
            llx = ulx
            lly = lry

            ul_lon, ul_lat = proj_to_geo(ulx, uly)
            ur_lon, ur_lat = proj_to_geo(urx, ury)
            lr_lon, lr_lat = proj_to_geo(lrx, lry)
            ll_lon, ll_lat = proj_to_geo(llx, lly)

            all_lons = [ul_lon, ur_lon, lr_lon, ll_lon]
            all_lats = [ul_lat, ur_lat, lr_lat, ll_lat]
            west_lon = min(all_lons)
            east_lon = max(all_lons)
            north_lat = max(all_lats)
            south_lat = min(all_lats)
        except Exception:
            pass

    band1 = ds.GetRasterBand(1)
    gdal_dtype = band1.DataType
    bits_per_sample = gdal.GetDataTypeSize(gdal_dtype)
    bit_depth_total = bits_per_sample * bands
    sample_type = guess_sample_type(gdal_dtype)

    if gdal_dtype in (gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_UInt32):
        sample_format = "Unsigned Integer"
    elif gdal_dtype in (gdal.GDT_Int16, gdal.GDT_Int32):
        sample_format = "Signed Integer"
    elif gdal_dtype in (gdal.GDT_Float32, gdal.GDT_Float64):
        sample_format = "Floating Point"
    else:
        sample_format = "Unknown"

    md_all = info_json.get("metadata", {})
    md_tiff = md_all.get("TIFF", {})
    md_geotiff_block = md_all.get("GeoTIFF", {})
    md_geotiff_flat = {k: v for k, v in md_all.items() if k.startswith("GeoTIFF::")}
    md_geotiff = {}
    md_geotiff.update(md_geotiff_block)
    md_geotiff.update(md_geotiff_flat)

    rows_per_strip = md_tiff.get("ROWS_PER_STRIP")
    if not rows_per_strip:
        bx, by = band1.GetBlockSize()
        rows_per_strip = by if by > 0 else "Unknown"

    compression = md_tiff.get("COMPRESSION")
    if not compression:
        md_img_struct = info_json.get("metadata", {}).get("IMAGE_STRUCTURE", {})
        compression = md_img_struct.get("COMPRESSION", "None")
    if not compression:
        compression = "Unknown"

    pixel_scale_str = f"( {abs(pixel_w)}, {abs(pixel_h)}, 1.0 )"
    tiepoints_str = f"( 0.00, 0.00, 0.00 ) --> ( {ulx}, {uly}, 0.000 )"

    model_type = md_geotiff.get("GeoTIFF::ModelTypeGeoKey", "Projection Coordinate System")
    raster_type = md_geotiff.get("GeoTIFF::RasterTypeGeoKey", "Pixel is Area")

    photometric = detect_photometric(info_json, bands)

    overview_lines = []
    band_json_list = info_json.get("bands", [])
    if band_json_list:
        ovs = band_json_list[0].get("overviews", [])
        for idx, ov in enumerate(ovs, start=1):
            size = ov.get("size", [])
            if len(size) == 2:
                o_cols, o_rows = size
                overview_lines.append(f"OVERVIEW {idx}=Pixel Size: {o_cols} x {o_rows}")

    geokey_lines = []
    for key in [
        "GeoTIFF::ProjLinearUnitsGeoKey",
        "GeoTIFF::ProjectedCSTypeGeoKey",
        "GeoTIFF::GeographicTypeGeoKey",
        "GeoTIFF::GeogSemiMajorAxisGeoKey",
        "GeoTIFF::GeogSemiMinorAxisGeoKey",
        "GeoTIFF::GeogEllipsoidGeoKey",
        "GeoTIFF::GeogToWGS84GeoKey",
    ]:
        if key in md_geotiff:
            geokey_lines.append(f"{key}={md_geotiff[key]}")

    try:
        ctime = datetime.datetime.fromtimestamp(os.path.getctime(tif_path))
        ctime_str = ctime.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        ctime_str = "Unknown"

    try:
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(tif_path))
        mtime_str = mtime.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        mtime_str = "Unknown"

    color_bands = ",".join(str(i) for i in range(bands))
    load_time_str = f"{load_time:.2f} s"

    lines = []
    lines.append(f"FILENAME={os.path.abspath(tif_path)}")
    lines.append(f"DESCRIPTION={os.path.basename(tif_path)}")
    lines.append(f"DRIVER={driver_short} ({driver_long})")
    lines.append(f"UPPER LEFT X={ulx}")
    lines.append(f"UPPER LEFT Y={uly}")
    lines.append(f"LOWER RIGHT X={lrx}")
    lines.append(f"LOWER RIGHT Y={lry}")
    lines.append(f"WEST LONGITUDE={dec_to_dms_str(west_lon, is_lon=True)}")
    lines.append(f"NORTH LATITUDE={dec_to_dms_str(north_lat, is_lon=False)}")
    lines.append(f"EAST LONGITUDE={dec_to_dms_str(east_lon, is_lon=True)}")
    lines.append(f"SOUTH LATITUDE={dec_to_dms_str(south_lat, is_lon=False)}")
    lines.append(f"UL CORNER LONGITUDE={dec_to_dms_str(ul_lon, is_lon=True)}")
    lines.append(f"UL CORNER LATITUDE={dec_to_dms_str(ul_lat, is_lon=False)}")
    lines.append(f"UR CORNER LONGITUDE={dec_to_dms_str(ur_lon, is_lon=True)}")
    lines.append(f"UR CORNER LATITUDE={dec_to_dms_str(ur_lat, is_lon=False)}")
    lines.append(f"LR CORNER LONGITUDE={dec_to_dms_str(lr_lon, is_lon=True)}")
    lines.append(f"LR CORNER LATITUDE={dec_to_dms_str(lr_lat, is_lon=False)}")
    lines.append(f"LL CORNER LONGITUDE={dec_to_dms_str(ll_lon, is_lon=True)}")
    lines.append(f"LL CORNER LATITUDE={dec_to_dms_str(ll_lat, is_lon=False)}")
    lines.append(f"PROJ_DESC={proj_desc}")
    lines.append(f"PROJ_DATUM={proj_datum}")
    lines.append(f"PROJ_UNITS={proj_units}")
    lines.append(f"EPSG_CODE={epsg_str}")
    lines.append(f"BBOX AREA={area_km2:.3f} sq km")
    lines.append(f"FILE_CREATION_TIME={ctime_str}")
    lines.append(f"FILE_MODIFIED_TIME={mtime_str}")
    lines.append(f"NUM COLUMNS={cols}")
    lines.append(f"NUM ROWS={rows}")
    lines.append(f"NUM BANDS={bands}")
    lines.append(f"COLOR BANDS={color_bands}")
    lines.append(f"PIXEL WIDTH={abs(pixel_w)} meters")
    lines.append(f"PIXEL HEIGHT={abs(pixel_h)} meters")
    lines.append(f"BIT DEPTH={bit_depth_total}")
    lines.append(f"SAMPLE TYPE={sample_type}")
    lines.append(f"GT_CITATION={pcs_citation}")
    lines.append(f"GEOG_CITATION={geog_citation}")
    lines.append(f"PHOTOMETRIC={photometric}")
    lines.append(f"SAMPLE_FORMAT={sample_format}")
    lines.append(f"ROWS_PER_STRIP={rows_per_strip}")
    lines.append(f"COMPRESSION={compression}")
    lines.append(f"PIXEL_SCALE={pixel_scale_str}")
    lines.append(f"TIEPOINTS={tiepoints_str}")
    lines.append(f"MODEL_TYPE={model_type if isinstance(model_type, str) else 'Projection Coordinate System'}")
    lines.append(f"RASTER_TYPE={raster_type if isinstance(raster_type, str) else 'Pixel is Area'}")

    for idx, bjson in enumerate(band_json_list, start=1):
        nodata_val = bjson.get("noDataValue")
        color_interp = bjson.get("colorInterpretation") or "Unknown"
        offset = bjson.get("offset")
        scale = bjson.get("scale")
        unit = bjson.get("unit")
        block = bjson.get("block", [])
        minimum = bjson.get("minimum")
        maximum = bjson.get("maximum")
        mean = bjson.get("mean")
        stddev = bjson.get("stdDev")

        lines.append(f"BAND{idx}_COLOR_INTERP={color_interp}")
        lines.append(f"BAND{idx}_NODATA={nodata_val if nodata_val is not None else 'None'}")
        lines.append(f"BAND{idx}_OFFSET={offset if offset is not None else 0}")
        lines.append(f"BAND{idx}_SCALE={scale if scale is not None else 1}")
        if unit:
            lines.append(f"BAND{idx}_UNIT={unit}")
        if block:
            lines.append(f"BAND{idx}_BLOCKSIZE={'x'.join(str(v) for v in block)}")
        if minimum is not None:
            lines.append(f"BAND{idx}_MIN={minimum}")
        if maximum is not None:
            lines.append(f"BAND{idx}_MAX={maximum}")
        if mean is not None:
            lines.append(f"BAND{idx}_MEAN={mean}")
        if stddev is not None:
            lines.append(f"BAND{idx}_STDDEV={stddev}")

        band_md = bjson.get("metadata", {})
        if band_md:
            lines.extend(flatten_metadata(band_md, prefix=f"BAND{idx}_META"))

    lines.extend(overview_lines)
    lines.extend(geokey_lines)

    if md_all:
        lines.extend(flatten_metadata(md_all, prefix="DATASET_META"))

    return "\n".join(lines)


def remover_revisoes(base: str) -> str:
    partes = base.split("_")
    while partes and re.fullmatch(r"R\d+", partes[-1]):
        partes.pop()
    return "_".join(partes)


def aplicar_regra(base: str, novo_trecho: str) -> str:
    base_lower = base.lower()
    if not any(x in base_lower for x in ("_np_", "_npc_c_", "_npc_t_", "_mdt_", "_mds_")):
        return ""
    base = remover_revisoes(base)
    base = base.replace("-", "_")
    base = re.sub(r"_(NPc?_C|NPc?_T|NP|MDT|MDS)_", novo_trecho, base, flags=re.IGNORECASE)
    if not base.endswith("_R0"):
        base = f"{base}_R0"
    return base


def ensure_tif_epsg(
    tif_path: Path, default_epsg: int = 31982, force_epsg: bool = False
) -> Optional[int]:
    if not tif_path.is_file():
        return None
    if _HAS_GDAL:
        ds = gdal.Open(str(tif_path), gdal.GA_Update)
        if ds is None:
            raise RuntimeError("GDAL nao conseguiu abrir o arquivo.")
        proj = ds.GetProjection()
        epsg = None
        if proj:
            srs = osr.SpatialReference()
            if srs.ImportFromWkt(proj) == 0:
                try:
                    srs.AutoIdentifyEPSG()
                except Exception:
                    pass
                epsg = srs.GetAuthorityCode(None)
                if epsg:
                    try:
                        epsg = int(epsg)
                    except Exception:
                        epsg = None
        if (not epsg) or (force_epsg and epsg != int(default_epsg)):
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(int(default_epsg))
            ds.SetProjection(srs.ExportToWkt())
            ds.FlushCache()
            epsg = int(default_epsg)
        ds = None
        return epsg
    if _HAS_RASTERIO:
        with rasterio.open(str(tif_path), "r+") as ds:
            epsg = ds.crs.to_epsg() if ds.crs else None
            if (not epsg) or (force_epsg and epsg != int(default_epsg)):
                ds.crs = CRS.from_epsg(int(default_epsg))
                epsg = int(default_epsg)
        return epsg
    raise RuntimeError("GDAL/rasterio nao disponivel para consultar/definir EPSG.")


def get_tif_band_count(tif_path: Path) -> Optional[int]:
    if not tif_path.is_file():
        return None
    if _HAS_GDAL:
        ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
        if ds is None:
            return None
        count = int(ds.RasterCount)
        ds = None
        return count
    if _HAS_RASTERIO:
        with rasterio.open(str(tif_path), "r") as ds:
            return int(ds.count)
    return None


def iter_files_safely(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        ignored_norm = {normalize_part(d) for d in IGNORED_DIRS}
        dirnames[:] = [d for d in dirnames if normalize_part(d) not in ignored_norm]
        for name in filenames:
            if name.lower() in IGNORED_FILES:
                continue
            yield Path(dirpath) / name


def normalize_part(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip().upper())


def is_valid_lote(part: str) -> bool:
    return normalize_part(part).startswith("LOTE_")


def is_valid_bloco(part: str) -> bool:
    return re.fullmatch(r"BLOCO_[A-Z]", normalize_part(part)) is not None


def subdir_key(name: str) -> str:
    norm = normalize_part(name)
    m = re.fullmatch(r"\d+_(.+)", norm)
    return m.group(1) if m else norm


def subdir_token(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", subdir_key(name))


def _closest_expected_name(current_name: str, expected_names: set[str]) -> Optional[str]:
    key = subdir_key(current_name)
    token = subdir_token(current_name)

    for expected in expected_names:
        if subdir_key(expected) == key:
            return expected

    # Fallback tolerante para nomes quase corretos (ex.: 2_NPP -> 1_NP).
    ranked: list[tuple[int, str]] = []
    for expected in expected_names:
        exp_token = subdir_token(expected)
        score = 0
        if token == exp_token:
            score = 100
        elif token.startswith(exp_token) or exp_token.startswith(token):
            score = 80 + min(len(token), len(exp_token))
        elif token in exp_token or exp_token in token:
            score = 60
        if score > 0:
            ranked.append((score, expected))

    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def expected_subdir_name(top: str, current_name: str) -> Optional[str]:
    return _closest_expected_name(current_name, EXPECTED_STRUCTURE.get(top, set()))


def expected_nuvem_inner_name(parent_name: str, current_name: str) -> Optional[str]:
    parent_norm = normalize_part(parent_name)
    return _closest_expected_name(current_name, EXPECTED_NUVEM_SUBSTRUCTURE.get(parent_norm, set()))


def merge_dir_contents(src: Path, dest: Path, log, dry_run: bool) -> None:
    for item in src.iterdir():
        target = dest / item.name
        if target.exists():
            log(f"[SUBDIR-WARN] conflito ao mesclar: {item} -> {target}")
            continue
        log(f"[SUBDIR-MOVE] {item} -> {target}")
        if not dry_run:
            shutil.move(str(item), str(target))
    try:
        if not any(src.iterdir()):
            log(f"[SUBDIR-RMDIR] {src}")
            if not dry_run:
                src.rmdir()
    except Exception:
        pass


def normalize_numbered_subfolders(base_dir: Path, log, dry_run: bool) -> None:
    for top in EXPECTED_STRUCTURE:
        top_dir = base_dir / top
        if not top_dir.is_dir():
            continue
        for lote_dir in [d for d in top_dir.iterdir() if d.is_dir() and is_valid_lote(d.name)]:
            for bloco_dir in [d for d in lote_dir.iterdir() if d.is_dir() and is_valid_bloco(d.name)]:
                for child in [d for d in bloco_dir.iterdir() if d.is_dir()]:
                    expected = expected_subdir_name(top, child.name)
                    if expected and child.name != expected:
                        dest = bloco_dir / expected
                        if dest.exists():
                            log(f"[SUBDIR-MERGE] {child} -> {dest}")
                            if not dry_run:
                                merge_dir_contents(child, dest, log, dry_run)
                        else:
                            log(f"[SUBDIR] {child} -> {dest}")
                            if not dry_run:
                                child.rename(dest)

                if top == "5_NUVEM_PONTOS":
                    for first_level in [d for d in bloco_dir.iterdir() if d.is_dir()]:
                        first_expected = expected_subdir_name(top, first_level.name)
                        if not first_expected:
                            continue
                        parent_for_map = first_expected
                        for second_level in [d for d in first_level.iterdir() if d.is_dir()]:
                            expected2 = expected_nuvem_inner_name(parent_for_map, second_level.name)
                            if expected2 and second_level.name != expected2:
                                dest2 = first_level / expected2
                                if dest2.exists():
                                    log(f"[SUBDIR-MERGE] {second_level} -> {dest2}")
                                    if not dry_run:
                                        merge_dir_contents(second_level, dest2, log, dry_run)
                                else:
                                    log(f"[SUBDIR] {second_level} -> {dest2}")
                                    if not dry_run:
                                        second_level.rename(dest2)


def ensure_base_structure(base_dir: Path, log, dry_run: bool) -> None:
    for top in EXPECTED_STRUCTURE:
        top_dir = base_dir / top
        if not top_dir.exists():
            log(f"[MKDIR] {top_dir}")
            if not dry_run:
                top_dir.mkdir(parents=True, exist_ok=True)

        for lote_dir in [d for d in top_dir.iterdir() if d.is_dir() and is_valid_lote(d.name)]:
            for bloco_dir in [d for d in lote_dir.iterdir() if d.is_dir() and is_valid_bloco(d.name)]:
                for sub in EXPECTED_STRUCTURE[top]:
                    p = bloco_dir / sub
                    if not p.exists():
                        log(f"[MKDIR] {p}")
                        if not dry_run:
                            p.mkdir(parents=True, exist_ok=True)

                if top == "5_NUVEM_PONTOS":
                    for first, inner_set in EXPECTED_NUVEM_SUBSTRUCTURE.items():
                        p1 = bloco_dir / first
                        if not p1.exists():
                            log(f"[MKDIR] {p1}")
                            if not dry_run:
                                p1.mkdir(parents=True, exist_ok=True)
                        for inner in inner_set:
                            p2 = p1 / inner
                            if not p2.exists():
                                log(f"[MKDIR] {p2}")
                                if not dry_run:
                                    p2.mkdir(parents=True, exist_ok=True)


def migrate_legacy_geotiff_dirs(base_dir: Path, log, dry_run: bool) -> None:
    products = [
        "8_IMG_HIPSOMETRICA_COMPOSTA",
        "9_IMG_INTENSIDADE",
    ]
    for product in products:
        root = base_dir / product
        if not root.is_dir():
            continue
        for lote_dir in [d for d in root.iterdir() if d.is_dir()]:
            for bloco_dir in [d for d in lote_dir.iterdir() if d.is_dir()]:
                new_dir = bloco_dir / "1_GEOTIFF"
                for legacy_name in ("2_GEOTIFF", "3_GEOTIFF"):
                    old_dir = bloco_dir / legacy_name
                    if not old_dir.is_dir():
                        continue
                    if not new_dir.exists():
                        log(f"[MIGRA] {old_dir} -> {new_dir}")
                        if not dry_run:
                            old_dir.rename(new_dir)
                    else:
                        log(f"[MIGRA-WARN] {old_dir} existe, mas {new_dir} tambem existe. Mantido para revisao.")


def is_allowed_relpath(rel: Path, is_dir: bool) -> bool:
    parts = [normalize_part(p) for p in rel.parts]
    suffix = rel.suffix.lower()
    if not parts:
        return True

    top = parts[0]
    if top not in EXPECTED_STRUCTURE:
        return False

    # Padrao geral: <TOP>/<LOTE>/<BLOCO>/<SUBDIR>/...
    if len(parts) == 1:
        return is_dir
    if len(parts) == 2:
        return is_dir and is_valid_lote(parts[1])
    if len(parts) == 3:
        return is_dir and is_valid_lote(parts[1]) and is_valid_bloco(parts[2])
    if len(parts) >= 4:
        if not (is_valid_lote(parts[1]) and is_valid_bloco(parts[2])):
            return False
        if parts[3] not in EXPECTED_STRUCTURE[top]:
            return False
        if is_dir:
            if top == "5_NUVEM_PONTOS" and len(parts) >= 5:
                if parts[3] == "1_NP":
                    return parts[4] == "1_1_LAS"
                if parts[3] == "2_NPC_COMPLETO":
                    return parts[4] in {"2_1_LAS", "2_2_LASDATASET"}
            return True

        if top == "5_NUVEM_PONTOS":
            if len(parts) < 5:
                return False
            if len(parts) >= 5:
                if parts[3] == "1_NP":
                    if parts[4] != "1_1_LAS":
                        return False
                    return suffix in {".las", ".laz", ".prj"}
                elif parts[3] == "2_NPC_COMPLETO":
                    if parts[4] == "2_1_LAS":
                        return suffix in {".las", ".laz", ".prj"}
                    if parts[4] == "2_2_LASDATASET":
                        return suffix in {".lasd", ".prj"}
                    if parts[4] not in {"2_1_LAS", "2_2_LASDATASET"}:
                        return False
            return False

        if top == "6_MDS":
            return suffix in {".tif", ".tiff", ".prj"}

        if top == "7_MDT":
            if parts[3] == "1_LAS":
                return suffix in {".las", ".laz", ".prj"}
            if parts[3] == "2_LASDATASET":
                return suffix in {".lasd", ".prj"}
            if parts[3] == "3_ASCII":
                return suffix in {".txt", ".asc", ".xyz", ".csv", ".prj"}
            if parts[3] == "4_GEOTIFF":
                return suffix in {".tif", ".tiff", ".prj"}
            return False

        if top in {"8_IMG_HIPSOMETRICA_COMPOSTA", "9_IMG_INTENSIDADE"}:
            if parts[3] == "1_GEOTIFF":
                return suffix in {".tif", ".tiff", ".prj"}
            if parts[3] == "2_METADADOS":
                return suffix == ".txt"
            return False

        return False
    return False


def send_out_of_pattern_to_apagar(base_dir: Path, log, dry_run: bool) -> None:
    apagar_root = base_dir.parent / "apagar" / base_dir.name
    to_move: list[Path] = []
    ignored_norm = {normalize_part(d) for d in IGNORED_DIRS}

    for p in sorted(base_dir.rglob("*"), key=lambda x: (len(x.parts), str(x)), reverse=True):
        if any(normalize_part(part) in ignored_norm for part in p.parts):
            continue
        if p.is_file() and p.name.lower() in IGNORED_FILES:
            continue
        rel = p.relative_to(base_dir)
        if not is_allowed_relpath(rel, is_dir=p.is_dir()):
            to_move.append(p)

    for src in to_move:
        # Se pai ja foi movido, ignora.
        if not src.exists():
            continue
        rel = src.relative_to(base_dir)
        dest = apagar_root / rel
        log(f"[APAGAR] {src} -> {dest}")
        if dry_run:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))


def verify_expected_file_counts(base_dir: Path, log) -> None:
    products = [
        "8_IMG_HIPSOMETRICA_COMPOSTA",
        "9_IMG_INTENSIDADE",
    ]
    mismatches = 0

    for product in products:
        for (lote, bloco), expected in sorted(EXPECTED_FILE_COUNTS.items()):
            geotiff_dir = base_dir / product / lote / bloco / "1_GEOTIFF"
            actual = 0
            if geotiff_dir.is_dir():
                actual = sum(
                    1
                    for p in geotiff_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
                )
            if actual != expected:
                mismatches += 1
                delta = actual - expected
                sign = "+" if delta > 0 else ""
                log(
                    f"[ALERTA-QTD] {product} {lote} {bloco}: esperado={expected} encontrado={actual} delta={sign}{delta}"
                )

    if mismatches == 0:
        log("[QTD] Conferencia de quantidade OK para todos os LOTE/BLOCO esperados.")
    else:
        log(f"[QTD] Conferencia finalizada com {mismatches} divergencias.")


def expected_lasd_name(lote: str, bloco: str) -> str:
    m_lote = re.search(r"(\d{1,2})", lote)
    yy = m_lote.group(1).zfill(2) if m_lote else "00"
    m_bloco = re.search(r"BLOCO_([A-Z])", normalize_part(bloco))
    bx = m_bloco.group(1) if m_bloco else "X"
    return f"ES_L{yy}_{bx}_NPc_C_R0.lasd"


def review_pasta_5_nuvem_pontos(base_dir: Path, log) -> None:
    top = base_dir / "5_NUVEM_PONTOS"
    if not top.is_dir():
        log("[P5-ALERTA] Pasta 5_NUVEM_PONTOS nao encontrada.")
        return

    divergencias = 0
    for lote_dir in [d for d in top.iterdir() if d.is_dir() and is_valid_lote(d.name)]:
        for bloco_dir in [d for d in lote_dir.iterdir() if d.is_dir() and is_valid_bloco(d.name)]:
            key = (lote_dir.name, bloco_dir.name)
            esperado = EXPECTED_FILE_COUNTS.get(key)
            if esperado is None:
                log(f"[P5-ALERTA] Sem quantidade esperada para {lote_dir.name}/{bloco_dir.name}.")
                continue

            np_dir = bloco_dir / "1_NP" / "1_1_LAS"
            npc_dir = bloco_dir / "2_NPC_COMPLETO" / "2_1_LAS"
            lasd_dir = bloco_dir / "2_NPC_COMPLETO" / "2_2_LASDATASET"

            qtd_np = (
                sum(1 for p in np_dir.iterdir() if p.is_file() and p.suffix.lower() in {".las", ".laz"})
                if np_dir.is_dir()
                else 0
            )
            qtd_npc = (
                sum(1 for p in npc_dir.iterdir() if p.is_file() and p.suffix.lower() in {".las", ".laz"})
                if npc_dir.is_dir()
                else 0
            )

            if qtd_np != esperado:
                divergencias += 1
                log(
                    f"[P5-QTD] 1_NP {lote_dir.name}/{bloco_dir.name}: esperado={esperado} encontrado={qtd_np}"
                )
            if qtd_npc != esperado:
                divergencias += 1
                log(
                    f"[P5-QTD] 2_NPC_COMPLETO {lote_dir.name}/{bloco_dir.name}: esperado={esperado} encontrado={qtd_npc}"
                )

            lasd_files = (
                [p for p in lasd_dir.iterdir() if p.is_file() and p.suffix.lower() == ".lasd"]
                if lasd_dir.is_dir()
                else []
            )
            nome_esperado_lasd = expected_lasd_name(lote_dir.name, bloco_dir.name)
            if len(lasd_files) != 1:
                divergencias += 1
                log(
                    f"[P5-LASD] {lote_dir.name}/{bloco_dir.name}: esperado 1 arquivo .lasd, encontrado {len(lasd_files)}"
                )
            elif lasd_files[0].name != nome_esperado_lasd:
                divergencias += 1
                log(
                    f"[P5-LASD] {lote_dir.name}/{bloco_dir.name}: nome incorreto ({lasd_files[0].name}) esperado ({nome_esperado_lasd})"
                )

    if divergencias == 0:
        log("[P5-OK] Revisao da pasta 5_NUVEM_PONTOS concluida sem divergencias.")
    else:
        log(f"[P5-ALERTA] Revisao da pasta 5_NUVEM_PONTOS com {divergencias} divergencias.")


def enforce_pasta_5_first_level(base_dir: Path, log, dry_run: bool) -> None:
    top = base_dir / "5_NUVEM_PONTOS"
    if not top.is_dir():
        return

    apagar_root = base_dir.parent / "apagar" / base_dir.name
    expected = {"1_NP", "2_NPC_COMPLETO"}

    for lote_dir in [d for d in top.iterdir() if d.is_dir() and is_valid_lote(d.name)]:
        for bloco_dir in [d for d in lote_dir.iterdir() if d.is_dir() and is_valid_bloco(d.name)]:
            # 1) corrige nomes numerados "quase certos" no nivel BLOCO.
            for child in [d for d in bloco_dir.iterdir() if d.is_dir()]:
                expected_name = expected_subdir_name("5_NUVEM_PONTOS", child.name)
                if expected_name and child.name != expected_name:
                    dest = bloco_dir / expected_name
                    if dest.exists():
                        log(f"[P5-ORG-MERGE] {child} -> {dest}")
                        if not dry_run:
                            merge_dir_contents(child, dest, log, dry_run)
                    else:
                        log(f"[P5-ORG-RENAME] {child} -> {dest}")
                        if not dry_run:
                            child.rename(dest)

            # 2) move tudo que nao e 1_NP/2_NPC_COMPLETO para apagar.
            for item in list(bloco_dir.iterdir()):
                if item.is_dir() and item.name in expected:
                    continue
                rel = item.relative_to(base_dir)
                dest = apagar_root / rel
                if item.is_dir():
                    log(f"[P5-ORG-APAGAR] pasta fora do padrao: {item} -> {dest}")
                else:
                    log(f"[P5-ORG-APAGAR] arquivo fora do padrao: {item} -> {dest}")
                if dry_run:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(dest))


def is_valid_mds_tif_name(stem: str, lote: str, bloco: str) -> bool:
    m_lote = re.search(r"(\d{1,2})", lote)
    yy = m_lote.group(1).zfill(2) if m_lote else None
    m_bloco = re.search(r"BLOCO_([A-Z])", normalize_part(bloco))
    bx = m_bloco.group(1) if m_bloco else None
    if not yy or not bx:
        return False
    s = stem.upper()
    if "_MDS_" not in s:
        return False
    if not s.endswith("_R0"):
        return False
    return s.startswith(f"ES_L{yy}_{bx}_")


def enforce_pasta_6_mds_first_level(base_dir: Path, log, dry_run: bool) -> None:
    top = base_dir / "6_MDS"
    if not top.is_dir():
        return
    apagar_root = base_dir.parent / "apagar" / base_dir.name

    for lote_dir in [d for d in top.iterdir() if d.is_dir() and is_valid_lote(d.name)]:
        for bloco_dir in [d for d in lote_dir.iterdir() if d.is_dir() and is_valid_bloco(d.name)]:
            # Corrige numero errado da pasta GEOTIFF (ex.: 2_GEOTIFF -> 1_GEOTIFF).
            for child in [d for d in bloco_dir.iterdir() if d.is_dir()]:
                expected_name = expected_subdir_name("6_MDS", child.name)
                if expected_name and child.name != expected_name:
                    dest = bloco_dir / expected_name
                    if dest.exists():
                        log(f"[P6-ORG-MERGE] {child} -> {dest}")
                        if not dry_run:
                            merge_dir_contents(child, dest, log, dry_run)
                    else:
                        log(f"[P6-ORG-RENAME] {child} -> {dest}")
                        if not dry_run:
                            child.rename(dest)

            # Mantem apenas 1_GEOTIFF no mesmo nivel.
            for item in list(bloco_dir.iterdir()):
                if item.is_dir() and item.name == "1_GEOTIFF":
                    continue
                rel = item.relative_to(base_dir)
                dest = apagar_root / rel
                kind = "pasta" if item.is_dir() else "arquivo"
                log(f"[P6-ORG-APAGAR] {kind} fora do padrao: {item} -> {dest}")
                if dry_run:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(dest))


def review_pasta_6_mds(base_dir: Path, log, dry_run: bool) -> None:
    top = base_dir / "6_MDS"
    if not top.is_dir():
        log("[P6-ALERTA] Pasta 6_MDS nao encontrada.")
        return

    divergencias = 0
    for lote_dir in [d for d in top.iterdir() if d.is_dir() and is_valid_lote(d.name)]:
        for bloco_dir in [d for d in lote_dir.iterdir() if d.is_dir() and is_valid_bloco(d.name)]:
            key = (lote_dir.name, bloco_dir.name)
            esperado = EXPECTED_FILE_COUNTS.get(key)
            if esperado is None:
                log(f"[P6-ALERTA] Sem quantidade esperada para {lote_dir.name}/{bloco_dir.name}.")
                continue

            geo_dir = bloco_dir / "1_GEOTIFF"
            if not geo_dir.is_dir():
                divergencias += 1
                log(f"[P6-ALERTA] Pasta ausente: {geo_dir}")
                continue

            tifs = [p for p in geo_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
            qtd = len(tifs)
            if qtd != esperado:
                divergencias += 1
                log(f"[P6-QTD] {lote_dir.name}/{bloco_dir.name}: esperado={esperado} encontrado={qtd}")

            for tif in tifs:
                if not is_valid_mds_tif_name(tif.stem, lote_dir.name, bloco_dir.name):
                    novo = aplicar_regra(tif.stem, "_MDS_")
                    if novo:
                        dest = tif.with_name(f"{novo}{tif.suffix.lower()}")
                        if dest != tif:
                            log(f"[P6-NOME] {tif.name} -> {dest.name}")
                            move_sidecar_prj(tif, dest, log, dry_run)
                            move_or_rename(tif, dest, log, dry_run)
                            tif = dest
                    if not is_valid_mds_tif_name(tif.stem, lote_dir.name, bloco_dir.name):
                        divergencias += 1
                        log(f"[P6-NOME-ALERTA] Nome fora do padrao: {tif}")

                try:
                    epsg = ensure_tif_epsg(tif, 31982, force_epsg=True)
                    if epsg != 31982:
                        divergencias += 1
                        log(f"[P6-EPSG-ALERTA] Falha ao ajustar EPSG 31982: {tif} (atual={epsg})")
                except Exception as e:
                    divergencias += 1
                    log(f"[P6-EPSG-ALERTA] Erro EPSG em {tif}: {e}")

                bands = get_tif_band_count(tif)
                if bands is None:
                    divergencias += 1
                    log(f"[P6-BANDAS-ALERTA] Nao foi possivel ler numero de bandas: {tif}")
                elif bands > 1:
                    enforce_tif_band_count(tif, 1, log, dry_run)
                    bands2 = get_tif_band_count(tif)
                    if bands2 != 1:
                        divergencias += 1
                        log(f"[P6-BANDAS-ALERTA] Falha apos ajuste, bandas={bands2}: {tif}")
                elif bands < 1:
                    divergencias += 1
                    log(f"[P6-BANDAS-ALERTA] Esperado 1 banda, encontrado {bands}: {tif}")

    if divergencias == 0:
        log("[P6-OK] Revisao da pasta 6_MDS concluida sem divergencias.")
    else:
        log(f"[P6-ALERTA] Revisao da pasta 6_MDS com {divergencias} divergencias.")


def expected_mdt_lasd_name(lote: str, bloco: str) -> str:
    m_lote = re.search(r"(\d{1,2})", lote)
    yy = m_lote.group(1).zfill(2) if m_lote else "00"
    m_bloco = re.search(r"BLOCO_([A-Z])", normalize_part(bloco))
    bx = m_bloco.group(1) if m_bloco else "X"
    return f"ES_L{yy}_{bx}_MDT_R0.lasd"


def enforce_tif_band_count(tif_path: Path, target_count: int, log, dry_run: bool) -> None:
    count = get_tif_band_count(tif_path)
    if count is None:
        log(f"[BANDAS-ALERTA] Nao foi possivel ler bandas: {tif_path}")
        return
    if count <= target_count:
        return

    log(f"[BANDAS] Ajustando para {target_count} bandas: {tif_path.name} ({count} -> {target_count})")
    if dry_run:
        return

    try:
        if _HAS_GDAL:
            ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError("GDAL nao abriu arquivo para ajuste de bandas.")
            tmp = tif_path.with_name(f"{tif_path.stem}__tmp3{tif_path.suffix.lower()}")
            opts = gdal.TranslateOptions(bandList=list(range(1, target_count + 1)))
            out = gdal.Translate(str(tmp), ds, options=opts)
            ds = None
            out = None
            shutil.move(str(tmp), str(tif_path))
            return

        if _HAS_RASTERIO:
            with rasterio.open(str(tif_path), "r") as src:
                profile = src.profile.copy()
                data = src.read(indexes=list(range(1, target_count + 1)))
            profile.update(count=target_count)
            fd, tmp_name = tempfile.mkstemp(suffix=tif_path.suffix.lower())
            os.close(fd)
            tmp = Path(tmp_name)
            with rasterio.open(str(tmp), "w", **profile) as dst:
                dst.write(data)
            shutil.move(str(tmp), str(tif_path))
            return
    except Exception as e:
        log(f"[BANDAS-ALERTA] Falha ao reduzir bandas de {tif_path}: {e}")


def convert_las_to_txt(src_las: Path, dst_txt: Path, log, dry_run: bool) -> None:
    if dst_txt.exists():
        return
    log(f"[P7-ASCII] Criando {dst_txt.name} a partir de {src_las.name}")
    if dry_run:
        return
    try:
        import laspy
    except Exception as e:
        log(f"[P7-ASCII-ALERTA] laspy indisponivel para converter {src_las}: {e}")
        return

    try:
        las = laspy.read(src_las)
        dst_txt.parent.mkdir(parents=True, exist_ok=True)
        with dst_txt.open("w", encoding="utf-8") as f:
            for x, y, z in zip(las.x, las.y, las.z):
                f.write(f"{x:.3f} {y:.3f} {z:.3f}\n")
    except Exception as e:
        log(f"[P7-ASCII-ALERTA] Erro convertendo {src_las} -> {dst_txt}: {e}")


def enforce_pasta_7_mdt_first_level(base_dir: Path, log, dry_run: bool) -> None:
    top = base_dir / "7_MDT"
    if not top.is_dir():
        return
    apagar_root = base_dir.parent / "apagar" / base_dir.name
    expected = {"1_LAS", "2_LASDATASET", "3_ASCII", "4_GEOTIFF"}

    for lote_dir in [d for d in top.iterdir() if d.is_dir() and is_valid_lote(d.name)]:
        for bloco_dir in [d for d in lote_dir.iterdir() if d.is_dir() and is_valid_bloco(d.name)]:
            for child in [d for d in bloco_dir.iterdir() if d.is_dir()]:
                expected_name = expected_subdir_name("7_MDT", child.name)
                if expected_name and child.name != expected_name:
                    dest = bloco_dir / expected_name
                    if dest.exists():
                        log(f"[P7-ORG-MERGE] {child} -> {dest}")
                        if not dry_run:
                            merge_dir_contents(child, dest, log, dry_run)
                    else:
                        log(f"[P7-ORG-RENAME] {child} -> {dest}")
                        if not dry_run:
                            child.rename(dest)

            for item in list(bloco_dir.iterdir()):
                if item.is_dir() and item.name in expected:
                    continue
                rel = item.relative_to(base_dir)
                dest = apagar_root / rel
                kind = "pasta" if item.is_dir() else "arquivo"
                log(f"[P7-ORG-APAGAR] {kind} fora do padrao: {item} -> {dest}")
                if dry_run:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(dest))


def review_pasta_7_mdt(base_dir: Path, log, dry_run: bool) -> None:
    top = base_dir / "7_MDT"
    if not top.is_dir():
        log("[P7-ALERTA] Pasta 7_MDT nao encontrada.")
        return

    divergencias = 0
    for lote_dir in [d for d in top.iterdir() if d.is_dir() and is_valid_lote(d.name)]:
        for bloco_dir in [d for d in lote_dir.iterdir() if d.is_dir() and is_valid_bloco(d.name)]:
            key = (lote_dir.name, bloco_dir.name)
            esperado = EXPECTED_FILE_COUNTS.get(key)
            if esperado is None:
                log(f"[P7-ALERTA] Sem quantidade esperada para {lote_dir.name}/{bloco_dir.name}.")
                continue

            las_dir = bloco_dir / "1_LAS"
            lasd_dir = bloco_dir / "2_LASDATASET"
            ascii_dir = bloco_dir / "3_ASCII"
            geo_dir = bloco_dir / "4_GEOTIFF"

            las_files = (
                [p for p in las_dir.iterdir() if p.is_file() and p.suffix.lower() in {".las", ".laz"}]
                if las_dir.is_dir()
                else []
            )
            if len(las_files) != esperado:
                divergencias += 1
                log(
                    f"[P7-QTD] 1_LAS {lote_dir.name}/{bloco_dir.name}: esperado={esperado} encontrado={len(las_files)}"
                )

            # Renomeia LAS para _MDT_ quando aplicavel.
            for las in list(las_files):
                novo = aplicar_regra(las.stem, "_MDT_")
                if novo:
                    dest = las.with_name(f"{novo}{las.suffix.lower()}")
                    if dest != las:
                        log(f"[P7-NOME] {las.name} -> {dest.name}")
                        move_sidecar_prj(las, dest, log, dry_run)
                        move_or_rename(las, dest, log, dry_run)

            las_files = (
                [p for p in las_dir.iterdir() if p.is_file() and p.suffix.lower() in {".las", ".laz"}]
                if las_dir.is_dir()
                else []
            )
            for las in las_files:
                if "_MDT_" not in las.stem.upper():
                    divergencias += 1
                    log(f"[P7-NOME-ALERTA] LAS sem token _MDT_: {las}")

            # LASDATASET: exatamente 1 .lasd com nome padrao.
            lasd_files = (
                [p for p in lasd_dir.iterdir() if p.is_file() and p.suffix.lower() == ".lasd"]
                if lasd_dir.is_dir()
                else []
            )
            nome_lasd = expected_mdt_lasd_name(lote_dir.name, bloco_dir.name)
            if len(lasd_files) != 1:
                divergencias += 1
                log(
                    f"[P7-LASD] {lote_dir.name}/{bloco_dir.name}: esperado 1 .lasd, encontrado {len(lasd_files)}"
                )
            elif lasd_files[0].name != nome_lasd:
                divergencias += 1
                dest = lasd_files[0].with_name(nome_lasd)
                log(f"[P7-LASD] Renomeando {lasd_files[0].name} -> {nome_lasd}")
                move_sidecar_prj(lasd_files[0], dest, log, dry_run)
                move_or_rename(lasd_files[0], dest, log, dry_run)

            # ASCII: criar .txt faltante para cada LAS (mesmo stem).
            if ascii_dir.is_dir():
                for las in las_files:
                    out_txt = ascii_dir / f"{las.stem}.txt"
                    convert_las_to_txt(las, out_txt, log, dry_run)
            else:
                divergencias += 1
                log(f"[P7-ASCII-ALERTA] Pasta ausente: {ascii_dir}")

            ascii_txts = (
                [p for p in ascii_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
                if ascii_dir.is_dir()
                else []
            )
            if len(ascii_txts) != len(las_files):
                divergencias += 1
                log(
                    f"[P7-ASCII-ALERTA] Quantidade .txt difere de 1_LAS em {lote_dir.name}/{bloco_dir.name}: txt={len(ascii_txts)} las={len(las_files)}"
                )

            # GEOTIFF: nome, EPSG, bandas e quantidade.
            tifs = (
                [p for p in geo_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
                if geo_dir.is_dir()
                else []
            )
            if len(tifs) != esperado:
                divergencias += 1
                log(
                    f"[P7-QTD] 4_GEOTIFF {lote_dir.name}/{bloco_dir.name}: esperado={esperado} encontrado={len(tifs)}"
                )

            for tif in list(tifs):
                novo = aplicar_regra(tif.stem, "_MDT_")
                if novo:
                    dest = tif.with_name(f"{novo}{tif.suffix.lower()}")
                    if dest != tif:
                        log(f"[P7-NOME] {tif.name} -> {dest.name}")
                        move_sidecar_prj(tif, dest, log, dry_run)
                        move_or_rename(tif, dest, log, dry_run)
                        tif = dest
                if "_MDT_" not in tif.stem.upper():
                    divergencias += 1
                    log(f"[P7-NOME-ALERTA] TIFF sem token _MDT_: {tif}")

                try:
                    epsg = ensure_tif_epsg(tif, 31982, force_epsg=True)
                    if epsg != 31982:
                        divergencias += 1
                        log(f"[P7-EPSG-ALERTA] Falha ao ajustar EPSG 31982: {tif} (atual={epsg})")
                except Exception as e:
                    divergencias += 1
                    log(f"[P7-EPSG-ALERTA] Erro EPSG em {tif}: {e}")

                bands = get_tif_band_count(tif)
                if bands is None:
                    divergencias += 1
                    log(f"[P7-BANDAS-ALERTA] Nao foi possivel ler bandas: {tif}")
                elif bands > 1:
                    enforce_tif_band_count(tif, 1, log, dry_run)
                    bands2 = get_tif_band_count(tif)
                    if bands2 != 1:
                        divergencias += 1
                        log(f"[P7-BANDAS-ALERTA] Falha apos ajuste, bandas={bands2}: {tif}")
                elif bands < 1:
                    divergencias += 1
                    log(f"[P7-BANDAS-ALERTA] Esperado 1 banda, encontrado {bands}: {tif}")

    if divergencias == 0:
        log("[P7-OK] Revisao da pasta 7_MDT concluida sem divergencias.")
    else:
        log(f"[P7-ALERTA] Revisao da pasta 7_MDT com {divergencias} divergencias.")


def is_valid_hc_tif_name(stem: str, lote: str, bloco: str) -> bool:
    m_lote = re.search(r"(\d{1,2})", lote)
    yy = m_lote.group(1).zfill(2) if m_lote else None
    m_bloco = re.search(r"BLOCO_([A-Z])", normalize_part(bloco))
    bx = m_bloco.group(1) if m_bloco else None
    if not yy or not bx:
        return False
    s = stem.upper()
    if "_IMG_HC_" not in s:
        return False
    if not s.endswith("_R0"):
        return False
    return s.startswith(f"ES_L{yy}_{bx}_")


def normalize_hc_tif_stem(stem: str) -> str:
    s = stem.replace("-", "_")
    s = re.sub(r"_IMG_(HC|INTENS)_", "_IMG_HC_", s, flags=re.IGNORECASE)
    s = re.sub(r"_R\d+$", "", s, flags=re.IGNORECASE)
    return f"{s}_R0"


def enforce_pasta_8_first_level(base_dir: Path, log, dry_run: bool) -> None:
    top = base_dir / "8_IMG_HIPSOMETRICA_COMPOSTA"
    if not top.is_dir():
        return
    apagar_root = base_dir.parent / "apagar" / base_dir.name
    expected = {"1_GEOTIFF", "2_METADADOS"}

    for lote_dir in [d for d in top.iterdir() if d.is_dir() and is_valid_lote(d.name)]:
        for bloco_dir in [d for d in lote_dir.iterdir() if d.is_dir() and is_valid_bloco(d.name)]:
            for child in [d for d in bloco_dir.iterdir() if d.is_dir()]:
                expected_name = expected_subdir_name("8_IMG_HIPSOMETRICA_COMPOSTA", child.name)
                if expected_name and child.name != expected_name:
                    dest = bloco_dir / expected_name
                    if dest.exists():
                        log(f"[P8-ORG-MERGE] {child} -> {dest}")
                        if not dry_run:
                            merge_dir_contents(child, dest, log, dry_run)
                    else:
                        log(f"[P8-ORG-RENAME] {child} -> {dest}")
                        if not dry_run:
                            child.rename(dest)

            for item in list(bloco_dir.iterdir()):
                if item.is_dir() and item.name in expected:
                    continue
                rel = item.relative_to(base_dir)
                dest = apagar_root / rel
                kind = "pasta" if item.is_dir() else "arquivo"
                log(f"[P8-ORG-APAGAR] {kind} fora do padrao: {item} -> {dest}")
                if dry_run:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(dest))


def review_pasta_8_hc(base_dir: Path, log, dry_run: bool) -> None:
    top = base_dir / "8_IMG_HIPSOMETRICA_COMPOSTA"
    if not top.is_dir():
        log("[P8-ALERTA] Pasta 8_IMG_HIPSOMETRICA_COMPOSTA nao encontrada.")
        return

    divergencias = 0
    las_files = index_las_files(base_dir)

    for lote_dir in [d for d in top.iterdir() if d.is_dir() and is_valid_lote(d.name)]:
        for bloco_dir in [d for d in lote_dir.iterdir() if d.is_dir() and is_valid_bloco(d.name)]:
            key = (lote_dir.name, bloco_dir.name)
            esperado = EXPECTED_FILE_COUNTS.get(key)
            if esperado is None:
                log(f"[P8-ALERTA] Sem quantidade esperada para {lote_dir.name}/{bloco_dir.name}.")
                continue

            geo_dir = bloco_dir / "1_GEOTIFF"
            meta_dir = bloco_dir / "2_METADADOS"

            if not geo_dir.is_dir():
                divergencias += 1
                log(f"[P8-ALERTA] Pasta ausente: {geo_dir}")
                continue
            if not meta_dir.is_dir():
                log(f"[P8-ALERTA] Pasta ausente: {meta_dir}")
                if not dry_run:
                    meta_dir.mkdir(parents=True, exist_ok=True)

            # Reporta arquivos fora do padrao de extensao nas subpastas.
            for p in geo_dir.iterdir():
                if p.is_file() and p.suffix.lower() not in {".tif", ".tiff", ".prj"}:
                    divergencias += 1
                    log(f"[P8-ARQ-ALERTA] Arquivo inesperado em 1_GEOTIFF: {p}")
            if meta_dir.is_dir():
                for p in meta_dir.iterdir():
                    if p.is_file() and p.suffix.lower() != ".txt":
                        divergencias += 1
                        log(f"[P8-ARQ-ALERTA] Arquivo inesperado em 2_METADADOS: {p}")

            tifs = [p for p in geo_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
            if len(tifs) != esperado:
                divergencias += 1
                log(
                    f"[P8-QTD] 1_GEOTIFF {lote_dir.name}/{bloco_dir.name}: esperado={esperado} encontrado={len(tifs)}"
                )

            for tif in list(tifs):
                if not is_valid_hc_tif_name(tif.stem, lote_dir.name, bloco_dir.name):
                    novo_stem = normalize_hc_tif_stem(tif.stem)
                    dest = tif.with_name(f"{novo_stem}{tif.suffix.lower()}")
                    if dest != tif:
                        log(f"[P8-NOME] {tif.name} -> {dest.name}")
                        move_sidecar_prj(tif, dest, log, dry_run)
                        move_or_rename(tif, dest, log, dry_run)
                        tif = dest
                    if not is_valid_hc_tif_name(tif.stem, lote_dir.name, bloco_dir.name):
                        divergencias += 1
                        log(f"[P8-NOME-ALERTA] Nome fora do padrao: {tif}")

                try:
                    epsg = ensure_tif_epsg(tif, 31982, force_epsg=True)
                    if epsg != 31982:
                        divergencias += 1
                        log(f"[P8-EPSG-ALERTA] Falha ao ajustar EPSG 31982: {tif} (atual={epsg})")
                except Exception as e:
                    divergencias += 1
                    log(f"[P8-EPSG-ALERTA] Erro EPSG em {tif}: {e}")

                bands = get_tif_band_count(tif)
                if bands is None:
                    divergencias += 1
                    log(f"[P8-BANDAS-ALERTA] Nao foi possivel ler bandas: {tif}")
                elif bands > 3:
                    enforce_tif_band_count(tif, 3, log, dry_run)
                    bands2 = get_tif_band_count(tif)
                    if bands2 != 3:
                        divergencias += 1
                        log(f"[P8-BANDAS-ALERTA] Falha apos ajuste, bandas={bands2}: {tif}")
                elif bands < 3:
                    divergencias += 1
                    log(f"[P8-BANDAS-ALERTA] Esperado 3 bandas, encontrado {bands}: {tif}")

                # Metadado: se existir, pula; se faltar, gera via LAS.
                out_txt = meta_dir / f"{tif.stem}.txt"
                if out_txt.exists():
                    continue

                code = extract_code_from_product_stem(tif.stem)
                las_match = find_matching_las(las_files, code=code, lote=lote_dir.name, bloco=bloco_dir.name)
                if las_match is None:
                    divergencias += 1
                    log(f"[P8-META-ALERTA] Sem LAS para gerar metadado de {tif.name}")
                    continue

                log(f"[P8-META] Gerando {out_txt.name} a partir de {las_match.name}")
                if not dry_run:
                    try:
                        metadata = build_las_metadata(las_match, produto="hipsometrica_composta")
                        out_txt.write_text(las_metadata_to_text(metadata), encoding="utf-8")
                    except Exception as e:
                        divergencias += 1
                        log(f"[P8-META-ALERTA] Erro ao gerar metadado {out_txt}: {e}")

    if divergencias == 0:
        log("[P8-OK] Revisao da pasta 8_IMG_HIPSOMETRICA_COMPOSTA concluida sem divergencias.")
    else:
        log(f"[P8-ALERTA] Revisao da pasta 8_IMG_HIPSOMETRICA_COMPOSTA com {divergencias} divergencias.")


def is_valid_int_tif_name(stem: str, lote: str, bloco: str) -> bool:
    m_lote = re.search(r"(\d{1,2})", lote)
    yy = m_lote.group(1).zfill(2) if m_lote else None
    m_bloco = re.search(r"BLOCO_([A-Z])", normalize_part(bloco))
    bx = m_bloco.group(1) if m_bloco else None
    if not yy or not bx:
        return False
    s = stem.upper()
    if "_IMG_INTENS_" not in s:
        return False
    if not s.endswith("_R0"):
        return False
    return s.startswith(f"ES_L{yy}_{bx}_")


def normalize_int_tif_stem(stem: str) -> str:
    s = stem.replace("-", "_")
    s = re.sub(r"_IMG_(HC|INTENS)_", "_IMG_INTENS_", s, flags=re.IGNORECASE)
    s = re.sub(r"_R\d+$", "", s, flags=re.IGNORECASE)
    return f"{s}_R0"


def enforce_pasta_9_first_level(base_dir: Path, log, dry_run: bool) -> None:
    top = base_dir / "9_IMG_INTENSIDADE"
    if not top.is_dir():
        return
    apagar_root = base_dir.parent / "apagar" / base_dir.name
    expected = {"1_GEOTIFF", "2_METADADOS"}

    for lote_dir in [d for d in top.iterdir() if d.is_dir() and is_valid_lote(d.name)]:
        for bloco_dir in [d for d in lote_dir.iterdir() if d.is_dir() and is_valid_bloco(d.name)]:
            for child in [d for d in bloco_dir.iterdir() if d.is_dir()]:
                expected_name = expected_subdir_name("9_IMG_INTENSIDADE", child.name)
                if expected_name and child.name != expected_name:
                    dest = bloco_dir / expected_name
                    if dest.exists():
                        log(f"[P9-ORG-MERGE] {child} -> {dest}")
                        if not dry_run:
                            merge_dir_contents(child, dest, log, dry_run)
                    else:
                        log(f"[P9-ORG-RENAME] {child} -> {dest}")
                        if not dry_run:
                            child.rename(dest)

            for item in list(bloco_dir.iterdir()):
                if item.is_dir() and item.name in expected:
                    continue
                rel = item.relative_to(base_dir)
                dest = apagar_root / rel
                kind = "pasta" if item.is_dir() else "arquivo"
                log(f"[P9-ORG-APAGAR] {kind} fora do padrao: {item} -> {dest}")
                if dry_run:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(dest))


def review_pasta_9_int(base_dir: Path, log, dry_run: bool) -> None:
    top = base_dir / "9_IMG_INTENSIDADE"
    if not top.is_dir():
        log("[P9-ALERTA] Pasta 9_IMG_INTENSIDADE nao encontrada.")
        return

    divergencias = 0
    las_files = index_las_files(base_dir)

    for lote_dir in [d for d in top.iterdir() if d.is_dir() and is_valid_lote(d.name)]:
        for bloco_dir in [d for d in lote_dir.iterdir() if d.is_dir() and is_valid_bloco(d.name)]:
            key = (lote_dir.name, bloco_dir.name)
            esperado = EXPECTED_FILE_COUNTS.get(key)
            if esperado is None:
                log(f"[P9-ALERTA] Sem quantidade esperada para {lote_dir.name}/{bloco_dir.name}.")
                continue

            geo_dir = bloco_dir / "1_GEOTIFF"
            meta_dir = bloco_dir / "2_METADADOS"

            if not geo_dir.is_dir():
                divergencias += 1
                log(f"[P9-ALERTA] Pasta ausente: {geo_dir}")
                continue
            if not meta_dir.is_dir():
                log(f"[P9-ALERTA] Pasta ausente: {meta_dir}")
                if not dry_run:
                    meta_dir.mkdir(parents=True, exist_ok=True)

            for p in geo_dir.iterdir():
                if p.is_file() and p.suffix.lower() not in {".tif", ".tiff", ".prj"}:
                    divergencias += 1
                    log(f"[P9-ARQ-ALERTA] Arquivo inesperado em 1_GEOTIFF: {p}")
            if meta_dir.is_dir():
                for p in meta_dir.iterdir():
                    if p.is_file() and p.suffix.lower() != ".txt":
                        divergencias += 1
                        log(f"[P9-ARQ-ALERTA] Arquivo inesperado em 2_METADADOS: {p}")

            tifs = [p for p in geo_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
            if len(tifs) != esperado:
                divergencias += 1
                log(
                    f"[P9-QTD] 1_GEOTIFF {lote_dir.name}/{bloco_dir.name}: esperado={esperado} encontrado={len(tifs)}"
                )

            for tif in list(tifs):
                if not is_valid_int_tif_name(tif.stem, lote_dir.name, bloco_dir.name):
                    novo_stem = normalize_int_tif_stem(tif.stem)
                    dest = tif.with_name(f"{novo_stem}{tif.suffix.lower()}")
                    if dest != tif:
                        log(f"[P9-NOME] {tif.name} -> {dest.name}")
                        move_sidecar_prj(tif, dest, log, dry_run)
                        move_or_rename(tif, dest, log, dry_run)
                        tif = dest
                    if not is_valid_int_tif_name(tif.stem, lote_dir.name, bloco_dir.name):
                        divergencias += 1
                        log(f"[P9-NOME-ALERTA] Nome fora do padrao: {tif}")

                try:
                    epsg = ensure_tif_epsg(tif, 31982, force_epsg=True)
                    if epsg != 31982:
                        divergencias += 1
                        log(f"[P9-EPSG-ALERTA] Falha ao ajustar EPSG 31982: {tif} (atual={epsg})")
                except Exception as e:
                    divergencias += 1
                    log(f"[P9-EPSG-ALERTA] Erro EPSG em {tif}: {e}")

                bands = get_tif_band_count(tif)
                if bands is None:
                    divergencias += 1
                    log(f"[P9-BANDAS-ALERTA] Nao foi possivel ler bandas: {tif}")
                elif bands > 1:
                    enforce_tif_band_count(tif, 1, log, dry_run)
                    bands2 = get_tif_band_count(tif)
                    if bands2 != 1:
                        divergencias += 1
                        log(f"[P9-BANDAS-ALERTA] Falha apos ajuste, bandas={bands2}: {tif}")
                elif bands < 1:
                    divergencias += 1
                    log(f"[P9-BANDAS-ALERTA] Esperado 1 banda, encontrado {bands}: {tif}")

                out_txt = meta_dir / f"{tif.stem}.txt"
                if out_txt.exists():
                    continue

                code = extract_code_from_product_stem(tif.stem)
                las_match = find_matching_las(las_files, code=code, lote=lote_dir.name, bloco=bloco_dir.name)
                if las_match is None:
                    divergencias += 1
                    log(f"[P9-META-ALERTA] Sem LAS para gerar metadado de {tif.name}")
                    continue

                log(f"[P9-META] Gerando {out_txt.name} a partir de {las_match.name}")
                if not dry_run:
                    try:
                        metadata = build_las_metadata(las_match, produto="intensidade")
                        out_txt.write_text(las_metadata_to_text(metadata), encoding="utf-8")
                    except Exception as e:
                        divergencias += 1
                        log(f"[P9-META-ALERTA] Erro ao gerar metadado {out_txt}: {e}")

            # Conferencia de quantidade de metadados apos possivel geracao.
            txts = (
                [p for p in meta_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
                if meta_dir.is_dir()
                else []
            )
            if len(txts) != esperado:
                divergencias += 1
                log(
                    f"[P9-QTD] 2_METADADOS {lote_dir.name}/{bloco_dir.name}: esperado={esperado} encontrado={len(txts)}"
                )

    if divergencias == 0:
        log("[P9-OK] Revisao da pasta 9_IMG_INTENSIDADE concluida sem divergencias.")
    else:
        log(f"[P9-ALERTA] Revisao da pasta 9_IMG_INTENSIDADE com {divergencias} divergencias.")


def detect_prefix(parts: list[str]) -> str:
    return ("_".join(parts[:3]) + "_") if len(parts) >= 3 else ("_".join(parts) + ("_" if parts else ""))


def normalize_code(token: str) -> str:
    return token.replace("-", "_").replace(" ", "_")


def categorize_suffix(raw_suffix: str) -> Optional[str]:
    s = raw_suffix.strip("_").lower()
    if s in {"hc", "int", "xyz"}:
        return s
    return None


def build_new_name_img(prefix: str, category: str, code: str) -> str:
    tag = "IMG_HC" if category == "hc" else "IMG_INTENS"
    return f"{prefix}{tag}_{code}_R0"


def parse_imagem_name(filename: str) -> Tuple[Optional[str], Optional[str]]:
    stem = Path(filename).stem
    parts = [p for p in re.split(r"[_-]+", stem) if p]
    if len(parts) < 5:
        return (None, None)
    prefix = detect_prefix(parts[:3])

    # Caso 1: arquivo cru no formato ..._<codigo>_HC/INT/XYZ
    raw_suffix = parts[-1]
    category = categorize_suffix(raw_suffix)
    if category is not None:
        mid_tokens = parts[3:-1]
        if not mid_tokens:
            return (None, None)
        code = normalize_code("_".join(mid_tokens))
        return build_new_name_img(prefix, category, code), category

    # Caso 2: arquivo ja padronizado contendo IMG_HC / IMG_INTENS.
    up = stem.upper()
    if "_IMG_HC_" in up:
        return (normalize_hc_tif_stem(stem), "hc")
    if "_IMG_INTENS_" in up:
        return (normalize_int_tif_stem(stem), "int")
    return (None, None)


def normalize_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def extract_sheet_id_from_stem(stem: str) -> Optional[str]:
    # Extrai apenas o identificador da folha (ex.: 2863_3_SE_A_III),
    # independente do tipo de produto (IMG_HC/IMG_INTENS/NP/NPc/MDT/MDS).
    m = re.match(
        r"^ES_L\d{2}_[A-Z]_(?:IMG_(?:HC|INTENS)|NPc?_C|NPc?_T|NP|MDT|MDS)_(.+?)_R\d+$",
        stem,
        flags=re.IGNORECASE,
    )
    if m:
        return normalize_token(m.group(1))

    # Fallback: pega bloco cartografico no final para evitar confundir I/II/III.
    m2 = re.search(
        r"(\d{4}[_-]\d[_-](?:SE|SO|NE|NO)[_-][A-Z][_-](?:I|II|III|IV|V))(?:(?:[_-]R\d+))?$",
        stem,
        flags=re.IGNORECASE,
    )
    if m2:
        return normalize_token(m2.group(1))
    return None


def extract_code_from_product_stem(stem: str) -> Optional[str]:
    # Para produtos IMG_*, retorna o identificador canonico da folha.
    return extract_sheet_id_from_stem(stem)


def index_las_files(base_dir: Path) -> list[Path]:
    las_files: list[Path] = []
    for p in iter_files_safely(base_dir):
        if p.is_file() and p.suffix.lower() in {".las", ".laz"}:
            las_files.append(p)
    return las_files


def find_matching_las(
    las_files: list[Path], code: Optional[str], lote: Optional[str], bloco: Optional[str]
) -> Optional[Path]:
    if not las_files:
        return None

    code_n = normalize_token(code or "")
    lote_n = normalize_token(lote or "")
    bloco_n = normalize_token(bloco or "")

    # Sem codigo, evita match fraco baseado apenas em lote/bloco.
    if not code_n:
        return None

    # Match exato por identificador da folha para evitar colisoes
    # entre I / II / III etc.
    exact = [p for p in las_files if extract_sheet_id_from_stem(p.stem) == code_n]
    if not exact:
        return None

    # Se houver mais de um candidato exato, desempata por lote/bloco.
    if len(exact) == 1:
        return exact[0]

    def score(path: Path) -> int:
        text = normalize_token(str(path))
        s = 0
        if lote_n and lote_n in text:
            s += 20
        if bloco_n and bloco_n in text:
            s += 20
        return s

    ranked = sorted(((score(p), p) for p in exact), key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def find_lote_bloco(parts: list[str]) -> Tuple[Optional[str], Optional[str]]:
    lote = None
    bloco = None

    def normalize_bloco(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        text = raw.strip()
        patterns = [
            r"^BLOCO[_-]?([A-Z])$",
            r"^B([A-Z])$",
            r"^([A-Z])$",
        ]
        for pattern in patterns:
            m = re.fullmatch(pattern, text, flags=re.IGNORECASE)
            if m:
                return f"BLOCO_{m.group(1).upper()}"
        return None

    for idx, p in enumerate(parts):
        up = p.upper()
        if up.startswith("LOTE_"):
            lote = p
        elif up == "LOTE" and idx + 1 < len(parts):
            lote = f"LOTE_{parts[idx + 1]}"
        elif up.startswith("BLOCO_"):
            bloco = normalize_bloco(p) or bloco
        elif up == "BLOCO" and idx + 1 < len(parts):
            bloco = normalize_bloco(parts[idx + 1]) or bloco
        elif bloco is None:
            bloco = normalize_bloco(p)
    return lote, bloco


def find_lote_bloco_in_name(stem: str) -> Tuple[Optional[str], Optional[str]]:
    # Aceita variantes como LOTE_09, LOTE-09, LOTE09 e BLOCO_A..BLOCO_Z.
    lote = None
    bloco = None

    def normalize_bloco(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        text = raw.strip()
        patterns = [
            r"^BLOCO[_-]?([A-Z])$",
            r"^B([A-Z])$",
            r"^([A-Z])$",
        ]
        for pattern in patterns:
            m = re.fullmatch(pattern, text, flags=re.IGNORECASE)
            if m:
                return f"BLOCO_{m.group(1).upper()}"
        return None

    tokens = re.split(r"[_-]+", stem)
    for t in tokens:
        up = t.upper()
        if not lote and re.fullmatch(r"L\d{1,2}", up):
            lote = f"LOTE_{up[1:].zfill(2)}"
        if not bloco:
            bloco = normalize_bloco(up)
    m = re.search(r"\bLOTE[_-]?([A-Za-z0-9]+)\b", stem, flags=re.IGNORECASE)
    if m:
        lote = f"LOTE_{m.group(1)}"
    m = re.search(r"\bBLOCO[_-]?([A-Za-z0-9]+)\b", stem, flags=re.IGNORECASE)
    if m:
        bloco = normalize_bloco(m.group(1)) or bloco
    # Padrao alternativo: ES_L09_B_... (Lote e Bloco abreviados)
    if not lote:
        m = re.search(r"(?:^|[_-])L(\d{1,2})(?:[_-]|$)", stem, flags=re.IGNORECASE)
        if m:
            lote = f"LOTE_{m.group(1).zfill(2)}"
    if not bloco:
        m = re.search(r"(?:^|[_-])B([A-Z])(?:[_-]|$)", stem, flags=re.IGNORECASE)
        if m:
            bloco = normalize_bloco("B" + m.group(1))
        else:
            m = re.search(r"(?:^|[_-])([A-Z])(?:[_-]|$)", stem, flags=re.IGNORECASE)
            if m:
                bloco = normalize_bloco(m.group(1))
    return lote, bloco


def move_or_rename(src: Path, dest: Path, log, dry_run: bool) -> None:
    if src.resolve() == dest.resolve():
        return
    log(f"[MOVE] {src} -> {dest}")
    if src.suffix.lower() in {".las", ".laz", ".lasd"}:
        update_prj_internal_references(src, dest, log, dry_run)
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))


def move_sidecar_prj(src: Path, dest: Path, log, dry_run: bool) -> None:
    # Mantem arquivo .prj com mesmo stem sincronizado com o arquivo principal.
    if src.suffix.lower() == ".prj":
        return
    src_prj = src.with_suffix(".prj")
    if not src_prj.exists():
        return
    dest_prj = dest.with_suffix(".prj")
    move_or_rename(src_prj, dest_prj, log, dry_run)


def find_bloco_root(path: Path) -> Optional[Path]:
    for parent in [path.parent] + list(path.parents):
        if re.fullmatch(r"BLOCO_[A-Z]", normalize_part(parent.name)):
            return parent
    return None


def update_prj_internal_references(
    src: Path,
    dest: Path,
    log,
    dry_run: bool,
) -> None:
    # Atualiza referências internas em .prj (ex.: "Block <arquivo>.las")
    # quando um arquivo é renomeado/movido.
    old_name = src.name
    new_name = dest.name
    if old_name == new_name:
        return

    bloco_root = find_bloco_root(src)
    if bloco_root is None:
        bloco_root = find_bloco_root(dest)
    if bloco_root is None or not bloco_root.exists():
        return

    for prj in bloco_root.rglob("*.prj"):
        try:
            content = prj.read_text(encoding="utf-8", errors="ignore")
            updated = content.replace(old_name, new_name)
            if updated == content:
                continue
            log(f"[PRJ] Atualizando referencias em {prj.name}: {old_name} -> {new_name}")
            if not dry_run:
                prj.write_text(updated, encoding="utf-8")
        except Exception as e:
            log(f"[PRJ-ERRO] {prj}: {e}")


def desired_product_tag_for_prj(prj_path: Path) -> Optional[str]:
    parts = [normalize_part(p) for p in prj_path.parts]
    if "5_NUVEM_PONTOS" in parts:
        if "1_NP" in parts or "1_1_LAS" in parts:
            return "NP"
        if "2_NPC_COMPLETO" in parts or "2_1_LAS" in parts or "2_2_LASDATASET" in parts:
            return "NPc_C"
    if "7_MDT" in parts and ("1_LAS" in parts or "2_LASDATASET" in parts):
        return "MDT"
    return None


def replace_block_product_token(filename: str, desired_tag: str) -> str:
    # Aplica a mesma regra de renomeacao usada nos arquivos reais.
    # Isso padroniza hifen/underscore, revisao e sufixo R0.
    p = Path(filename)
    ext = p.suffix
    stem = p.stem
    novo_trecho = f"_{desired_tag}_"
    normalized = aplicar_regra(stem, novo_trecho)
    if normalized:
        return normalized + ext
    # Fallback se nao casar com aplicar_regra: troca somente o token.
    return re.sub(r"_(NPc?_C|NPc?_T|NP|MDT|MDS)_", f"_{desired_tag}_", filename, flags=re.IGNORECASE)


def enforce_prj_block_names(prj_path: Path, desired_tag: str, log, dry_run: bool) -> None:
    try:
        content = prj_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log(f"[PRJ-ERRO] leitura {prj_path}: {e}")
        return

    changed = False
    out_lines: list[str] = []
    for line in content.splitlines():
        m = re.match(r"^(\s*Block\s+)(\S+)(\s*)$", line)
        if m:
            prefix, fname, suffix = m.groups()
            new_fname = replace_block_product_token(fname, desired_tag)
            if new_fname != fname:
                changed = True
                line = f"{prefix}{new_fname}{suffix}"
        out_lines.append(line)

    if not changed:
        return

    log(f"[PRJ-BLOCK] Ajustando todos os Block em {prj_path.name} para {desired_tag}")
    if not dry_run:
        prj_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def enforce_all_prj_block_names(base_dir: Path, log, dry_run: bool) -> None:
    for prj in iter_files_safely(base_dir):
        if not prj.is_file() or prj.suffix.lower() != ".prj":
            continue
        desired_tag = desired_product_tag_for_prj(prj)
        if desired_tag:
            enforce_prj_block_names(prj, desired_tag, log, dry_run)


def process_np_mds_mdt(base_dir: Path, log, dry_run: bool) -> None:
    roots = [
        base_dir / "5_NUVEM_PONTOS",
        base_dir / "6_MDS",
        base_dir / "7_MDT",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for p in iter_files_safely(root):
            if not p.is_file():
                continue
            parts_norm = {normalize_part(s) for s in p.parts}
            rel_parts = [part for part in p.parts if normalize_part(part) in {"1_NP", "2_NPC_COMPLETO"}]
            novo_trecho = None
            if "5_NUVEM_PONTOS" in parts_norm:
                if "1_NP" in {normalize_part(s) for s in rel_parts}:
                    novo_trecho = "_NP_"
                elif "2_NPC_COMPLETO" in {normalize_part(s) for s in rel_parts}:
                    novo_trecho = "_NPc_C_"
            if "6_MDS" in parts_norm:
                novo_trecho = "_MDS_"
            if "7_MDT" in parts_norm:
                novo_trecho = "_MDT_"
            if not novo_trecho:
                continue
            if p.suffix.lower() in {".tif", ".tiff"}:
                if dry_run:
                    log(f"[EPSG] DRY-RUN {p}")
                else:
                    ensure_tif_epsg(p, 31982)
            new_base = aplicar_regra(p.stem, novo_trecho)
            if not new_base:
                continue
            dest = p.with_name(f"{new_base}{p.suffix.lower()}")
            if dest != p:
                log(f"[RENAME] {p.name} -> {dest.name}")
                move_sidecar_prj(p, dest, log, dry_run)
                move_or_rename(p, dest, log, dry_run)


def process_hc_int(
    base_dir: Path,
    log,
    dry_run: bool,
    enabled_categories: Optional[set[str]] = None,
) -> None:
    if enabled_categories is None:
        enabled_categories = {"hc", "int"}
    roots = [
        base_dir,
        base_dir / "8_IMG_HIPSOMETRICA_COMPOSTA",
        base_dir / "9_IMG_INTENSIDADE",
    ]
    products_for_metadata: list[Tuple[Path, str, Optional[str], Optional[str], Optional[str]]] = []
    las_files = index_las_files(base_dir)
    log(f"[META] LAS encontrados para metadado: {len(las_files)}")
    for root in roots:
        if not root.is_dir():
            continue
        for p in iter_files_safely(root):
            if not p.is_file():
                continue
            # Evita reprocessar arquivos que ja estao nas pastas finais
            if root == base_dir:
                parts_upper = {part.upper() for part in p.parts}
                if "8_IMG_HIPSOMETRICA_COMPOSTA" in parts_upper or "9_IMG_INTENSIDADE" in parts_upper:
                    continue
            new_base, category = parse_imagem_name(p.name)
            if not new_base or not category:
                continue
            if category not in enabled_categories:
                continue
            ext = p.suffix.lower()
            lote, bloco = find_lote_bloco(list(p.parts))
            if not lote or not bloco:
                # Tenta extrair lote/bloco do nome do arquivo
                name_parts = p.stem.split("_")
                lote2, bloco2 = find_lote_bloco(name_parts)
                lote = lote or lote2
                bloco = bloco or bloco2
            if not lote or not bloco:
                lote3, bloco3 = find_lote_bloco_in_name(p.stem)
                lote = lote or lote3
                bloco = bloco or bloco3
            if not lote or not bloco:
                log(f"[HC/INT] Sem LOTE/BLOCO para {p}")
                log(f"[HC/INT] parts={list(p.parts)}")
                log(f"[HC/INT] stem={p.stem}")
                log(f"[HC/INT] lote={lote} bloco={bloco}")
                # sem lote/bloco, renomeia no lugar
                dest = p.with_name(f"{new_base}{ext}")
                if dest != p:
                    log(f"[RENAME] {p.name} -> {dest.name}")
                    if not dry_run:
                        p.rename(dest)
                continue

            if category == "hc":
                base_root = base_dir / "8_IMG_HIPSOMETRICA_COMPOSTA" / lote / bloco
                if ext in {".tif", ".tiff"}:
                    sub = "1_GEOTIFF"
                else:
                    sub = None
                if ext in {".tif", ".tiff"}:
                    if dry_run:
                        log(f"[EPSG] DRY-RUN {p}")
                    else:
                        ensure_tif_epsg(p, 31982)
            else:
                base_root = base_dir / "9_IMG_INTENSIDADE" / lote / bloco
                if ext in {".tif", ".tiff"}:
                    sub = "1_GEOTIFF"
                else:
                    sub = None
                if ext in {".tif", ".tiff"}:
                    if dry_run:
                        log(f"[EPSG] DRY-RUN {p}")
                    else:
                        ensure_tif_epsg(p, 31982)

            if sub:
                dest = base_root / sub / f"{new_base}{ext}"
                move_sidecar_prj(p, dest, log, dry_run)
                move_or_rename(p, dest, log, dry_run)
                if ext in {".tif", ".tiff"}:
                    code = extract_code_from_product_stem(Path(new_base).stem)
                    products_for_metadata.append((dest, category, lote, bloco, code))
            else:
                dest = p.with_name(f"{new_base}{ext}")
                if dest != p:
                    log(f"[RENAME] {p.name} -> {dest.name}")
                    move_sidecar_prj(p, dest, log, dry_run)
                    move_or_rename(p, dest, log, dry_run)
                if ext in {".tif", ".tiff"}:
                    code = extract_code_from_product_stem(Path(new_base).stem)
                    products_for_metadata.append((dest, category, lote, bloco, code))

    if not _HAS_LAS_METADATA:
        log("[WARN] metadados.py nao disponivel. Pulo de metadados.")
        return

    for product_path, category, lote, bloco, code in products_for_metadata:
        try:
            if category == "hc":
                meta_dir = product_path.parent.parent / "2_METADADOS"
                produto = "hipsometrica_composta"
            else:
                meta_dir = product_path.parent.parent / "2_METADADOS"
                produto = "intensidade"

            las_match = find_matching_las(las_files, code=code, lote=lote, bloco=bloco)
            if las_match is None:
                log(f"[META-ERRO] Sem LAS correspondente para {product_path.name}")
                continue

            out_txt = meta_dir / (product_path.stem + ".txt")
            log(f"[META] {las_match.name} -> {out_txt}")
            if not dry_run:
                meta_dir.mkdir(parents=True, exist_ok=True)
                if out_txt.exists():
                    log(f"[META] Ja existe, pulando: {out_txt}")
                    continue
                metadata = build_las_metadata(las_match, produto=produto)
                meta_text = las_metadata_to_text(metadata)
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(meta_text)
        except Exception as e:
            log(f"[META-ERRO] {product_path} -> {e}")


def default_selection() -> dict[str, bool]:
    return {
        "normalize_subfolders": True,
        "ensure_structure": True,
        "migrate_legacy": True,
        "process_np_mds_mdt": True,
        "p5_org": True,
        "p5_review": True,
        "p6_org": True,
        "p6_review": True,
        "p7_org": True,
        "p7_review": True,
        "process_hc": True,
        "process_int": True,
        "p8_org": True,
        "p8_review": True,
        "p9_org": True,
        "p9_review": True,
        "prj_blocks": True,
        "send_apagar": True,
        "verify_counts": True,
    }


def run_process(base_path: Path, dry_run: bool, log, selection: Optional[dict[str, bool]] = None) -> None:
    sel = default_selection()
    if selection:
        sel.update(selection)

    if sel["normalize_subfolders"]:
        log("[INI] Normalizando nomes das subpastas numeradas...")
        normalize_numbered_subfolders(base_path, log, dry_run)
    if sel["ensure_structure"]:
        log("[INI] Validando/criando estrutura base...")
        ensure_base_structure(base_path, log, dry_run)
    if sel["migrate_legacy"]:
        log("[INI] Migrando pastas legadas de GEOTIFF...")
        migrate_legacy_geotiff_dirs(base_path, log, dry_run)

    if sel["p5_org"]:
        log("[INI] Ajustando organizacao de 5_NUVEM_PONTOS (nivel 1_NP/2_NPC_COMPLETO)...")
        enforce_pasta_5_first_level(base_path, log, dry_run)
    if sel["p6_org"]:
        log("[INI] Ajustando organizacao de 6_MDS (nivel 1_GEOTIFF)...")
        enforce_pasta_6_mds_first_level(base_path, log, dry_run)
    if sel["p7_org"]:
        log("[INI] Ajustando organizacao de 7_MDT (niveis 1..4)...")
        enforce_pasta_7_mdt_first_level(base_path, log, dry_run)

    # Fase 1: processamentos/geracoes automáticas.
    if sel["process_hc"] or sel["process_int"]:
        enabled = set()
        if sel["process_hc"]:
            enabled.add("hc")
        if sel["process_int"]:
            enabled.add("int")
        log("[INI] Processando HC/INT + metadados...")
        process_hc_int(base_path, log, dry_run, enabled_categories=enabled)

    if sel["p7_review"]:
        log("[INI] Processando regras da pasta 7_MDT (inclui ASCII e ajustes tecnicos)...")
        review_pasta_7_mdt(base_path, log, dry_run)

    # Fase 2: renomeacao/padronizacao final.
    if sel["process_np_mds_mdt"]:
        log("[INI] Renomeando e padronizando NP/MDS/MDT...")
        process_np_mds_mdt(base_path, log, dry_run)

    if sel["p8_org"]:
        log("[INI] Ajustando organizacao de 8_IMG_HIPSOMETRICA_COMPOSTA (nivel 1/2)...")
        enforce_pasta_8_first_level(base_path, log, dry_run)
    if sel["p8_review"]:
        log("[INI] Padronizando/revisando regras da pasta 8_IMG_HIPSOMETRICA_COMPOSTA...")
        review_pasta_8_hc(base_path, log, dry_run)
    if sel["p9_org"]:
        log("[INI] Ajustando organizacao de 9_IMG_INTENSIDADE (nivel 1/2)...")
        enforce_pasta_9_first_level(base_path, log, dry_run)
    if sel["p9_review"]:
        log("[INI] Padronizando/revisando regras da pasta 9_IMG_INTENSIDADE...")
        review_pasta_9_int(base_path, log, dry_run)

    if sel["p5_review"]:
        log("[INI] Revisando regras da pasta 5_NUVEM_PONTOS...")
        review_pasta_5_nuvem_pontos(base_path, log)
    if sel["p6_review"]:
        log("[INI] Revisando regras da pasta 6_MDS...")
        review_pasta_6_mds(base_path, log, dry_run)

    if sel["prj_blocks"]:
        log("[INI] Ajustando todos os Block dos arquivos PRJ...")
        enforce_all_prj_block_names(base_path, log, dry_run)
    if sel["send_apagar"]:
        log("[INI] Movendo itens fora do padrao para pasta apagar...")
        send_out_of_pattern_to_apagar(base_path, log, dry_run)
    if sel["verify_counts"]:
        log("[INI] Conferindo quantidade esperada por LOTE/BLOCO...")
        verify_expected_file_counts(base_path, log)

    log("[OK] Finalizado.")


def main_gui():
    sg.theme("SystemDefaultForReal")
    sections = {
        "GERAL": {
            "title": "Geral",
            "rows": [
                [sg.Checkbox("Normalizar subpastas", key="SEL_normalize_subfolders", default=True)],
                [sg.Checkbox("Validar/criar estrutura", key="SEL_ensure_structure", default=True)],
                [sg.Checkbox("Migrar GEOTIFF legado", key="SEL_migrate_legacy", default=True)],
                [sg.Checkbox("Ajustar Blocks em PRJ", key="SEL_prj_blocks", default=True)],
                [sg.Checkbox("Mover fora do padrao para apagar", key="SEL_send_apagar", default=True)],
                [sg.Checkbox("Conferir quantidade esperada", key="SEL_verify_counts", default=True)],
            ],
        },
        "5": {
            "title": "5_NUVEM_PONTOS",
            "rows": [
                [sg.Checkbox("Renomear NP/MDS/MDT", key="SEL_process_np_mds_mdt", default=True)],
                [sg.Checkbox("Organizacao nivel 1/2", key="SEL_p5_org", default=True)],
                [sg.Checkbox("Revisao regras P5", key="SEL_p5_review", default=True)],
            ],
        },
        "6": {
            "title": "6_MDS",
            "rows": [
                [sg.Checkbox("Organizacao nivel 1_GEOTIFF", key="SEL_p6_org", default=True)],
                [sg.Checkbox("Revisao regras P6", key="SEL_p6_review", default=True)],
            ],
        },
        "7": {
            "title": "7_MDT",
            "rows": [
                [sg.Checkbox("Organizacao nivel 1..4", key="SEL_p7_org", default=True)],
                [sg.Checkbox("Revisao regras P7", key="SEL_p7_review", default=True)],
            ],
        },
        "8": {
            "title": "8_IMG_HIPSOMETRICA_COMPOSTA",
            "rows": [
                [sg.Checkbox("Processar HC (renomear/mover/meta)", key="SEL_process_hc", default=True)],
                [sg.Checkbox("Organizacao nivel 1/2", key="SEL_p8_org", default=True)],
                [sg.Checkbox("Revisao regras P8", key="SEL_p8_review", default=True)],
            ],
        },
        "9": {
            "title": "9_IMG_INTENSIDADE",
            "rows": [
                [sg.Checkbox("Processar INT (renomear/mover/meta)", key="SEL_process_int", default=True)],
                [sg.Checkbox("Organizacao nivel 1/2", key="SEL_p9_org", default=True)],
                [sg.Checkbox("Revisao regras P9", key="SEL_p9_review", default=True)],
            ],
        },
    }

    def collapsible(sec_id: str, title: str, rows) -> list:
        return [
            [
                sg.Button("+", key=f"TOGGLE_{sec_id}", size=(2, 1), pad=((0, 6), (2, 2))),
                sg.Text(title, font=("Segoe UI", 10, "bold")),
            ],
            [
                sg.pin(
                    sg.Column(
                        rows,
                        key=f"COL_{sec_id}",
                        visible=False,
                        pad=((24, 0), (0, 4)),
                    )
                )
            ],
        ]

    section_layout = []
    for sid, cfg in sections.items():
        section_layout.extend(collapsible(sid, cfg["title"], cfg["rows"]))

    layout = [
        [sg.Text("Pasta base (contendo 5..9, cada uma com LOTE/BLOCO/subpastas):")],
        [sg.Input(key="BASE", size=(70, 1)), sg.FolderBrowse("Procurar...")],
        *section_layout,
        [sg.Checkbox("Dry-run (nao altera)", key="DRY", default=False)],
        [sg.Button("Executar", bind_return_key=True), sg.Button("Sair")],
        [sg.Multiline("", key="LOG", size=(110, 22), autoscroll=True, disabled=True)],
    ]
    window = sg.Window("Renomear Produtos (Auto)", layout, finalize=True)
    log_queue: queue.Queue[str] = queue.Queue()
    worker: Optional[threading.Thread] = None
    expanded = {sid: False for sid in sections}

    def log(msg: str):
        log_queue.put(msg)

    def flush_logs() -> None:
        while True:
            try:
                msg = log_queue.get_nowait()
            except queue.Empty:
                break
            window["LOG"].update(msg + "\n", append=True)

    def run_worker(base_dir: Path, dry: bool, selection: dict[str, bool]) -> None:
        try:
            run_process(base_dir, dry, log, selection=selection)
        except Exception as e:
            log_queue.put(f"[ERRO] {e}")
        finally:
            log_queue.put("[FIM] Processamento encerrado.")

    while True:
        ev, v = window.read(timeout=200)
        flush_logs()

        if worker is not None and not worker.is_alive():
            worker = None
            window["Executar"].update(disabled=False)

        if ev in (sg.WIN_CLOSED, "Sair"):
            break
        if isinstance(ev, str) and ev.startswith("TOGGLE_"):
            sid = ev.replace("TOGGLE_", "")
            if sid in expanded:
                expanded[sid] = not expanded[sid]
                window[f"COL_{sid}"].update(visible=expanded[sid])
                window[f"TOGGLE_{sid}"].update("-" if expanded[sid] else "+")
            continue
        if ev == "Executar":
            if worker is not None and worker.is_alive():
                continue
            window["LOG"].update("")
            base_dir = (v.get("BASE") or "").strip()
            if not base_dir or not Path(base_dir).is_dir():
                sg.popup_error("Informe uma pasta base valida.")
                continue
            selection = {
                "normalize_subfolders": bool(v.get("SEL_normalize_subfolders")),
                "ensure_structure": bool(v.get("SEL_ensure_structure")),
                "migrate_legacy": bool(v.get("SEL_migrate_legacy")),
                "process_np_mds_mdt": bool(v.get("SEL_process_np_mds_mdt")),
                "p5_org": bool(v.get("SEL_p5_org")),
                "p5_review": bool(v.get("SEL_p5_review")),
                "p6_org": bool(v.get("SEL_p6_org")),
                "p6_review": bool(v.get("SEL_p6_review")),
                "p7_org": bool(v.get("SEL_p7_org")),
                "p7_review": bool(v.get("SEL_p7_review")),
                "process_hc": bool(v.get("SEL_process_hc")),
                "process_int": bool(v.get("SEL_process_int")),
                "p8_org": bool(v.get("SEL_p8_org")),
                "p8_review": bool(v.get("SEL_p8_review")),
                "p9_org": bool(v.get("SEL_p9_org")),
                "p9_review": bool(v.get("SEL_p9_review")),
                "prj_blocks": bool(v.get("SEL_prj_blocks")),
                "send_apagar": bool(v.get("SEL_send_apagar")),
                "verify_counts": bool(v.get("SEL_verify_counts")),
            }
            if not any(selection.values()):
                sg.popup_error("Marque pelo menos uma opcao de processamento.")
                continue
            window["Executar"].update(disabled=True)
            worker = threading.Thread(
                target=run_worker,
                args=(Path(base_dir), bool(v.get("DRY")), selection),
                daemon=True,
            )
            worker.start()

    window.close()


def main():
    parser = argparse.ArgumentParser(description="Renomear e organizar produtos por padrao de pastas.")
    parser.add_argument("--base", dest="base_dir", help="Pasta base que contem 5..9 no formato TOP/LOTE/BLOCO/subpastas.")
    parser.add_argument("--gui", action="store_true", help="Abrir interface grafica (se disponivel).")
    args = parser.parse_args()

    if args.gui or (not args.base_dir and _HAS_SG):
        if not _HAS_SG:
            print("PySimpleGUI nao disponivel. Use --base no modo CLI.")
            return
        main_gui()
        return

    base_dir = args.base_dir or input("Informe a pasta base: ").strip()
    if not base_dir:
        print("Pasta base nao informada.")
        return
    base_path = Path(base_dir)
    if not base_path.is_dir():
        print("Pasta base invalida.")
        return

    def log(msg: str):
        print(msg)

    run_process(base_path, dry_run=False, log=log)


if __name__ == "__main__":
    main()
