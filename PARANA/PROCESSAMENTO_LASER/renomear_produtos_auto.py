#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Renomeia e organiza produtos (NP/MDS/MDT/HC/INT) a partir de um caminho base.
Regras:
- NP/NPc/MDS/MDT: corrige nomes conforme pasta (5_NUVEM_PONTOS / 6_MDS / 7_MDT)
- HC/INT: renomeia IMG_HC / IMG_INTENS e move para subpastas corretas
- GeoTIFF: se nao tiver EPSG, define 31982
- Metadados HC/INT: gera TXT apos renomear e ajustar EPSG
"""

from __future__ import annotations
import argparse
import os
import re
import shutil
import time
import datetime
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

_HAS_METADATA = False
if _HAS_GDAL:
    gdal.UseExceptions()
    _HAS_METADATA = True

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
    "FOUND.000",
    "Config.Msi",
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


def ensure_tif_epsg(tif_path: Path, default_epsg: int = 31982) -> Optional[int]:
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
        if not epsg:
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
            if not epsg:
                ds.crs = CRS.from_epsg(int(default_epsg))
                epsg = int(default_epsg)
        return epsg
    raise RuntimeError("GDAL/rasterio nao disponivel para consultar/definir EPSG.")


def iter_files_safely(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for name in filenames:
            yield Path(dirpath) / name


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
    parts = stem.split("_")
    if len(parts) < 5:
        return (None, None)
    prefix = detect_prefix(parts)
    raw_suffix = parts[-1]
    category = categorize_suffix(raw_suffix)
    if category is None:
        return (None, None)
    mid_tokens = parts[3:-1]
    if not mid_tokens:
        return (None, None)
    code = normalize_code(mid_tokens[-1])
    return build_new_name_img(prefix, category, code), category


def find_lote_bloco(parts: list[str]) -> Tuple[Optional[str], Optional[str]]:
    lote = None
    bloco = None
    for idx, p in enumerate(parts):
        up = p.upper()
        if up.startswith("LOTE_"):
            lote = p
        elif up == "LOTE" and idx + 1 < len(parts):
            lote = f"LOTE_{parts[idx + 1]}"
        elif up.startswith("BLOCO_"):
            bloco = p
        elif up == "BLOCO" and idx + 1 < len(parts):
            bloco = f"BLOCO_{parts[idx + 1]}"
    return lote, bloco


def find_lote_bloco_in_name(stem: str) -> Tuple[Optional[str], Optional[str]]:
    # Aceita variantes como LOTE_09, LOTE-09, LOTE09 e BLOCO_B, BLOCO-B, BLOCOB
    lote = None
    bloco = None
    tokens = re.split(r"[_-]+", stem)
    for t in tokens:
        up = t.upper()
        if not lote and re.fullmatch(r"L\d{1,2}", up):
            lote = f"LOTE_{up[1:].zfill(2)}"
        if not bloco and up == "B":
            bloco = "BLOCO_B"
        if not bloco and re.fullmatch(r"B[A-Z]", up):
            bloco = f"BLOCO_{up[1:]}"
    m = re.search(r"\bLOTE[_-]?([A-Za-z0-9]+)\b", stem, flags=re.IGNORECASE)
    if m:
        lote = f"LOTE_{m.group(1)}"
    m = re.search(r"\bBLOCO[_-]?([A-Za-z0-9]+)\b", stem, flags=re.IGNORECASE)
    if m:
        bloco = f"BLOCO_{m.group(1)}"
    # Padrao alternativo: ES_L09_B_... (Lote e Bloco abreviados)
    if not lote:
        m = re.search(r"(?:^|[_-])L(\d{1,2})(?:[_-]|$)", stem, flags=re.IGNORECASE)
        if m:
            lote = f"LOTE_{m.group(1).zfill(2)}"
    if not bloco:
        m = re.search(r"(?:^|[_-])B([A-Za-z])(?:[_-]|$)", stem, flags=re.IGNORECASE)
        if m:
            bloco = f"BLOCO_{m.group(1).upper()}"
        else:
            m = re.search(r"(?:^|[_-])B(?:[_-]|$)", stem, flags=re.IGNORECASE)
            if m:
                bloco = "BLOCO_B"
    return lote, bloco


def move_or_rename(src: Path, dest: Path, log, dry_run: bool) -> None:
    if src.resolve() == dest.resolve():
        return
    log(f"[MOVE] {src} -> {dest}")
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))


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
            rel_parts = [part for part in p.parts if part.upper() in {"1_NP", "2_NPC_COMPLETO", "3_NPC_TERRENO"}]
            novo_trecho = None
            if "5_NUVEM_PONTOS" in (s.upper() for s in p.parts):
                if "1_NP" in (s.upper() for s in rel_parts):
                    novo_trecho = "_NP_"
                elif "2_NPC_COMPLETO" in (s.upper() for s in rel_parts):
                    novo_trecho = "_NPc_C_"
                elif "3_NPC_TERRENO" in (s.upper() for s in rel_parts):
                    novo_trecho = "_NPc_T_"
            if "6_MDS" in (s.upper() for s in p.parts):
                novo_trecho = "_MDS_"
            if "7_MDT" in (s.upper() for s in p.parts):
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
                if not dry_run:
                    p.rename(dest)


def process_hc_int(base_dir: Path, log, dry_run: bool) -> None:
    roots = [
        base_dir,
        base_dir / "8_IMG_HIPSOMETRICA_COMPOSTA",
        base_dir / "9_IMG_INTENSIDADE",
    ]
    tifs_for_metadata: list[Tuple[Path, str]] = []
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
                    sub = "2_GEOTIFF"
                elif ext == ".ecw":
                    sub = "1_ECW"
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
                    sub = "3_GEOTIFF"
                elif ext == ".ecw":
                    sub = "2_ECW"
                elif ext in {".txt", ".asc", ".xyz"}:
                    sub = "1_ASCII"
                else:
                    sub = None
                if ext in {".tif", ".tiff"}:
                    if dry_run:
                        log(f"[EPSG] DRY-RUN {p}")
                    else:
                        ensure_tif_epsg(p, 31982)

            if sub:
                dest = base_root / sub / f"{new_base}{ext}"
                move_or_rename(p, dest, log, dry_run)
                if ext in {".tif", ".tiff"}:
                    tifs_for_metadata.append((dest, category))
            else:
                dest = p.with_name(f"{new_base}{ext}")
                if dest != p:
                    log(f"[RENAME] {p.name} -> {dest.name}")
                    if not dry_run:
                        p.rename(dest)
                if ext in {".tif", ".tiff"}:
                    tifs_for_metadata.append((dest, category))

    if not _HAS_METADATA:
        log("[WARN] metadados.py nao importado. Pulo de metadados.")
        return

    for tif_path, category in tifs_for_metadata:
        try:
            if category == "hc":
                meta_dir = tif_path.parent.parent / "3_METADADOS"
            else:
                meta_dir = tif_path.parent.parent / "4_METADADOS"
            out_txt = meta_dir / (tif_path.stem + ".txt")
            log(f"[META] {tif_path.name} -> {out_txt}")
            if not dry_run:
                meta_dir.mkdir(parents=True, exist_ok=True)
                meta_text = build_metadata_text(str(tif_path))
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(meta_text)
        except Exception as e:
            log(f"[META-ERRO] {tif_path} -> {e}")


def run_process(base_path: Path, dry_run: bool, do_np: bool, do_hc: bool, log) -> None:
    if do_np:
        log("[INI] Processando NP/MDS/MDT...")
        process_np_mds_mdt(base_path, log, dry_run)
    if do_hc:
        log("[INI] Processando HC/INT + metadados...")
        process_hc_int(base_path, log, dry_run)
    log("[OK] Finalizado.")


def main_gui():
    sg.theme("SystemDefaultForReal")
    layout = [
        [sg.Text("Pasta base (contendo 5_NUVEM_PONTOS/6_MDS/7_MDT/8_IMG/9_IMG):")],
        [sg.Input(key="BASE", size=(70, 1)), sg.FolderBrowse("Procurar...")],
        [sg.Checkbox("Processar NP/MDS/MDT", key="DO_NP", default=True),
         sg.Checkbox("Processar HC/INT + metadados", key="DO_HC", default=True),
         sg.Checkbox("Dry-run (nao altera)", key="DRY", default=False)],
        [sg.Button("Executar", bind_return_key=True), sg.Button("Sair")],
        [sg.Multiline("", key="LOG", size=(110, 22), autoscroll=True, disabled=True)],
    ]
    window = sg.Window("Renomear Produtos (Auto)", layout, finalize=True)

    def log(msg: str):
        window["LOG"].update(msg + "\n", append=True)

    while True:
        ev, v = window.read()
        if ev in (sg.WIN_CLOSED, "Sair"):
            break
        if ev == "Executar":
            window["LOG"].update("")
            base_dir = (v.get("BASE") or "").strip()
            if not base_dir or not Path(base_dir).is_dir():
                sg.popup_error("Informe uma pasta base valida.")
                continue
            run_process(Path(base_dir), bool(v.get("DRY")), bool(v.get("DO_NP")), bool(v.get("DO_HC")), log)

    window.close()


def main():
    parser = argparse.ArgumentParser(description="Renomear e organizar produtos por padrao de pastas.")
    parser.add_argument("--base", dest="base_dir", help="Pasta base que contem 5_NUVEM_PONTOS/6_MDS/7_MDT/8_IMG.../9_IMG...")
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

    run_process(base_path, dry_run=False, do_np=True, do_hc=True, log=log)


if __name__ == "__main__":
    main()
