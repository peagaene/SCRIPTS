from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import laspy
import numpy as np

try:
    from pyproj import CRS, Transformer
except ImportError:  # pragma: no cover - fallback when pyproj is not installed
    CRS = Any  # type: ignore[misc,assignment]
    Transformer = None


def format_pt_br_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def format_dms(value: float, is_lat: bool) -> str:
    hemisphere = "N" if is_lat else "E"
    if value < 0:
        hemisphere = "S" if is_lat else "W"

    abs_value = abs(value)
    degrees = int(abs_value)
    minutes_float = (abs_value - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    return f'{degrees}° {minutes}\' {seconds:.4f}" {hemisphere}'


def safe_epsg(crs: Any | None) -> int | None:
    if crs is None:
        return None
    try:
        return crs.to_epsg()
    except Exception:
        return None


def geotiff_ellipsoid_key(ellipsoid_name: str | None) -> str:
    if not ellipsoid_name:
        return "N/A"
    name = ellipsoid_name.lower()
    if "grs 1980" in name or "grs80" in name:
        return "7019"
    if "wgs 84" in name or "wgs84" in name:
        return "7030"
    return "N/A"


def fmt_float(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M:%S")


def display_filename_path(file_path: Path) -> str:
    full = str(file_path.resolve()).replace("/", "\\")
    markers = [
        "\\5_NUVEM_PONTOS\\",
        "\\6_MDS\\",
        "\\7_MDT\\",
        "\\8_IMG_HIPSOMETRICA_COMPOSTA\\",
        "\\9_IMG_INTENSIDADE\\",
    ]
    full_up = full.upper()
    indexes = [full_up.find(m.upper()) for m in markers if full_up.find(m.upper()) >= 0]
    if not indexes:
        return full
    idx = min(indexes)
    return "..." + full[idx:]


def projected_to_geographic(
    x: float, y: float, source_crs: Any | None
) -> Tuple[float | None, float | None]:
    if source_crs is None:
        return None, None

    if source_crs.is_geographic:
        lon, lat = x, y
        return lon, lat

    try:
        if Transformer is None:
            return None, None
        target = source_crs.geodetic_crs or CRS.from_epsg(4326)
        transformer = Transformer.from_crs(source_crs, target, always_xy=True)
        lon, lat = transformer.transform(x, y)
        return lon, lat
    except Exception:
        return None, None


def build_metadata(
    file_path: Path, produto: str = "intensidade", default_epsg: int = 31982
) -> OrderedDict[str, str]:
    las = laspy.read(file_path)
    header = las.header
    try:
        crs = header.parse_crs()
    except Exception:
        crs = None

    # Fallback para reduzir N/A quando o LAS nao expor CRS de forma direta.
    if crs is None and Transformer is not None:
        try:
            crs = CRS.from_epsg(default_epsg)
        except Exception:
            crs = None

    x_min, y_min, z_min = float(header.mins[0]), float(header.mins[1]), float(header.mins[2])
    x_max, y_max, z_max = float(header.maxs[0]), float(header.maxs[1]), float(header.maxs[2])

    width = x_max - x_min
    height = y_max - y_min
    bbox_area_m2 = max(width * height, 0.0)

    point_count = int(header.point_count)
    density = (point_count / bbox_area_m2) if bbox_area_m2 > 0 else 0.0
    spacing = (1.0 / np.sqrt(density)) if density > 0 else 0.0

    point_record_length = int(header.point_format.size)
    estimated_mb = (point_count * point_record_length) / (1024 * 1024)

    ul = (x_min, y_max)
    ur = (x_max, y_max)
    lr = (x_max, y_min)
    ll = (x_min, y_min)

    west_lon, _ = projected_to_geographic(x_min, y_min, crs)
    _, north_lat = projected_to_geographic(x_min, y_max, crs)
    east_lon, _ = projected_to_geographic(x_max, y_max, crs)
    _, south_lat = projected_to_geographic(x_max, y_min, crs)

    ul_lon, ul_lat = projected_to_geographic(*ul, source_crs=crs)
    ur_lon, ur_lat = projected_to_geographic(*ur, source_crs=crs)
    lr_lon, lr_lat = projected_to_geographic(*lr, source_crs=crs)
    ll_lon, ll_lat = projected_to_geographic(*ll, source_crs=crs)

    epsg = safe_epsg(crs)
    geodetic_crs = crs.geodetic_crs if crs else None
    geographic_epsg = safe_epsg(geodetic_crs)
    ellipsoid = geodetic_crs.ellipsoid if geodetic_crs else None

    min_intensity = int(np.min(las.intensity)) if "intensity" in las.point_format.dimension_names else 0
    max_intensity = int(np.max(las.intensity)) if "intensity" in las.point_format.dimension_names else 0

    metadata = OrderedDict()
    metadata["FILENAME"] = display_filename_path(file_path)
    metadata["DESCRIPTION"] = file_path.name
    metadata["AREA COUNT"] = "0"
    metadata["LINE COUNT"] = "0"
    metadata["POINT COUNT"] = "0"
    metadata["MESH COUNT"] = "0"
    metadata["LIDAR POINT COUNT"] = format_pt_br_int(point_count)
    metadata["POINT CLOUD MEMORY"] = f"{estimated_mb:.1f} MB (Estimated)"
    metadata["LIDAR POINT DENSITY"] = f"{density:.3f} samples / m^2"
    metadata["LIDAR POINT SPACING"] = f"{spacing:.4f} m"
    metadata["LIDAR OFFSET"] = (
        f"( {header.offsets[0]:.1f}, {header.offsets[1]:.1f}, {header.offsets[2]:.1f} )"
    )
    metadata["LIDAR SCALE"] = (
        f"( {header.scales[0]:.3f}, {header.scales[1]:.3f}, {header.scales[2]:.3f} )"
    )
    metadata["UPPER LEFT X"] = fmt_float(ul[0], 3)
    metadata["UPPER LEFT Y"] = fmt_float(ul[1], 3)
    metadata["LOWER RIGHT X"] = fmt_float(lr[0], 3)
    metadata["LOWER RIGHT Y"] = fmt_float(lr[1], 3)

    metadata["WEST LONGITUDE"] = format_dms(west_lon, is_lat=False) if west_lon is not None else "N/A"
    metadata["NORTH LATITUDE"] = format_dms(north_lat, is_lat=True) if north_lat is not None else "N/A"
    metadata["EAST LONGITUDE"] = format_dms(east_lon, is_lat=False) if east_lon is not None else "N/A"
    metadata["SOUTH LATITUDE"] = format_dms(south_lat, is_lat=True) if south_lat is not None else "N/A"

    metadata["UL CORNER LONGITUDE"] = format_dms(ul_lon, is_lat=False) if ul_lon is not None else "N/A"
    metadata["UL CORNER LATITUDE"] = format_dms(ul_lat, is_lat=True) if ul_lat is not None else "N/A"
    metadata["UR CORNER LONGITUDE"] = format_dms(ur_lon, is_lat=False) if ur_lon is not None else "N/A"
    metadata["UR CORNER LATITUDE"] = format_dms(ur_lat, is_lat=True) if ur_lat is not None else "N/A"
    metadata["LR CORNER LONGITUDE"] = format_dms(lr_lon, is_lat=False) if lr_lon is not None else "N/A"
    metadata["LR CORNER LATITUDE"] = format_dms(lr_lat, is_lat=True) if lr_lat is not None else "N/A"
    metadata["LL CORNER LONGITUDE"] = format_dms(ll_lon, is_lat=False) if ll_lon is not None else "N/A"
    metadata["LL CORNER LATITUDE"] = format_dms(ll_lat, is_lat=True) if ll_lat is not None else "N/A"

    if crs:
        metadata["PROJ_DESC"] = crs.name or "N/A"
        metadata["PROJ_DATUM"] = (geodetic_crs.datum.name if geodetic_crs and geodetic_crs.datum else "N/A")
        metadata["PROJ_UNITS"] = (
            crs.axis_info[0].unit_name if crs.axis_info and crs.axis_info[0].unit_name else "N/A"
        )
        metadata["EPSG_CODE"] = f"EPSG:{epsg}" if epsg else "N/A"
    else:
        metadata["PROJ_DESC"] = "N/A"
        metadata["PROJ_DATUM"] = "N/A"
        metadata["PROJ_UNITS"] = "N/A"
        metadata["EPSG_CODE"] = "N/A"

    metadata["BBOX AREA"] = f"{bbox_area_m2 / 1_000_000:.3f} sq km"

    stats = file_path.stat()
    metadata["FILE_CREATION_TIME"] = format_timestamp(stats.st_ctime)
    metadata["FILE_MODIFIED_TIME"] = format_timestamp(stats.st_mtime)

    units = str(metadata["PROJ_UNITS"]).lower()
    metadata["GeoTIFF::ProjLinearUnitsGeoKey"] = "9001" if units in {"metre", "meter", "meters"} else "N/A"
    metadata["GeoTIFF::ProjectedCSTypeGeoKey"] = str(epsg) if epsg else "N/A"
    metadata["GeoTIFF::GeographicTypeGeoKey"] = str(geographic_epsg) if geographic_epsg else "N/A"
    metadata["GeoTIFF::GeogSemiMajorAxisGeoKey"] = (
        f"{ellipsoid.semi_major_metre}" if ellipsoid else "N/A"
    )
    metadata["GeoTIFF::GeogSemiMinorAxisGeoKey"] = (
        f"{ellipsoid.semi_minor_metre}" if ellipsoid else "N/A"
    )
    metadata["GeoTIFF::GeogEllipsoidGeoKey"] = geotiff_ellipsoid_key(ellipsoid.name if ellipsoid else None)
    metadata["GeoTIFF::GeogToWGS84GeoKey"] = "{ 0.000, 0.000, 0.000, 0.0000000, 0.0000000, 0.0000000, 0.0000000 }"

    if produto == "hipsometrica_composta":
        metadata["MIN ELEVATION"] = f"{z_min:.2f} METERS"
        metadata["MAX ELEVATION"] = f"{z_max:.2f} METERS"
    else:
        metadata["MIN INTENSITY"] = str(min_intensity)
        metadata["MAX INTENSITY"] = str(max_intensity)
    metadata["LAS_VERSION"] = f"{header.version.major}.{header.version.minor}"
    metadata["GLOBAL_ENCODING"] = str(int(header.global_encoding.value))
    metadata["GEN_SOFTWARE"] = header.generating_software.strip() if header.generating_software else "N/A"
    return metadata


def metadata_to_text(metadata: Dict[str, str]) -> str:
    return "\n".join(f"{k}={v}" for k, v in metadata.items())


def iter_las_files(folder: Path, recursive: bool = False) -> Iterable[Path]:
    pattern = "**/*.las" if recursive else "*.las"
    yield from sorted(folder.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai metadados de arquivos LAS em uma pasta.")
    parser.add_argument("folder", type=Path, help="Pasta com arquivos .las")
    parser.add_argument(
        "--recursive", action="store_true", help="Procura .las recursivamente em subpastas"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Pasta para salvar .txt de metadados (padrão: mesma pasta do LAS)",
    )
    parser.add_argument(
        "--produto",
        choices=["intensidade", "hipsometrica_composta"],
        default="intensidade",
        help="Tipo de produto para os campos finais do metadado",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Pasta inválida: {folder}")

    las_files = list(iter_las_files(folder, recursive=args.recursive))
    if not las_files:
        print(f"Nenhum arquivo .las encontrado em: {folder}")
        return

    for las_file in las_files:
        metadata = build_metadata(las_file, produto=args.produto)
        txt = metadata_to_text(metadata)
        print(txt)
        print()

        out_dir = args.output_dir.resolve() if args.output_dir else las_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{las_file.stem}.txt"
        out_file.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
