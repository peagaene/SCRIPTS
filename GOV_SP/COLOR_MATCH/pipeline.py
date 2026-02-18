import argparse
import glob
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
try:
    from osgeo import gdal as _gdal
except ModuleNotFoundError:
    _gdal = None


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".jp2", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class RasterData:
    array: np.ndarray  # (bands, height, width)
    nodata_values: List[float]
    geotransform: Tuple[float, ...]
    projection: str
    gdal_dtype: int
    metadata: Dict[str, str]


def _require_gdal():
    if _gdal is None:
        raise ModuleNotFoundError(
            "Pacote 'osgeo' (GDAL) nao encontrado. "
            "Instale GDAL no ambiente para ler/escrever GeoTIFF com georreferencia."
        )
    return _gdal


def _suppress_runtime_warnings() -> None:
    # Mantem o terminal limpo durante processamentos longos.
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")


def _normalize_channel_order(order: str) -> str:
    key = order.strip().upper()
    aliases = {
        "RGBIR": "RGBI",
        "CIR": "IRG",
        "IRRG": "IRG",
        "NIRRG": "IRG",
    }
    return aliases.get(key, key)


def parse_channel_order(order: str) -> Dict[str, int]:
    normalized = _normalize_channel_order(order)
    valid = {"R", "G", "B", "I"}
    channel_map: Dict[str, int] = {}
    for index, channel in enumerate(normalized):
        if channel not in valid:
            raise ValueError(f"Canal invalido '{channel}' em channel_order='{order}'.")
        if channel in channel_map:
            raise ValueError(f"Canal repetido '{channel}' em channel_order='{order}'.")
        channel_map[channel] = index
    return channel_map


def _expand_path_entry(entry: str) -> List[Path]:
    if any(ch in entry for ch in "*?[]"):
        expanded: List[Path] = []
        for match in sorted(glob.glob(entry, recursive=True)):
            path = Path(match)
            if path.is_file():
                expanded.append(path)
            elif path.is_dir():
                for ext in SUPPORTED_EXTENSIONS:
                    expanded.extend(path.rglob(f"*{ext}"))
        return sorted(set(expanded))

    path = Path(entry)
    if path.is_file():
        return [path]
    if path.is_dir():
        files: List[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(path.rglob(f"*{ext}"))
        return sorted(files)
    return []


def resolve_input_files(entries: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    seen = set()
    for entry in entries:
        for path in _expand_path_entry(entry):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    files.append(path)
    return files


def read_raster(path: Path) -> RasterData:
    gdal = _require_gdal()
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Nao foi possivel abrir: {path}")

    array = dataset.ReadAsArray()
    if array is None:
        raise RuntimeError(f"Nao foi possivel ler bandas de: {path}")
    if array.ndim == 2:
        array = array[np.newaxis, ...]

    nodata_values: List[float] = []
    for band_index in range(dataset.RasterCount):
        band = dataset.GetRasterBand(band_index + 1)
        nodata_values.append(band.GetNoDataValue())

    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    gdal_dtype = dataset.GetRasterBand(1).DataType
    metadata = dataset.GetMetadata()

    dataset = None
    return RasterData(
        array=array,
        nodata_values=nodata_values,
        geotransform=geotransform,
        projection=projection,
        gdal_dtype=gdal_dtype,
        metadata=metadata,
    )


def _is_valid_nodata(value: float) -> bool:
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def _valid_mask(array: np.ndarray, nodata_values: Sequence[float], band_indices: Iterable[int]) -> np.ndarray:
    mask = np.ones(array.shape[1:], dtype=bool)
    for idx in band_indices:
        band = array[idx]
        mask &= np.isfinite(band)
        nodata = nodata_values[idx]
        if _is_valid_nodata(nodata):
            mask &= band != nodata
    return mask


def sample_channels(
    files: Sequence[Path],
    channel_map: Dict[str, int],
    max_pixels_per_image: int,
    rng: np.random.Generator,
    downsample_max_side: int = 1024,
    progress_label: str = "",
) -> Dict[str, np.ndarray]:
    samples: Dict[str, List[np.ndarray]] = {ch: [] for ch in channel_map.keys()}
    gdal = _require_gdal()

    total = len(files)
    for i, path in enumerate(files, start=1):
        if progress_label and (i == 1 or i == total or i % 10 == 0):
            print(f"[sample:{progress_label}] {i}/{total}")
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if ds is None:
            continue
        bands = ds.RasterCount
        for channel, idx in channel_map.items():
            if idx >= bands:
                ds = None
                raise ValueError(
                    f"Arquivo '{path}' tem {bands} bandas, mas canal '{channel}' "
                    f"foi mapeado para indice {idx}."
                )

        x_size = ds.RasterXSize
        y_size = ds.RasterYSize
        buf_x = min(max(1, downsample_max_side), x_size)
        buf_y = min(max(1, downsample_max_side), y_size)

        used_indices = list(channel_map.values())
        band_arrays: Dict[int, np.ndarray] = {}
        nodata_values: Dict[int, float] = {}
        for idx in used_indices:
            band = ds.GetRasterBand(idx + 1)
            nodata_values[idx] = band.GetNoDataValue()
            arr = band.ReadAsArray(0, 0, x_size, y_size, buf_x, buf_y)
            if arr is None:
                arr = np.array([], dtype=np.float32).reshape(0, 0)
            band_arrays[idx] = arr
        ds = None

        if not band_arrays:
            continue

        base_shape = next(iter(band_arrays.values())).shape
        if len(base_shape) != 2 or base_shape[0] == 0 or base_shape[1] == 0:
            continue

        stack = np.zeros((max(used_indices) + 1, base_shape[0], base_shape[1]), dtype=np.float32)
        nodata = [None] * (max(used_indices) + 1)
        for idx, arr in band_arrays.items():
            stack[idx] = arr
            nodata[idx] = nodata_values[idx]

        mask = _valid_mask(stack, nodata, used_indices)
        flat_valid = np.flatnonzero(mask.ravel())
        if flat_valid.size == 0:
            continue

        if max_pixels_per_image > 0 and flat_valid.size > max_pixels_per_image:
            flat_valid = rng.choice(flat_valid, size=max_pixels_per_image, replace=False)

        flat = stack.reshape(stack.shape[0], -1)
        for channel, idx in channel_map.items():
            values = flat[idx, flat_valid].astype(np.float64, copy=False)
            if values.size > 0:
                samples[channel].append(values)

    merged: Dict[str, np.ndarray] = {}
    for channel, chunks in samples.items():
        merged[channel] = np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
    return merged


def _downsampled_band_mean(path: Path, band_idx: int, max_side: int = 256) -> float:
    gdal = _require_gdal()
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        return np.nan

    if band_idx >= ds.RasterCount:
        ds = None
        return np.nan

    rb = ds.GetRasterBand(band_idx + 1)
    nodata = rb.GetNoDataValue()
    x_size = ds.RasterXSize
    y_size = ds.RasterYSize
    buf_x = min(max(1, max_side), x_size)
    buf_y = min(max(1, max_side), y_size)
    band = rb.ReadAsArray(0, 0, x_size, y_size, buf_x, buf_y)
    ds = None
    if band is None:
        return np.nan

    band = band.astype(np.float64, copy=False)
    mask = np.isfinite(band)
    if _is_valid_nodata(nodata):
        mask &= band != nodata
    if not np.any(mask):
        return np.nan

    # Decima para reduzir custo no filtro de outlier.
    h, w = band.shape
    step_y = max(1, h // max_side)
    step_x = max(1, w // max_side)
    b = band[::step_y, ::step_x]
    m = mask[::step_y, ::step_x]
    if not np.any(m):
        return np.nan
    return float(np.mean(b[m]))


def filter_calibration_tiles(
    files: Sequence[Path],
    channel_map: Dict[str, int],
    filter_cfg: Dict,
) -> List[Path]:
    if not files:
        return []

    enabled = bool(filter_cfg.get("enabled", False))
    if not enabled:
        return list(files)

    channels = filter_cfg.get("channels", list(channel_map.keys()))
    z_thresh = float(filter_cfg.get("zscore_threshold", 3.5))
    max_side = int(filter_cfg.get("downsample_max_side", 256))

    valid_channels = [ch for ch in channels if ch in channel_map]
    if not valid_channels:
        return list(files)

    rows: List[Tuple[Path, np.ndarray]] = []
    total = len(files)
    for i, path in enumerate(files, start=1):
        if i == 1 or i == total or i % 20 == 0:
            print(f"[outlier-filter] analisando {i}/{total}")
        means = []
        for ch in valid_channels:
            means.append(_downsampled_band_mean(path, channel_map[ch], max_side=max_side))
        rows.append((path, np.array(means, dtype=np.float64)))

    mat = np.array([r[1] for r in rows], dtype=np.float64)
    mu = np.nanmean(mat, axis=0)
    sigma = np.nanstd(mat, axis=0)

    kept: List[Path] = []
    removed: List[Tuple[float, Path]] = []

    for path, vec in rows:
        z = np.where(sigma > 0, np.abs((vec - mu) / sigma), 0.0)
        score = float(np.nansum(z))
        if np.isnan(score) or score > z_thresh * len(valid_channels):
            removed.append((score, path))
        else:
            kept.append(path)

    min_keep = int(filter_cfg.get("min_keep", 3))
    if len(kept) < min_keep:
        print(
            f"[outlier-filter] Poucos tiles restantes ({len(kept)}). "
            f"Mantendo todos os {len(files)} tiles de calibracao."
        )
        return list(files)

    print(
        f"[outlier-filter] Tiles calibracao: total={len(files)}, "
        f"mantidos={len(kept)}, removidos={len(removed)}"
    )
    if removed:
        removed_sorted = sorted(removed, key=lambda x: (np.nan_to_num(x[0], nan=1e9)), reverse=True)
        for score, p in removed_sorted[:10]:
            print(f"[outlier-filter] removido score={score:.3f} file={p}")
    return kept


def _compute_affine_from_percentiles(
    src_values: np.ndarray,
    ref_values: np.ndarray,
    p_low: float,
    p_high: float,
) -> Dict[str, float]:
    eps = 1e-9
    src_low, src_high = np.percentile(src_values, [p_low, p_high])
    ref_low, ref_high = np.percentile(ref_values, [p_low, p_high])
    gain = float((ref_high - ref_low) / (src_high - src_low + eps))
    offset = float(ref_low - gain * src_low)
    return {
        "gain": gain,
        "offset": offset,
        "src_p_low": float(src_low),
        "src_p_high": float(src_high),
        "ref_p_low": float(ref_low),
        "ref_p_high": float(ref_high),
    }


def _clamp_gain_offset(
    channel: str,
    gain: float,
    offset: float,
    constraints_cfg: Dict,
    source_percentile_low: float,
    source_percentile_high: float,
) -> Tuple[float, float]:
    gain_min = float(constraints_cfg.get("gain_min", -np.inf))
    gain_max = float(constraints_cfg.get("gain_max", np.inf))

    per_channel = constraints_cfg.get("per_channel", {})
    ch_cfg = per_channel.get(channel, {})
    gain_min = float(ch_cfg.get("gain_min", gain_min))
    gain_max = float(ch_cfg.get("gain_max", gain_max))

    clamped_gain = float(np.clip(gain, gain_min, gain_max))
    if clamped_gain == gain:
        return gain, offset

    # Recalibra offset para manter o ponto medio do intervalo de ajuste.
    src_mid = 0.5 * (source_percentile_low + source_percentile_high)
    ref_mid = gain * src_mid + offset
    new_offset = ref_mid - clamped_gain * src_mid
    return clamped_gain, float(new_offset)


def _resolve_base_params(refinement_cfg: Dict) -> Dict:
    base_params = refinement_cfg.get("base_params")
    if isinstance(base_params, dict):
        return base_params

    base_params_file = refinement_cfg.get("base_params_file")
    if isinstance(base_params_file, str) and base_params_file.strip():
        return load_json(Path(base_params_file))

    return {}


def _apply_refinement_with_base_params(channels_result: Dict[str, Dict[str, float]], refinement_cfg: Dict) -> None:
    if not bool(refinement_cfg.get("enabled", False)):
        return

    base_params = _resolve_base_params(refinement_cfg)
    base_channels = base_params.get("channels", {}) if isinstance(base_params, dict) else {}
    if not isinstance(base_channels, dict) or not base_channels:
        return

    blend = float(refinement_cfg.get("blend", 1.0))
    blend = float(np.clip(blend, 0.0, 1.0))

    max_delta_gain_default = float(refinement_cfg.get("max_delta_gain", np.inf))
    max_delta_offset_default = float(refinement_cfg.get("max_delta_offset", np.inf))
    per_channel = refinement_cfg.get("per_channel", {})

    for channel, ch in channels_result.items():
        if not ch.get("enabled", False):
            continue
        base_ch = base_channels.get(channel)
        if not isinstance(base_ch, dict) or not base_ch.get("enabled", False):
            continue

        base_gain = float(base_ch.get("gain", 1.0))
        base_offset = float(base_ch.get("offset", 0.0))
        curr_gain = float(ch.get("gain", 1.0))
        curr_offset = float(ch.get("offset", 0.0))

        ch_cfg = per_channel.get(channel, {}) if isinstance(per_channel, dict) else {}
        max_delta_gain = float(ch_cfg.get("max_delta_gain", max_delta_gain_default))
        max_delta_offset = float(ch_cfg.get("max_delta_offset", max_delta_offset_default))

        delta_gain = curr_gain - base_gain
        delta_offset = curr_offset - base_offset

        if np.isfinite(max_delta_gain):
            delta_gain = float(np.clip(delta_gain, -max_delta_gain, max_delta_gain))
        if np.isfinite(max_delta_offset):
            delta_offset = float(np.clip(delta_offset, -max_delta_offset, max_delta_offset))

        bounded_gain = base_gain + delta_gain
        bounded_offset = base_offset + delta_offset

        final_gain = base_gain + blend * (bounded_gain - base_gain)
        final_offset = base_offset + blend * (bounded_offset - base_offset)

        ch["gain_raw"] = curr_gain
        ch["offset_raw"] = curr_offset
        ch["base_gain"] = base_gain
        ch["base_offset"] = base_offset
        ch["gain"] = float(final_gain)
        ch["offset"] = float(final_offset)
        ch["refinement_applied"] = True


def fit_transformation(config: Dict) -> Dict:
    source_cfg = config["source"]
    ref_cfg = config["reference"]
    sampling_cfg = config.get("sampling", {})

    max_pixels = int(sampling_cfg.get("max_pixels_per_image", 200000))
    downsample_max_side = int(sampling_cfg.get("downsample_max_side", 1024))
    p_low = float(sampling_cfg.get("percentile_low", 2.0))
    p_high = float(sampling_cfg.get("percentile_high", 98.0))
    seed = int(sampling_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    filter_cfg = config.get("calibration_filter", {})
    constraints_cfg = config.get("constraints", {})
    ir_tuning_cfg = config.get("ir_tuning", {})
    refinement_cfg = config.get("refinement", {})

    source_order = parse_channel_order(source_cfg.get("channel_order", "RGBI"))
    rgb_order = parse_channel_order(ref_cfg.get("rgb_channel_order", "RGB"))
    cir_order = parse_channel_order(ref_cfg.get("cir_channel_order", "IRRG"))

    calibration_tiles = resolve_input_files(source_cfg["calibration_tiles"])
    if not calibration_tiles:
        raise ValueError("Nenhum tile de calibracao foi encontrado.")
    calibration_tiles = filter_calibration_tiles(calibration_tiles, source_order, filter_cfg)

    rgb_refs = resolve_input_files(ref_cfg.get("rgb_images", []))
    cir_refs = resolve_input_files(ref_cfg.get("cir_images", []))
    if not rgb_refs and not cir_refs:
        raise ValueError("Nenhuma referencia RGB/CIR foi encontrada.")

    source_channels = {ch: idx for ch, idx in source_order.items() if ch in {"R", "G", "B", "I"}}
    source_samples = sample_channels(
        calibration_tiles,
        source_channels,
        max_pixels,
        rng,
        downsample_max_side=downsample_max_side,
        progress_label="source",
    )

    ref_values: Dict[str, List[np.ndarray]] = {"R": [], "G": [], "B": [], "I": []}
    if rgb_refs:
        rgb_channels = {ch: rgb_order[ch] for ch in ("R", "G", "B") if ch in rgb_order}
        rgb_samples = sample_channels(
            rgb_refs,
            rgb_channels,
            max_pixels,
            rng,
            downsample_max_side=downsample_max_side,
            progress_label="ref_rgb",
        )
        for ch, vals in rgb_samples.items():
            if vals.size > 0:
                ref_values[ch].append(vals)

    # CIR/IRRG entra apenas para o canal IR (I).
    # R/G/B devem vir exclusivamente das referencias RGB.
    if cir_refs:
        cir_channels = {ch: cir_order[ch] for ch in ("I",) if ch in cir_order}
        cir_samples = sample_channels(
            cir_refs,
            cir_channels,
            max_pixels,
            rng,
            downsample_max_side=downsample_max_side,
            progress_label="ref_ir",
        )
        for ch, vals in cir_samples.items():
            if vals.size > 0:
                ref_values[ch].append(vals)

    merged_ref: Dict[str, np.ndarray] = {}
    for channel, chunks in ref_values.items():
        merged_ref[channel] = np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)

    channels_result: Dict[str, Dict[str, float]] = {}
    for channel in ("R", "G", "B", "I"):
        src = source_samples.get(channel, np.array([], dtype=np.float64))
        ref = merged_ref.get(channel, np.array([], dtype=np.float64))

        if src.size == 0:
            channels_result[channel] = {
                "enabled": False,
                "reason": "Sem amostras na area para esse canal.",
                "gain": 1.0,
                "offset": 0.0,
            }
            continue

        if ref.size == 0:
            channels_result[channel] = {
                "enabled": False,
                "reason": "Sem amostras de referencia para esse canal.",
                "gain": 1.0,
                "offset": 0.0,
            }
            continue

        affine = _compute_affine_from_percentiles(src, ref, p_low, p_high)
        gain = affine["gain"]
        offset = affine["offset"]

        gain, offset = _clamp_gain_offset(
            channel,
            gain,
            offset,
            constraints_cfg,
            affine["src_p_low"],
            affine["src_p_high"],
        )

        if channel == "I":
            gain_scale = float(ir_tuning_cfg.get("gain_scale", 1.0))
            offset_shift = float(ir_tuning_cfg.get("offset_shift", 0.0))
            gain = gain * gain_scale
            offset = offset + offset_shift

        channels_result[channel] = {
            "enabled": True,
            "reason": "ok",
            "gain": gain,
            "offset": offset,
            "source_samples": int(src.size),
            "reference_samples": int(ref.size),
            "source_percentile_low": affine["src_p_low"],
            "source_percentile_high": affine["src_p_high"],
            "reference_percentile_low": affine["ref_p_low"],
            "reference_percentile_high": affine["ref_p_high"],
        }

    _apply_refinement_with_base_params(channels_result, refinement_cfg)

    return {
        "model": "per_channel_affine_percentile",
        "percentiles": {"low": p_low, "high": p_high},
        "source_channel_order": source_cfg.get("channel_order", "RGBI"),
        "calibration_tiles_used": len(calibration_tiles),
        "refinement": {
            "enabled": bool(refinement_cfg.get("enabled", False)),
            "base_params_file": refinement_cfg.get("base_params_file"),
            "blend": float(refinement_cfg.get("blend", 1.0)),
        },
        "channels": channels_result,
    }


def _dtype_limits(dtype: np.dtype) -> Tuple[float, float]:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        return float(info.min), float(info.max)
    return 0.0, 255.0


def _smoothstep(values: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    if edge1 <= edge0:
        return (values >= edge0).astype(np.float64)
    t = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _vegetation_weight(
    arr: np.ndarray,
    source_order: Dict[str, int],
    nodata_values: Sequence[float],
    vegetation_cfg: Dict,
) -> np.ndarray:
    eps = 1e-9
    mask_cfg = vegetation_cfg.get("mask", {})

    r_idx = source_order.get("R")
    g_idx = source_order.get("G")
    b_idx = source_order.get("B")
    if r_idx is None or g_idx is None or b_idx is None:
        return np.zeros(arr.shape[1:], dtype=np.float64)

    r = arr[r_idx]
    g = arr[g_idx]
    b = arr[b_idx]

    valid = np.isfinite(r) & np.isfinite(g) & np.isfinite(b)
    for idx in (r_idx, g_idx, b_idx):
        nodata = nodata_values[idx] if idx < len(nodata_values) else None
        if _is_valid_nodata(nodata):
            valid &= arr[idx] != nodata

    exg = 2.0 * g - r - b
    rg_ratio = g / (r + eps)
    gb_ratio = g / (b + eps)

    exg_threshold = float(mask_cfg.get("exg_threshold", 20.0))
    exg_softness = float(mask_cfg.get("exg_softness", 15.0))
    rg_threshold = float(mask_cfg.get("rg_ratio_threshold", 1.05))
    rg_softness = float(mask_cfg.get("rg_ratio_softness", 0.10))
    gb_threshold = float(mask_cfg.get("gb_ratio_threshold", 1.05))
    gb_softness = float(mask_cfg.get("gb_ratio_softness", 0.10))

    w_exg = _smoothstep(exg, exg_threshold, exg_threshold + exg_softness)
    w_rg = _smoothstep(rg_ratio, rg_threshold, rg_threshold + rg_softness)
    w_gb = _smoothstep(gb_ratio, gb_threshold, gb_threshold + gb_softness)
    weight = w_exg * w_rg * w_gb

    use_ndvi = bool(mask_cfg.get("use_ndvi", True))
    i_idx = source_order.get("I")
    if use_ndvi and i_idx is not None and i_idx < arr.shape[0]:
        i = arr[i_idx]
        ndvi = (i - r) / (i + r + eps)
        ndvi_threshold = float(mask_cfg.get("ndvi_threshold", 0.05))
        ndvi_softness = float(mask_cfg.get("ndvi_softness", 0.05))
        w_ndvi = _smoothstep(ndvi, ndvi_threshold, ndvi_threshold + ndvi_softness)
        nodata_i = nodata_values[i_idx] if i_idx < len(nodata_values) else None
        valid_i = np.isfinite(i)
        if _is_valid_nodata(nodata_i):
            valid_i &= i != nodata_i
        weight *= w_ndvi
        valid &= valid_i

    strength = float(vegetation_cfg.get("strength", 1.0))
    weight = np.clip(weight * strength, 0.0, 1.0)
    weight[~valid] = 0.0
    return weight


def _apply_selective_vegetation_tuning(
    arr: np.ndarray,
    source_order: Dict[str, int],
    nodata_values: Sequence[float],
    vegetation_cfg: Dict,
) -> None:
    if not bool(vegetation_cfg.get("enabled", False)):
        return

    weight = _vegetation_weight(arr, source_order, nodata_values, vegetation_cfg)
    if not np.any(weight > 0):
        return

    adjust_cfg = vegetation_cfg.get("adjust", {})
    channel_adjustments = {
        "R": (
            float(adjust_cfg.get("red_gain", 1.0)),
            float(adjust_cfg.get("red_offset", 0.0)),
        ),
        "G": (
            float(adjust_cfg.get("green_gain", 1.06)),
            float(adjust_cfg.get("green_offset", 0.0)),
        ),
        "B": (
            float(adjust_cfg.get("blue_gain", 1.0)),
            float(adjust_cfg.get("blue_offset", 0.0)),
        ),
        "I": (
            float(adjust_cfg.get("ir_gain", 1.0)),
            float(adjust_cfg.get("ir_offset", 0.0)),
        ),
    }

    for channel, (gain, offset) in channel_adjustments.items():
        idx = source_order.get(channel)
        if idx is None or idx >= arr.shape[0]:
            continue
        band = arr[idx]
        band[:] = band * (1.0 + weight * (gain - 1.0)) + weight * offset
        arr[idx] = band


def _build_output_path(tile: Path, output_dir: Path, common_root: Path, suffix: str) -> Path:
    rel = tile.name
    if common_root is not None:
        try:
            rel = str(tile.resolve().relative_to(common_root.resolve()))
        except ValueError:
            rel = tile.name
    output = output_dir / rel
    if suffix:
        output = output.with_name(output.stem + suffix + output.suffix)
    return output


def apply_transformation(config: Dict, params: Dict) -> None:
    source_cfg = config["source"]
    output_cfg = config.get("output", {})
    vegetation_cfg = config.get("vegetation_tuning", {})

    apply_tiles = resolve_input_files(source_cfg.get("apply_tiles", source_cfg["calibration_tiles"]))
    if not apply_tiles:
        raise ValueError("Nenhum tile para aplicacao foi encontrado.")

    source_order = parse_channel_order(source_cfg.get("channel_order", "RGBI"))
    output_dir = Path(output_cfg["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    overwrite = bool(output_cfg.get("overwrite", False))
    suffix = str(output_cfg.get("suffix", "_matched"))

    common_root = None
    try:
        common_root = Path(os.path.commonpath([str(p.resolve()) for p in apply_tiles]))
    except Exception:
        common_root = None

    for idx, tile in enumerate(apply_tiles, start=1):
        out_path = _build_output_path(tile, output_dir, common_root, suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            print(f"[{idx}/{len(apply_tiles)}] Ja existe, pulando: {out_path}")
            continue

        raster = read_raster(tile)
        arr = raster.array.astype(np.float64, copy=True)
        out_dtype = raster.array.dtype
        dtype_min, dtype_max = _dtype_limits(out_dtype)

        for channel, band_idx in source_order.items():
            if channel not in params["channels"]:
                continue
            channel_params = params["channels"][channel]
            if not channel_params.get("enabled", False):
                continue
            if band_idx >= arr.shape[0]:
                continue

            nodata = raster.nodata_values[band_idx]
            band = arr[band_idx]
            valid = np.isfinite(band)
            if _is_valid_nodata(nodata):
                valid &= band != nodata

            gain = float(channel_params["gain"])
            offset = float(channel_params["offset"])
            band[valid] = band[valid] * gain + offset
            if np.issubdtype(out_dtype, np.integer):
                band[valid] = np.clip(band[valid], dtype_min, dtype_max)
            arr[band_idx] = band

        _apply_selective_vegetation_tuning(
            arr,
            source_order,
            raster.nodata_values,
            vegetation_cfg,
        )
        if np.issubdtype(out_dtype, np.integer):
            arr = np.clip(arr, dtype_min, dtype_max)

        if np.issubdtype(out_dtype, np.integer):
            arr = np.rint(arr).astype(out_dtype, copy=False)
        else:
            arr = arr.astype(out_dtype, copy=False)

        write_raster(out_path, arr, raster)
        print(f"[{idx}/{len(apply_tiles)}] Salvo: {out_path}")


def write_raster(path: Path, array: np.ndarray, template: RasterData) -> None:
    gdal = _require_gdal()
    bands, height, width = array.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), width, height, bands, template.gdal_dtype)
    if dataset is None:
        raise RuntimeError(f"Nao foi possivel criar: {path}")

    if template.geotransform:
        dataset.SetGeoTransform(template.geotransform)
    if template.projection:
        dataset.SetProjection(template.projection)
    if template.metadata:
        dataset.SetMetadata(template.metadata)

    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(array[i])
        nodata = template.nodata_values[i] if i < len(template.nodata_values) else None
        if _is_valid_nodata(nodata):
            band.SetNoDataValue(float(nodata))

    dataset.FlushCache()
    dataset = None


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Color match por area para tiles RGBIR usando referencias RGB e/ou IRRG (CIR)."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit_cmd = sub.add_parser("fit", help="Calcula parametros de transformacao da area.")
    fit_cmd.add_argument("--config", required=True, help="Caminho do JSON de configuracao.")
    fit_cmd.add_argument("--params-out", required=True, help="JSON de saida com parametros.")

    apply_cmd = sub.add_parser("apply", help="Aplica parametros em lote nos tiles da area.")
    apply_cmd.add_argument("--config", required=True, help="Caminho do JSON de configuracao.")
    apply_cmd.add_argument("--params-in", required=True, help="JSON de entrada com parametros.")

    run_cmd = sub.add_parser("run", help="Calcula parametros e aplica em lote.")
    run_cmd.add_argument("--config", required=True, help="Caminho do JSON de configuracao.")
    run_cmd.add_argument("--params-out", required=True, help="JSON de saida com parametros.")

    return parser


def main(argv: Sequence[str] = None) -> int:
    _suppress_runtime_warnings()
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_json(Path(args.config))

    if args.command == "fit":
        params = fit_transformation(config)
        save_json(Path(args.params_out), params)
        print(f"Parametros salvos em: {args.params_out}")
        return 0

    if args.command == "apply":
        params = load_json(Path(args.params_in))
        apply_transformation(config, params)
        return 0

    if args.command == "run":
        params = fit_transformation(config)
        save_json(Path(args.params_out), params)
        print(f"Parametros salvos em: {args.params_out}")
        apply_transformation(config, params)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
