"""
Contours processing.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, List

import ezdxf
from shapely.geometry import LineString, Polygon
from shapely.ops import substring
from osgeo import gdal, ogr


DEF_LYR_CURVA_I = "HM_CURVA_NIV_INTERMEDIARIA"
DEF_LYR_CURVA_M = "HM_CURVA_NIV_MESTRA"
DEF_LYR_CURVA_TXT = "TOP_CURVA_NIV"


def _ensure(doc, layer: str):
    try:
        if layer not in doc.layers:
            doc.layers.add(layer)
    except Exception:
        pass


def _ensure_text_style(doc: ezdxf.EzDxf, style_name: str):
    try:
        if style_name not in doc.styles:
            doc.styles.new(style_name, dxfattribs={})
    except Exception:
        pass


def _as_gdal_dataset(mdt_src):
    if isinstance(mdt_src, gdal.Dataset):
        return mdt_src
    try:
        return gdal.Open(mdt_src)
    except Exception:
        return mdt_src


def _ogr_lines_to_shapely(geom) -> List[LineString]:
    out: List[LineString] = []
    if geom is None:
        return out
    t = geom.GetGeometryType()
    if t == ogr.wkbLineString or t == ogr.wkbLineString25D:
        pts = [(geom.GetPoint(i)[0], geom.GetPoint(i)[1]) for i in range(geom.GetPointCount())]
        if len(pts) >= 2:
            out.append(LineString(pts))
    elif t == ogr.wkbMultiLineString:
        for i in range(geom.GetGeometryCount()):
            out.extend(_ogr_lines_to_shapely(geom.GetGeometryRef(i)))
    return out


def _clip_lines_to_poly(lines: List[LineString], poly: Optional[Polygon]) -> List[LineString]:
    if poly is None:
        return [ls for ls in lines if not ls.is_empty]
    out: List[LineString] = []
    for ls in lines:
        if ls.is_empty:
            continue
        inter = ls.intersection(poly)
        if inter.is_empty:
            continue
        if inter.geom_type == "LineString":
            out.append(inter)  # type: ignore
        elif inter.geom_type == "MultiLineString":
            out.extend([g for g in inter.geoms if g.length > 0])
    return out


def _tangent_angle_deg(ls: LineString, dist: float) -> float:
    dist = max(0.0, min(float(dist), float(ls.length)))
    a = max(0.0, dist - 0.01)
    b = min(ls.length, dist + 0.01)
    pa = ls.interpolate(a)
    pb = ls.interpolate(b)
    dx, dy = (pb.x - pa.x), (pb.y - pa.y)
    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
        return 0.0
    ang = math.degrees(math.atan2(dy, dx))
    while ang <= -180:
        ang += 360
    while ang > 180:
        ang -= 360
    if ang > 90:
        ang -= 180
    if ang <= -90:
        ang += 180
    return ang


def _calculate_text_position_outside(ls: LineString, dist: float, offset_m: float = 2.0) -> Tuple[float, float, float]:
    pt = ls.interpolate(dist)
    angle_deg = _tangent_angle_deg(ls, dist)
    angle_rad = math.radians(angle_deg)
    perp_x = -math.sin(angle_rad)
    perp_y = math.cos(angle_rad)
    final_x = pt.x + perp_x * offset_m
    final_y = pt.y + perp_y * offset_m
    return final_x, final_y, angle_deg


def _format_elev(v: float, prec: int = 1) -> str:
    s = f"{v:.{prec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _text_len_m(content: str, h: float, char_w_factor: float) -> float:
    return max(0.01, len(content) * h * float(char_w_factor))


def _cut_line_by_gaps(ls: LineString, gaps: List[Tuple[float, float]]) -> List[LineString]:
    if not gaps:
        return [ls]
    L = float(ls.length)
    norm = []
    for s0, s1 in gaps:
        a, b = sorted((max(0.0, s0), min(L, s1)))
        if b - a > 1e-6:
            norm.append((a, b))
    if not norm:
        return [ls]
    norm.sort()
    merged = []
    ca, cb = norm[0]
    for a, b in norm[1:]:
        if a <= cb + 1e-6:
            cb = max(cb, b)
        else:
            merged.append((ca, cb))
            ca, cb = a, b
    merged.append((ca, cb))
    segs = []
    start = 0.0
    for a, b in merged:
        if a - start > 1e-6:
            segs.append(substring(ls, start, a))
        start = b
    if L - start > 1e-6:
        segs.append(substring(ls, start, L))
    out: List[LineString] = []
    for s in segs:
        if s.is_empty:
            continue
        if s.geom_type == "LineString":
            out.append(s)  # type: ignore
        elif s.geom_type == "MultiLineString":
            out.extend([g for g in s.geoms if g.length > 0])
    return out


def gerar_curvas_nivel(ms, mdt_src, params, clip_poly: Optional[Polygon] = None):
    eq = float(getattr(params, "curva_equidist", 1.0))
    mestra_cada = int(getattr(params, "curva_mestra_cada", 5))
    min_len = float(getattr(params, "curva_min_len", 10.0))
    h_text = float(getattr(params, "altura_texto_curva", 0.4))
    char_w_factor = float(getattr(params, "curva_char_w_factor", 0.6))
    gap_margin = float(getattr(params, "curva_gap_margin", 0.5))
    step_m = float(getattr(params, "curva_label_step_m", 80.0))
    ends_only = bool(getattr(params, "curva_label_ends_only", False))
    only_master_labels = bool(getattr(params, "curva_label_only_master", True))
    label_precision = int(getattr(params, "curva_label_precision", 1))
    label_offset_m = max(0.0, float(getattr(params, "curva_label_offset_m", 0.25)))
    label_gap_enabled = bool(getattr(params, "curva_label_gap_enabled", False))

    style_name = getattr(params, "style_texto", "SIMPLEX") or "SIMPLEX"
    layer_curva_i = getattr(params, "layer_curvas", DEF_LYR_CURVA_I)
    layer_curva_m = getattr(params, "layer_curvas_mestra", DEF_LYR_CURVA_M)
    layer_txt = getattr(params, "layer_curvas_txt", DEF_LYR_CURVA_TXT)

    doc = ms.doc
    _ensure_text_style(doc, style_name)
    for lyr in (layer_curva_i, layer_curva_m, layer_txt):
        _ensure(doc, lyr)

    ds = _as_gdal_dataset(mdt_src)
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    use_nodata = 1 if nodata is not None else 0
    no_val = float(nodata) if nodata is not None else 0.0

    drv = ogr.GetDriverByName("Memory")
    memds = drv.CreateDataSource("mem_contours")
    lyr = memds.CreateLayer("contours", srs=None, geom_type=ogr.wkbLineString)
    lyr.CreateField(ogr.FieldDefn("ELEV", ogr.OFTReal))
    elev_field_index = 0

    gdal.ContourGenerate(band, eq, 0.0, [], use_nodata, no_val, lyr, -1, elev_field_index)

    def _draw_polyline(coords, layer, elevation: float | None = None):
        try:
            pl = ms.add_lwpolyline(coords, dxfattribs={"layer": layer, "closed": False})
        except Exception:
            pl = ms.add_polyline2d(coords, dxfattribs={"layer": layer})
        try:
            if elevation is not None:
                pl.dxf.elevation = float(elevation)
        except Exception:
            pass
        return pl

    def _add_text(content: str, x: float, y: float, angle_deg: float):
        try:
            t = ms.add_text(content, dxfattribs={"height": h_text, "style": style_name, "layer": layer_txt})
            t.dxf.rotation = float(angle_deg)
            t.dxf.insert = (x, y)
            t.dxf.align_point = (x, y)
            t.dxf.halign = 1
            t.dxf.valign = 2
        except Exception:
            pass

    feat = lyr.GetNextFeature()
    while feat:
        elev = float(feat.GetFieldAsDouble(elev_field_index))
        geom = feat.GetGeometryRef()
        lines = _ogr_lines_to_shapely(geom)
        if clip_poly is not None:
            lines = _clip_lines_to_poly(lines, clip_poly)
        idx = int(round(elev / eq))
        is_master = (idx % mestra_cada == 0)

        for ls in lines:
            if ls.length < min_len:
                continue

            label_positions: List[float] = []
            if is_master or not only_master_labels:
                if ends_only:
                    if ls.length <= 0.01:
                        label_positions = [0.0]
                    else:
                        label_positions = [0.0, float(ls.length)]
                else:
                    nlab = max(1, int(ls.length // max(step_m, 1.0)))
                    for i in range(nlab):
                        s = (i + 0.5) * (ls.length / nlab)
                        label_positions.append(s)

            gaps: List[Tuple[float, float]] = []
            if not ends_only and label_gap_enabled:
                for s in label_positions:
                    txt = _format_elev(elev, label_precision)
                    half = 0.5 * _text_len_m(txt, h_text, char_w_factor) + gap_margin
                    gaps.append((s - half, s + half))

            if (not ends_only) and label_gap_enabled and gaps:
                segs = _cut_line_by_gaps(ls, gaps)
            else:
                segs = [ls]
            layer_line = layer_curva_m if is_master else layer_curva_i
            for seg in segs:
                coords = [(float(x), float(y)) for x, y in seg.coords]
                _draw_polyline(coords, layer_line, elevation=elev)

            for s in label_positions:
                s_original = float(s)
                s_clamped = max(0.0, min(float(ls.length), s_original))

                if ends_only and ls.length > 0.0:
                    shift_ref = max(0.2, min(1.0, ls.length * 0.05))
                    if s_original <= 1e-6:
                        s_clamped = min(ls.length, shift_ref)
                    elif abs(s_original - ls.length) <= 1e-6:
                        s_clamped = max(0.0, ls.length - shift_ref)

                px, py, ang = _calculate_text_position_outside(ls, s_clamped, offset_m=label_offset_m)
                if label_offset_m <= 1e-6:
                    p = ls.interpolate(s_clamped)
                    px, py = float(p.x), float(p.y)

                _add_text(_format_elev(elev, label_precision), px, py, ang)

        feat = lyr.GetNextFeature()

    try:
        memds.Destroy()
    except Exception:
        pass


__all__ = ["gerar_curvas_nivel"]
