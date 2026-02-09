"""
Perimeter processing.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

import math
from typing import List, Tuple

from shapely.geometry import Polygon, Point

from reurb.config.layers import (
    BLOCO_VERTICE,
    LAYER_VERTICE_PTO,
    LAYER_VERTICE_TXT,
    LAYER_TABELA,
    LAYER_ORDINATE,
    STYLE_TEXTO,
)
from reurb.config.dimensions import Params
from reurb.geometry.calculations import dist, azimute, bbox_max
from reurb.renderers.table_renderer import table_header, table_row
from reurb.renderers.dimension_renderer import add_ordinate
from reurb.utils.logging_utils import REURBLogger

logger = REURBLogger(__name__, verbose=False)


def _vertices_do_poligono(poly: Polygon) -> List[Tuple[float, float]]:
    """Usa a sequencia do contorno EXTERNO invertida e remove o ultimo vertice duplicado."""
    coords = list(poly.exterior.coords)
    coords = list(reversed(coords))
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [(x, y) for (x, y) in coords]


def _dms_str(az_deg: float) -> str:
    d = int(az_deg)
    m_f = (az_deg - d) * 60.0
    m = int(m_f)
    s = int(round((m_f - m) * 60.0))
    if s == 60:
        s = 0
        m += 1
    if m == 60:
        m = 0
        d = (d + 1) % 360
    deg = "\u00B0"
    return f"{d:02d}{deg}{m:02d}'{s:02d}''"


def processar_perimetros(ms, doc, params: Params, perimetros: list, perimetro_limite=None):
    """Processa perimetros e gera vertices, tabela e ordinate dimensions."""
    if not perimetros:
        return
    poly = perimetros[0]
    style_texto = getattr(params, "style_texto", STYLE_TEXTO)
    V = _vertices_do_poligono(poly)

    ref_geom = perimetro_limite if perimetro_limite is not None else poly
    try:
        centroid = ref_geom.centroid if ref_geom is not None else poly.centroid
    except Exception:
        centroid = None
    try:
        boundary_geom = ref_geom.buffer(0) if ref_geom is not None else poly.buffer(0)
    except Exception:
        boundary_geom = None
    offset_m = float(getattr(params, "p_label_offset_m", 1.8))
    step_m = max(0.1, float(getattr(params, "p_label_offset_step", 0.5)))
    max_dist = float(getattr(params, "p_label_max_dist", 1.2))
    if max_dist < 0.1:
        max_dist = 1.2
    if offset_m > max_dist:
        offset_m = max_dist
    if step_m > max_dist:
        step_m = max_dist
    placed_boxes = []

    def _text_box(cx, cy, text, h, w_factor=0.6):
        w = max(0.01, len(str(text)) * float(h) * float(w_factor))
        hh = float(h)
        return (cx - w / 2.0, cy - hh / 2.0, cx + w / 2.0, cy + hh / 2.0)

    def _boxes_overlap(a, b) -> bool:
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

    def _find_label_pos(px, py, dir_x, dir_y, text, h):
        perp_x, perp_y = -dir_y, dir_x
        offsets = [0.0, step_m, -step_m, 2 * step_m, -2 * step_m]
        dist0 = offset_m
        while dist0 <= max_dist + 1e-6:
            for lat in offsets:
                cx = px + dir_x * dist0 + perp_x * lat
                cy = py + dir_y * dist0 + perp_y * lat
                if math.hypot(cx - px, cy - py) > max_dist + 1e-6:
                    continue
                box = _text_box(cx, cy, text, h)
                if any(_boxes_overlap(box, b) for b in placed_boxes):
                    continue
                placed_boxes.append(box)
                return cx, cy
            dist0 += step_m
        cx = px + dir_x * offset_m
        cy = py + dir_y * offset_m
        placed_boxes.append(_text_box(cx, cy, text, h))
        return cx, cy

    for i, (x, y) in enumerate(V, start=1):
        ms.add_blockref(BLOCO_VERTICE, (x, y), dxfattribs={"layer": LAYER_VERTICE_PTO})
        px, py = float(x), float(y)
        if centroid is not None:
            dir_x = px - float(centroid.x)
            dir_y = py - float(centroid.y)
            norm = float(math.hypot(dir_x, dir_y))
            if norm < 1e-6:
                dir_x, dir_y = 1.0, 0.0
            else:
                dir_x /= norm
                dir_y /= norm
            dist0 = offset_m
            candidate = Point(px + dir_x * dist0, py + dir_y * dist0)
            tries = 0
            while boundary_geom is not None and (boundary_geom.contains(candidate) or boundary_geom.distance(candidate) < 0.05) and tries < 40:
                dist0 += step_m
                if dist0 > max_dist:
                    break
                candidate = Point(px + dir_x * dist0, py + dir_y * dist0)
                tries += 1
            px, py = candidate.x, candidate.y
            label_x, label_y = _find_label_pos(px, py, dir_x, dir_y, f"P{i}", params.altura_texto_P)
        else:
            dir_x, dir_y = 1.0, 0.0
            label_x, label_y = _find_label_pos(px, py, dir_x, dir_y, f"P{i}", params.altura_texto_P)

        t = ms.add_text(f"P{i}", dxfattribs={"height": params.altura_texto_P, "style": style_texto, "layer": LAYER_VERTICE_TXT})
        t.dxf.insert = (label_x, label_y)
        t.dxf.align_point = (label_x, label_y)
        t.dxf.halign = 1
        t.dxf.valign = 2

    max_x, max_y = bbox_max(V)
    x0 = max_x + params.tabela_offset_x
    y0 = max_y + params.tabela_offset_y
    ch, th = params.tabela_cell_h, params.altura_texto_tabela
    header_h = float(getattr(params, "tabela_header_h", ch))
    col_ws = [
        float(getattr(params, "tabela_col_w_segmento", 20.0)),
        float(getattr(params, "tabela_col_w_distancia", 20.0)),
        float(getattr(params, "tabela_col_w_azimute", 20.0)),
        float(getattr(params, "tabela_col_w_ponto", 15.0)),
        float(getattr(params, "tabela_col_w_e", 20.0)),
        float(getattr(params, "tabela_col_w_n", 20.0)),
    ]

    table_header(ms, x0, y0, col_ws, header_h, ch, th, LAYER_TABELA, style_texto)

    n = len(V)
    for i in range(n):
        a = V[i]
        b = V[(i + 1) % n]
        seg = f"{i+1} - {((i+1)%n)+1}"
        d = dist(a, b)
        az = _dms_str(azimute(a, b))
        pid = f"P{i+1}"
        e, ncoord = a[0], a[1]
        table_row(ms, i, x0, y0, col_ws, header_h, ch, th, LAYER_TABELA, style_texto, seg, d, az, pid, e, ncoord)

    try:
        setattr(params, "area_table_anchor", (x0, y0 - (2 + n) * ch - ch))
    except Exception as e:
        logger.warning(f"Falha ao setar area_table_anchor: {e}")

    for i in range(n):
        a = V[i]
        b = V[(i + 1) % n]
        mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
        d = dist(a, b)
        az = azimute(a, b)
        texto = f"\\S{d:.2f}/ Az. {_dms_str(az)};"
        add_ordinate(ms, mid, texto, LAYER_ORDINATE, params, style_texto)


__all__ = ["processar_perimetros"]
