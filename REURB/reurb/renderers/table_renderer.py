"""
Table rendering helpers.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

import math
from typing import Optional, Any

from shapely.geometry import Polygon

from reurb.renderers.text_renderer import add_centered_text
from reurb.utils.logging_utils import REURBLogger

logger = REURBLogger(__name__, verbose=False)


def format_area_br_m2(area_m2: float, casas: int = 3) -> str:
    """Formata area com milhar '.' e decimal ',' seguido de m²."""
    try:
        s = f"{float(area_m2):,.{casas}f}"
    except Exception as e:
        logger.warning(f"Falha ao formatar area: {e}")
        return f"{area_m2} m²"
    s = s.replace(",", "|").replace(".", ",").replace("|", ".")
    return f"{s} m²"


def _draw_cell(ms, x, y, w, h, layer):
    ms.add_lwpolyline([(x, y), (x + w, y), (x + w, y - h), (x, y - h), (x, y)], close=True, dxfattribs={"layer": layer})


def _add_text_center(ms, text, cx, cy, h, layer, style):
    t = ms.add_text(text, dxfattribs={"height": h, "style": style, "layer": layer})
    t.dxf.insert = (cx, cy)
    t.dxf.align_point = (cx, cy)
    t.dxf.halign = 1
    t.dxf.valign = 2
    return t


def table_header(ms, x0, y0, col_ws, header_h, ch, txt_h, layer, style):
    headers = ["Segmento", "Distância (m)", "Azimute", "Ponto"]
    x = x0
    for i, htxt in enumerate(headers):
        w = col_ws[i]
        _draw_cell(ms, x, y0, w, header_h, layer)
        _add_text_center(ms, htxt, x + w / 2, y0 - header_h / 2, txt_h, layer, style)
        x += w
    w_e = col_ws[4]
    w_n = col_ws[5]
    coord_y0 = y0
    _draw_cell(ms, x, coord_y0, w_e + w_n, ch, layer)
    _add_text_center(ms, "Coordenadas", x + (w_e + w_n) / 2, coord_y0 - ch / 2, txt_h, layer, style)
    y1 = y0 - ch
    _draw_cell(ms, x, y1, w_e, ch, layer)
    _add_text_center(ms, "E", x + w_e / 2, y1 - ch / 2, txt_h, layer, style)
    _draw_cell(ms, x + w_e, y1, w_n, ch, layer)
    _add_text_center(ms, "N", x + w_e + w_n / 2, y1 - ch / 2, txt_h, layer, style)


def table_row(ms, row_idx, x0, y0, col_ws, header_h, ch, txt_h, layer, style, seg, dist, az, pid, e, n):
    y = y0 - header_h - (row_idx * ch)
    cols = [seg, f"{dist:.2f}", az, pid, f"{e:.4f}", f"{n:.4f}"]
    x = x0
    for i, val in enumerate(cols):
        w = col_ws[i]
        _draw_cell(ms, x, y, w, ch, layer)
        _add_text_center(ms, val, x + w / 2, y - ch / 2, txt_h, layer, style)
        x += w


def create_area_table(ms, per_interesse: Optional[Polygon] = None, per_levantamento: Optional[Polygon] = None, params: Any = None) -> bool:
    """
    Cria tabela com areas de levantamento e nucleo (PER_INTERESSE).
    """
    try:
        layer_tabela = getattr(params, "layer_tabela", "TOP_TABELA")
        altura_texto = getattr(params, "altura_texto_tabela", 2.0)
        style_texto = getattr(params, "style_texto", "SIMPLEX")

        anchor = getattr(params, "area_table_anchor", None)
        offset_x = getattr(params, "tabela_offset_x", 120.0)
        offset_y = getattr(params, "tabela_offset_y", 0.0)
        cell_width = getattr(params, "area_table_cell_w", getattr(params, "tabela_cell_w", 25.0))
        cell_height = getattr(params, "tabela_cell_h", 6.0)

        area_nucleo = None
        area_levantamento = None

        if per_interesse and not per_interesse.is_empty:
            try:
                area_nucleo = float(per_interesse.area)
            except Exception as e:
                logger.warning(f"Falha ao calcular area nucleo: {e}")

        if per_levantamento and not per_levantamento.is_empty:
            try:
                area_levantamento = float(per_levantamento.area)
            except Exception as e:
                logger.warning(f"Falha ao calcular area levantamento: {e}")

        if area_nucleo is None and area_levantamento is None:
            return False

        if isinstance(anchor, tuple) and len(anchor) == 2:
            start_x, start_y = float(anchor[0]), float(anchor[1])
        else:
            start_x = offset_x
            start_y = offset_y

        total_w = cell_width * 2.0
        total_h = cell_height

        try:
            ms.add_lwpolyline(
                [
                    (start_x, start_y),
                    (start_x + total_w, start_y),
                    (start_x + total_w, start_y - total_h),
                    (start_x, start_y - total_h),
                    (start_x, start_y),
                ],
                close=True,
                dxfattribs={"layer": layer_tabela},
            )
            ms.add_line(
                (start_x + total_w / 2.0, start_y),
                (start_x + total_w / 2.0, start_y - total_h),
                dxfattribs={"layer": layer_tabela},
            )
        except Exception as e:
            logger.warning(f"Falha ao desenhar tabela: {e}")

        def add_center(titulo: str, valor: str, cx: float):
            y_mid = start_y - total_h / 2.0
            texto = f"{titulo}: {valor}"
            add_centered_text(ms, texto, cx, y_mid, altura_texto, style_texto, layer_tabela, 0.0)

        left_data = None
        right_data = None
        if area_nucleo is not None:
            left_data = ("ÁREA TOTAL DO NÚCLEO", format_area_br_m2(area_nucleo))
        if area_levantamento is not None:
            right_data = ("ÁREA TOTAL DE LEVANTAMENTO", format_area_br_m2(area_levantamento))

        if left_data and right_data:
            add_center(left_data[0], left_data[1], start_x + total_w * 0.25)
            add_center(right_data[0], right_data[1], start_x + total_w * 0.75)
        elif left_data:
            add_center(left_data[0], left_data[1], start_x + total_w * 0.5)
        elif right_data:
            add_center(right_data[0], right_data[1], start_x + total_w * 0.5)

        return True

    except Exception as e:
        logger.warning(f"Falha ao criar tabela de areas: {e}")
        return False


__all__ = [
    "format_area_br_m2",
    "create_area_table",
    "table_header",
    "table_row",
]
