"""
Lot dimensions and area labels.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

import math
from typing import List

from shapely.geometry import Point

from reurb.config.layers import LAYER_ORDINATE, LAYER_SOLEIRA_AREA, STYLE_TEXTO
from reurb.config.dimensions import Params, TOLERANCES
from reurb.geometry.segments import _noded_lote_segments
from reurb.geometry.spatial_index import SpatialIndex
from reurb.geometry.rotations import encontrar_rotacao_por_lote, encontrar_rotacao_por_via, normal_from_rotation
from reurb.renderers.dimension_renderer import add_dim_aligned
from reurb.renderers.text_renderer import add_centered_text
from reurb.utils.logging_utils import REURBLogger

logger = REURBLogger(__name__, verbose=False)

DEDUP_ROUND_SCALE = int(getattr(TOLERANCES, "DEDUP_ROUND_SCALE", 1000))
ANG_TOL = float(getattr(TOLERANCES, "ANGLE_TOLERANCE_DEG", 5.0))


def _rotular_areas_lote(ms, lotes, edificacoes, vias, params: Params):
    """
    Escreve somente a AREA do lote em LAYER_SOLEIRA_AREA para todos os poligonos de 'lotes'.
    Mesma logica de POSICIONAMENTO do fluxo TXT: base no ponto representativo da maior edificacao (fallback: lote),
    rotacao pela testada do lote (fallback: via mais proxima), texto "em cima" (offset +1*line_gap ao longo da normal).
    """

    def _upright(deg):
        if deg is None:
            return None
        a = deg % 180.0
        return a if a <= 90.0 else a - 180.0

    line_gap = max(float(getattr(params, "altura_texto_soleira", 0.5)), 0.50)

    edif_index = SpatialIndex(edificacoes) if edificacoes else None

    for lote_poly in (lotes or []):
        try:
            area_txt = f"{float(lote_poly.area):.2f} m2"
        except Exception as e:
            logger.debug(f"Falha ao formatar area do lote: {e}")
            continue

        try:
            e_sel = None
            if edificacoes:
                max_a = -1.0
                candidatos = edificacoes
                if edif_index is not None:
                    candidatos = edif_index.query(lote_poly)
                for e in candidatos:
                    try:
                        c = e.representative_point()
                        if not lote_poly.contains(c):
                            continue
                        a = float(e.area)
                        if a > max_a:
                            max_a = a
                            e_sel = e
                    except Exception as e:
                        logger.debug(f"Falha ao avaliar edificacao: {e}")
                        continue
            if e_sel is not None:
                base_x, base_y = e_sel.representative_point().coords[0]
            else:
                base_x, base_y = lote_poly.representative_point().coords[0]
        except Exception as e:
            logger.debug(f"Falha ao calcular ponto base do lote: {e}")
            continue

        rot_txt = None
        try:
            rot_txt = encontrar_rotacao_por_lote((base_x, base_y), [lote_poly], delta=getattr(params, "delta_interp", 2.0), raio=12.0)
        except Exception as e:
            logger.debug(f"Falha ao calcular rotacao por lote: {e}")
            rot_txt = None
        if rot_txt is None:
            try:
                rot_txt = encontrar_rotacao_por_via((base_x, base_y), vias or [], getattr(params, "dist_busca_rot", 50.0), getattr(params, "delta_interp", 2.0))
            except Exception as e:
                logger.debug(f"Falha ao calcular rotacao por via: {e}")
                rot_txt = None
        rot_txt = _upright(rot_txt) if rot_txt is not None else None
        nux, nuy = normal_from_rotation(rot_txt)

        tx_x = float(base_x) + nux * (1.0 * line_gap)
        tx_y = float(base_y) + nuy * (1.0 * line_gap)

        try:
            t = ms.add_text(
                area_txt,
                dxfattribs={"height": float(getattr(params, "altura_texto_area", 0.6)), "style": STYLE_TEXTO, "layer": LAYER_SOLEIRA_AREA},
            )
            t.dxf.insert = (tx_x, tx_y)
            try:
                t.dxf.halign = 1
                t.dxf.valign = 2
                t.dxf.align_point = (tx_x, tx_y)
            except Exception as e:
                logger.debug(f"Falha ao ajustar alinhamento do texto: {e}")
            try:
                t.dxf.rotation = 0.0
            except Exception as e:
                logger.debug(f"Falha ao ajustar rotacao do texto: {e}")
        except Exception as e:
            logger.debug(f"Falha ao inserir texto de area do lote: {e}")
            continue


def processar_lotes_dimensoes(ms, doc, params: Params, lotes: list):
    """Gera cotas de dimensoes dos lotes no DXF."""
    if not lotes:
        return
    style_texto = getattr(params, "style_texto", STYLE_TEXTO)
    dim_style = getattr(params, "lote_dim_style", "Cota_Lote") or style_texto
    layer_dim = LAYER_ORDINATE
    min_len = float(getattr(params, "lote_dim_min_len_m", 0.0))
    offset_m = float(getattr(params, "lote_dim_offset_m", 0.60))
    min_spacing = float(getattr(params, "lote_dim_min_spacing_m", 0.0))
    min_spacing2 = min_spacing * min_spacing
    snap_tol = float(getattr(params, "lote_dim_snap_tol_m", 0.20))

    placed_pts: list[tuple[float, float]] = []
    segments = _noded_lote_segments(lotes, snap_tol=snap_tol)
    kept_mid_pts: list[tuple[float, float, float]] = []
    dim_close_tol = 0.50
    dim_val_tol = 0.02
    sum_tol = max(0.01, snap_tol * 2.0)
    ang_tol = ANG_TOL
    line_tol = max(0.01, snap_tol * 2.0)

    seg_meta = []
    seg_key_map = {}
    for s in segments:
        try:
            coords = list(s.coords)
            if len(coords) < 2:
                continue
            a, b = coords[0], coords[-1]
            seg_len = float(math.hypot(b[0] - a[0], b[1] - a[1]))
            if seg_len <= 0:
                continue
            mx = (a[0] + b[0]) * 0.5
            my = (a[1] + b[1]) * 0.5
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            ang = math.degrees(math.atan2(dy, dx))
            L = math.hypot(dx, dy)
            ux, uy = dx / L, dy / L
            t1 = a[0] * ux + a[1] * uy
            t2 = b[0] * ux + b[1] * uy
            tmin, tmax = (t1, t2) if t1 <= t2 else (t2, t1)
            idx = len(seg_meta)
            seg_meta.append({"a": a, "b": b, "len": seg_len, "mx": mx, "my": my, "ang": ang, "ux": ux, "uy": uy, "tmin": tmin, "tmax": tmax})
            ax, ay = int(round(a[0] * DEDUP_ROUND_SCALE)), int(round(a[1] * DEDUP_ROUND_SCALE))
            bx, by = int(round(b[0] * DEDUP_ROUND_SCALE)), int(round(b[1] * DEDUP_ROUND_SCALE))
            key = (ax, ay, bx, by) if (ax, ay) <= (bx, by) else (bx, by, ax, ay)
            seg_key_map[key] = idx
        except Exception as e:
            logger.debug(f"Falha ao processar metadados de segmento: {e}")
            continue

    def _ang_diff_180(a: float, b: float) -> float:
        d = abs((a - b) % 180.0)
        return d if d <= 90.0 else 180.0 - d

    skip_idx = set()
    n = len(seg_meta)
    for i in range(n):
        if i in skip_idx:
            continue
        si = seg_meta[i]
        candidates = []
        for j in range(n):
            if j == i:
                continue
            sj = seg_meta[j]
            if sj["len"] >= si["len"]:
                continue
            ang_diff = _ang_diff_180(si["ang"], sj["ang"])
            if ang_diff > ang_tol:
                continue
            dxm = sj["mx"] - si["mx"]
            dym = sj["my"] - si["my"]
            dist_line = abs(dxm * (-si["uy"]) + dym * (si["ux"]))
            if dist_line > line_tol:
                continue
            t1 = sj["a"][0] * si["ux"] + sj["a"][1] * si["uy"]
            t2 = sj["b"][0] * si["ux"] + sj["b"][1] * si["uy"]
            tmin, tmax = (t1, t2) if t1 <= t2 else (t2, t1)
            if tmax < si["tmin"] - sum_tol or tmin > si["tmax"] + sum_tol:
                continue
            candidates.append({"tmin": tmin, "tmax": tmax})

        if len(candidates) >= 2:
            candidates.sort(key=lambda x: x["tmin"])
            cover_min = candidates[0]["tmin"]
            cover_max = candidates[0]["tmax"]
            for c in candidates[1:]:
                if c["tmin"] <= cover_max + sum_tol:
                    cover_max = max(cover_max, c["tmax"])
                else:
                    break
            if cover_min <= si["tmin"] + sum_tol and cover_max >= si["tmax"] - sum_tol:
                skip_idx.add(i)

    for s in segments:
        try:
            coords = list(s.coords)
            if len(coords) < 2:
                continue
            a, b = coords[0], coords[-1]
        except Exception as e:
            logger.debug(f"Falha ao ler segmento: {e}")
            continue
        seg_len = float(math.hypot(b[0] - a[0], b[1] - a[1]))
        if seg_len < min_len or round(seg_len, 2) == 0.0:
            continue
        mid = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        dup = False
        for mx, my, mlen in kept_mid_pts:
            if abs(mlen - seg_len) <= dim_val_tol:
                dx = mid[0] - mx
                dy = mid[1] - my
                if (dx * dx + dy * dy) <= dim_close_tol * dim_close_tol:
                    dup = True
                    break
        if dup:
            continue
        if seg_meta:
            ax, ay = int(round(a[0] * DEDUP_ROUND_SCALE)), int(round(a[1] * DEDUP_ROUND_SCALE))
            bx, by = int(round(b[0] * DEDUP_ROUND_SCALE)), int(round(b[1] * DEDUP_ROUND_SCALE))
            key = (ax, ay, bx, by) if (ax, ay) <= (bx, by) else (bx, by, ax, ay)
            idx = seg_key_map.get(key)
            if idx is not None and idx in skip_idx:
                continue
        if min_spacing2 > 0.0:
            too_close = False
            for px, py in placed_pts:
                dx = mid[0] - px
                dy = mid[1] - py
                if (dx * dx + dy * dy) < min_spacing2:
                    too_close = True
                    break
            if too_close:
                continue

        try:
            add_dim_aligned(
                ms,
                (a[0], a[1]),
                (b[0], b[1]),
                +1.0,
                float(offset_m),
                layer_dim,
                float(getattr(params, "dimtxt_ordinate", 0.5)),
                float(getattr(params, "dimasz_ordinate", 0.2)),
                2,
                dim_style,
            )
            placed_pts.append(mid)
            kept_mid_pts.append((mid[0], mid[1], seg_len))
        except Exception as e:
            logger.debug(f"Falha ao inserir cota alinhada: {e}")


__all__ = ["processar_lotes_dimensoes", "_rotular_areas_lote"]
