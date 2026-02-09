"""
Road measurements processing.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

import math
from typing import List, Optional, Callable, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union, linemerge

from reurb.renderers.text_renderer import place_text_center
from reurb.renderers.dimension_renderer import add_dim_aligned
from reurb.geometry.rotations import tangent_angle_at_frac, upright_angle_deg


def _seg_len(p1, p2) -> float:
    if p1 is None or p2 is None:
        return 0.0
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _round_pt(pt, tol=1e-3):
    return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)


def _tangent_angle_pts(p0, p1) -> float:
    dx, dy = (p1[0] - p0[0]), (p1[1] - p0[1])
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def _dedup_merge_lines(eixos: List[LineString]) -> List[LineString]:
    if not eixos:
        return []
    try:
        merged = linemerge(unary_union([ls for ls in eixos if not ls.is_empty and ls.length > 0]))
        if isinstance(merged, LineString):
            return [merged]
        return [ls for ls in merged.geoms if isinstance(ls, LineString) and ls.length > 0.5]
    except Exception:
        out, seen = [], set()
        for ls in eixos:
            sig = (
                round(ls.bounds[0], 3),
                round(ls.bounds[1], 3),
                round(ls.bounds[2], 3),
                round(ls.bounds[3], 3),
                round(ls.length, 3),
            )
            if sig not in seen:
                seen.add(sig)
                out.append(ls)
        return out


def _union_guides(guia_list: List[LineString]):
    try:
        return linemerge(unary_union([g for g in guia_list if not g.is_empty]))
    except Exception:
        return unary_union([g for g in guia_list if not g.is_empty])


def _lots_boundary(lotes_polygons: List[Polygon]):
    if not lotes_polygons:
        return None
    try:
        u = unary_union(lotes_polygons)
        return u.boundary
    except Exception:
        try:
            lines = []
            for p in lotes_polygons:
                b = p.boundary
                if b.is_empty:
                    continue
                if b.geom_type == "LineString":
                    lines.append(list(b.coords))
                elif b.geom_type == "MultiLineString":
                    for g in b.geoms:
                        lines.append(list(g.coords))
            return linemerge(unary_union([LineString(c) for c in lines if len(c) >= 2]))
        except Exception:
            return None


def _closest_hits(center: Point, n_unit: Tuple[float, float], span: float, geom) -> Tuple[Tuple[float, float], Tuple[float, float]] | None:
    if geom is None:
        return None
    pA = (center.x - n_unit[0] * span, center.y - n_unit[1] * span)
    pB = (center.x + n_unit[0] * span, center.y + n_unit[1] * span)
    cross = LineString([pA, pB])

    inter = cross.intersection(geom)
    if inter.is_empty:
        return None

    pts: List[Point] = []
    gt = inter.geom_type
    if gt == "Point":
        pts = [inter]  # type: ignore
    elif gt == "MultiPoint":
        pts = list(inter.geoms)  # type: ignore
    elif gt in ("LineString", "MultiLineString"):
        try:
            geoms = [inter] if gt == "LineString" else list(inter.geoms)
            for g in geoms:  # type: ignore
                c = list(g.coords)
                pts.append(Point(c[0]))
                pts.append(Point(c[-1]))
        except Exception:
            pass
    if not pts:
        return None

    def proj_t(pt: Point) -> float:
        vx, vy = (pt.x - center.x), (pt.y - center.y)
        return vx * n_unit[0] + vy * n_unit[1]

    t_vals = [(proj_t(p), p) for p in pts]
    t_pos = sorted([tp for tp in t_vals if tp[0] > 0.0], key=lambda x: x[0])
    t_neg = sorted([tp for tp in t_vals if tp[0] < 0.0], key=lambda x: -x[0])

    if not t_pos or not t_neg:
        return None

    p_pos = t_pos[0][1]
    p_neg = t_neg[0][1]
    return (float(p_neg.x), float(p_neg.y)), (float(p_pos.x), float(p_pos.y))


def medir_e_rotular_vias(
    ms,
    eixos: List[LineString],
    bordas_guia: List[LineString],
    lotes_polygons: List[Polygon],
    testada_extra_lines: List[LineString],
    texto_altura: float,
    offset_texto_m: float,
    cross_span_m: float,
    amostras_fracs: tuple = (1 / 3, 2 / 3),
    name_provider: Optional[Callable[[float, float], str]] = None,
    ativar_testada_testada: bool = True,
    nome_offset_m: float = 0.60,
    nome_offset_side: str = "auto",
    nome_sufixo: str = "",
    style_texto: str = "SIMPLEX",
    dim_text_style: str | None = None,
    layer_via_medida: str = "TOP_COTAS_VIARIO",
    layer_via_nome: str = "TOP_SISTVIA",
    sample_mode: str = "entre_intersecoes",
    offset_lote_lote_extra_m: float = 0.60,
    dim_gap_m: float = 0.60,
    dim_min_len_m: float = 12.0,
    dim_min_spacing_m: float = 25.0,
    dim_max_por_trecho: int = 2,
    dim_max_dist_m: float = 20.0,
    dim_min_sep_area_m: float = 10.0,
    dim_equal_tol_m: float = 0.05,
    nome_offset_add_dim_m: float | None = None,
    nome_side_mode: str = "oposto_dim",
    nome_shift_along_m: float = 6.0,
    nome_case: str = "as_is",
):
    if dim_text_style is None:
        dim_text_style = style_texto

    for lyr in [layer_via_medida, layer_via_nome]:
        if lyr not in ms.doc.layers:
            ms.doc.layers.add(lyr)

    lotes_union = unary_union(lotes_polygons) if lotes_polygons else None
    lotes_boundary = _lots_boundary(lotes_polygons)
    guia_union = _union_guides(bordas_guia)

    arrow_size_factor = 0.5
    span = max(float(cross_span_m), 1.0)

    eixos_proc = _dedup_merge_lines([ls for ls in eixos if not ls.is_empty and ls.length > 0.5])

    for idx_full, eixo_full in enumerate(eixos_proc, start=1):
        if eixo_full.length < 1.0:
            continue

        if eixo_full.length >= 5.0:
            try:
                coords = list(eixo_full.coords)
                best_len = -1.0
                best_seg = None
                for i in range(len(coords) - 1):
                    a = coords[i]
                    b = coords[i + 1]
                    seg_len = float(math.hypot(b[0] - a[0], b[1] - a[1]))
                    if seg_len > best_len:
                        best_len = seg_len
                        best_seg = (a, b)
                if best_seg is not None:
                    mid_all = Point((best_seg[0][0] + best_seg[1][0]) * 0.5, (best_seg[0][1] + best_seg[1][1]) * 0.5)
                else:
                    mid_all = eixo_full.interpolate(0.5, normalized=True)
            except Exception:
                mid_all = eixo_full.interpolate(0.5, normalized=True)
            nome = None
            if name_provider:
                try:
                    nome = name_provider(float(mid_all.x), float(mid_all.y))
                except Exception:
                    nome = None
            if not nome:
                nome = f"VIA-{idx_full:03d}"
            if nome_sufixo:
                nome = f"{nome}{nome_sufixo}"
            if str(nome_case).lower() == "upper":
                nome = nome.upper()

            if "best_seg" in locals() and best_seg is not None:
                ang_nome = math.degrees(math.atan2(best_seg[1][1] - best_seg[0][1], best_seg[1][0] - best_seg[0][0]))
            else:
                ang_nome = tangent_angle_at_frac(eixo_full, 0.5, delta_norm=0.02)
            ang_rad = math.radians(ang_nome)
            tx, ty = math.cos(ang_rad), math.sin(ang_rad)
            nx_nom, ny_nom = -math.sin(ang_rad), math.cos(ang_rad)

            if nome_side_mode == "oposto_dim":
                side = +1.0
            else:
                side = +1.0
                if isinstance(nome_offset_side, (int, float)):
                    side = float(nome_offset_side)
                elif isinstance(nome_offset_side, str) and nome_offset_side.strip() != "auto":
                    side = +1.0 if nome_offset_side.strip() == "+1" else -1.0
                else:
                    if lotes_union is not None:
                        test_pt = Point(mid_all.x + nx_nom * nome_offset_m, mid_all.y + ny_nom * nome_offset_m)
                        try:
                            if lotes_union.buffer(0).contains(test_pt):
                                side = -1.0
                        except Exception:
                            side = +1.0

            nome_extra = offset_texto_m if nome_offset_add_dim_m is None else float(nome_offset_add_dim_m)
            nome_offset_total = float(nome_offset_m) + float(nome_extra)

            base_x = float(mid_all.x + tx * float(nome_shift_along_m))
            base_y = float(mid_all.y + ty * float(nome_shift_along_m))

            try:
                place_text_center(
                    ms,
                    nome,
                    base_x + side * nx_nom * nome_offset_total,
                    base_y + side * ny_nom * nome_offset_total,
                    float(texto_altura),
                    rotation_deg=ang_nome,
                    layer=layer_via_nome,
                    style=style_texto,
                )
            except Exception:
                pass

        trechos: List[LineString]
        if sample_mode == "por_trecho":
            coords = list(eixo_full.coords)
            trechos = []
            for i in range(len(coords) - 1):
                a, b = coords[i], coords[i + 1]
                if math.hypot(b[0] - a[0], b[1] - a[1]) > 0.75:
                    trechos.append(LineString([a, b]))
        else:
            trechos = [eixo_full]

        placed_mid_pts: list[tuple[float, float]] = []
        min_sep2 = float(dim_min_sep_area_m) ** 2 if dim_min_sep_area_m else 0.0

        for eixo in trechos:
            try:
                eixo_len = float(eixo.length)
            except Exception:
                eixo_len = 0.0
            if eixo_len < float(dim_min_len_m):
                continue

            if dim_min_spacing_m and float(dim_min_spacing_m) > 0:
                max_by_len = max(1, int(eixo_len / float(dim_min_spacing_m)))
                n = max(1, min(int(dim_max_por_trecho), max_by_len))
                fracs_eff = np.linspace(0.20, 0.80, n).tolist()
            else:
                fracs_eff = list(amostras_fracs)

            for frac in fracs_eff:
                frac = max(0.05, min(0.95, float(frac)))
                mid = eixo.interpolate(frac, normalized=True)
                if min_sep2 > 0.0:
                    mx, my = float(mid.x), float(mid.y)
                    too_close = False
                    for px, py in placed_mid_pts:
                        dx = mx - px
                        dy = my - py
                        if (dx * dx + dy * dy) < min_sep2:
                            too_close = True
                            break
                    if too_close:
                        continue
                ang_loc = tangent_angle_at_frac(eixo, frac, delta_norm=0.02)
                ang_rad = math.radians(ang_loc)
                nx, ny = -math.sin(ang_rad), math.cos(ang_rad)

                hits_guias = _closest_hits(Point(mid.x, mid.y), (nx, ny), span * 0.5, guia_union)
                if not hits_guias:
                    continue
                g_neg, g_pos = _round_pt(hits_guias[0]), _round_pt(hits_guias[1])
                if g_neg == g_pos:
                    continue

                hits_lotes = _closest_hits(Point(mid.x, mid.y), (nx, ny), span, lotes_boundary) if lotes_boundary is not None else None
                l_neg = l_pos = None
                if hits_lotes:
                    l_neg, l_pos = _round_pt(hits_lotes[0]), _round_pt(hits_lotes[1])

                gap = max(0.1, float(dim_gap_m))
                off_gg = float(offset_texto_m)
                off_gl = float(offset_texto_m) + gap
                off_ll = float(offset_texto_m) + 2.0 * gap + float(offset_lote_lote_extra_m)

                max_dist = float(dim_max_dist_m)
                dist_gg = float(Point(g_neg).distance(Point(g_pos)))
                eq_tol = float(dim_equal_tol_m)
                if dist_gg > max_dist:
                    continue

                add_dim_aligned(ms, g_neg, g_pos, -1.0, off_gg, layer_via_medida, float(texto_altura), arrow_size_factor * float(texto_altura), 2, dim_text_style)

                if l_neg is not None:
                    dist_gl_neg = float(Point(g_neg).distance(Point(l_neg)))
                    if dist_gl_neg <= max_dist:
                        add_dim_aligned(ms, g_neg, l_neg, -1.0, off_gl, layer_via_medida, float(texto_altura), arrow_size_factor * float(texto_altura), 2, dim_text_style)
                if l_pos is not None:
                    dist_gl_pos = float(Point(g_pos).distance(Point(l_pos)))
                    if dist_gl_pos <= max_dist:
                        add_dim_aligned(ms, g_pos, l_pos, -1.0, off_gl, layer_via_medida, float(texto_altura), arrow_size_factor * float(texto_altura), 2, dim_text_style)

                if (l_neg is not None) and (l_pos is not None) and (l_neg != l_pos):
                    dist_ll = float(Point(l_neg).distance(Point(l_pos)))
                    if dist_ll <= max_dist and abs(dist_ll - dist_gg) > eq_tol:
                        add_dim_aligned(ms, l_neg, l_pos, -1.0, off_ll, layer_via_medida, float(texto_altura), arrow_size_factor * float(texto_altura), 2, dim_text_style)
                if min_sep2 > 0.0:
                    placed_mid_pts.append((float(mid.x), float(mid.y)))


__all__ = ["medir_e_rotular_vias"]
