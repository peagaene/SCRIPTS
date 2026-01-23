# === vias_medidas.py ===
from __future__ import annotations
import math
from typing import List, Optional, Callable, Tuple

import ezdxf
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union, linemerge


# ===========================
# utilidades geomAtricas
# ===========================

EPS_DIM = 1e-4


def _seg_len(p1, p2) -> float:
    if p1 is None or p2 is None:
        return 0.0
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _round_pt(pt, tol=1e-3):
    return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)


def _upright_angle_deg(a: float) -> float:
    """Normaliza Angulo para manter o texto 'em pA' (-90..+90)."""
    while a <= -180.0:
        a += 360.0
    while a > 180.0:
        a -= 360.0
    if a > 90.0:
        a -= 180.0
    if a <= -90.0:
        a += 180.0
    if abs(a) > 89.0:
        a = 90.0
    return float(a)


def _tangent_angle_pts(p0, p1) -> float:
    dx, dy = (p1[0] - p0[0]), (p1[1] - p0[1])
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def _tangent_angle_at_frac(ls: LineString, frac: float, delta_norm: float = 0.01) -> float:
    """
    Angulo (graus) da tangente local do eixo na fraAAo 'frac' (0..1).
    Usa janela +/- delta_norm para derivar a direAAo e devolve 'em pA'.
    """
    frac = max(0.0, min(1.0, float(frac)))
    f0 = max(0.0, frac - float(delta_norm))
    f1 = min(1.0, frac + float(delta_norm))
    p0 = ls.interpolate(f0, normalized=True)
    p1 = ls.interpolate(f1, normalized=True)
    dx, dy = (p1.x - p0.x), (p1.y - p0.y)
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 0.0
    ang = math.degrees(math.atan2(dy, dx))
    return _upright_angle_deg(ang)


def _place_text_center(ms, content: str, x: float, y: float, height: float,

                       rotation_deg: float, layer: str, style: str):

    t = ms.add_text(content, dxfattribs={"height": height, "style": style, "layer": layer})

    t.dxf.rotation = float(rotation_deg)

    t.dxf.insert = (x, y)

    t.dxf.align_point = (x, y)

    t.dxf.halign = 1  # center

    t.dxf.valign = 2  # middle

    return t




def _add_dim_aligned(ms, p1, p2, offset_sign, offset_mag,
                     layer, text_height, arrow_size, dim_decimals_or_flag, style_texto):
    """Cria dimensAo alinhada robusta (pula segmentos degenerados e protege render)."""
    if p1 is None or p2 is None:
        return None
    if _seg_len(p1, p2) < EPS_DIM:
        return None
    try:
        offset_mag = float(offset_mag)
    except Exception:
        offset_mag = 0.0
    if not offset_mag or abs(offset_mag) < 1e-6:
        offset_mag = max(float(text_height) if text_height else 0.01, 0.01)
    try:
        dimdec = int(dim_decimals_or_flag) if isinstance(dim_decimals_or_flag, (int, float)) else 2
    except Exception:
        dimdec = 2

    overrides = {
        "dimtxt": float(text_height) if text_height is not None else 0.25,
        "dimasz": float(arrow_size) if arrow_size is not None else 0.18,
        "dimtxsty": style_texto,
        "dimdec": dimdec,
    }
    try:
        dim = ms.add_aligned_dim(
            p1=p1, p2=p2,
            distance=float(offset_sign) * float(offset_mag),
            dxfattribs={"layer": layer},
            override=overrides,
        )
    except Exception:
        return None

    try:
        dim.render()
    except Exception:
        return None

    return dim


# ===========================
# helpers de topologia
# ===========================

def _dedup_merge_lines(eixos: List[LineString]) -> List[LineString]:
    """Funde rede de eixos e devolve linhas jA partidas em interseAAes (entre interseAAes)."""
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
            sig = (round(ls.bounds[0], 3), round(ls.bounds[1], 3),
                   round(ls.bounds[2], 3), round(ls.bounds[3], 3), round(ls.length, 3))
            if sig not in seen:
                seen.add(sig)
                out.append(ls)
        return out


def _union_guides(guia_list: List[LineString]):
    """Une as bordas de guia em uma geometria Aonica para interseAAo rApida."""
    try:
        return linemerge(unary_union([g for g in guia_list if not g.is_empty]))
    except Exception:
        return unary_union([g for g in guia_list if not g.is_empty])


def _lots_boundary(lotes_polygons: List[Polygon]):
    """Borda (alinhamento) dos lotes como MultiLineString/GeometryCollection."""
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
    """
    InterseAAo da seAAo transversal com 'geom'.
    Retorna os dois pontos mais prA3ximos de cada lado (negativo e positivo) ao longo da normal.
    """
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
                pts.append(Point(c[0])); pts.append(Point(c[-1]))
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


# ===========================
# principal
# ===========================

def medir_e_rotular_vias(ms,
                         eixos: List[LineString],
                         bordas_guia: List[LineString],
                         lotes_polygons: List[Polygon],
                         testada_extra_lines: List[LineString],
                         texto_altura: float,
                         offset_texto_m: float,
                         cross_span_m: float,
                         amostras_fracs: tuple = (1/3, 2/3),  # duas secoes por trecho
                         name_provider: Optional[Callable[[float, float], str]] = None,
                         ativar_testada_testada: bool = True,
                         nome_offset_m: float = 0.60,
                         nome_offset_side: str = "auto",  # "+1" | "-1" | "auto" (usado so em nome_side_mode="auto")
                         nome_sufixo: str = "",
                         style_texto: str = "SIMPLEX",
                         dim_text_style: str | None = None,
                         layer_via_medida: str = "TOP_COTAS_VIARIO",
                         layer_via_nome: str = "TOP_SISTVIA",
                         sample_mode: str = "entre_intersecoes",  # "entre_intersecoes" | "por_trecho"
                         offset_lote_lote_extra_m: float = 0.60,   # empurra a cota lotelote
                         nome_offset_add_dim_m: float | None = None,  # extra normal p/ nome
                         nome_side_mode: str = "oposto_dim",  # "oposto_dim" | "auto"
                         nome_shift_along_m: float = 6.0,  # deslocamento ao longo do eixo p/ afastar do bloco de cotas
                         nome_case: str = "as_is",  # "as_is" | "upper"
                         ):
    """
    Mede largura viaria com intersecao real:
      - GuiaGuia (largura total)
      - GuiaLote (calcadas)
      - LoteLote (alinhamento)

    Padroes:
       2 secoes por trecho (1/3 e 2/3), com menos poluicao (entre intersecoes).
       Nome do lado oposto as dimensoes e deslocado ao longo do eixo.
    """

    if dim_text_style is None:
        dim_text_style = style_texto
# garante layers
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

        # ---------- nome da via (1x por eixo mesclado) ----------
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

        ang_nome = _tangent_angle_at_frac(eixo_full, 0.5, delta_norm=0.02)
        ang_rad = math.radians(ang_nome)
        # tangente e normal
        tx, ty = math.cos(ang_rad), math.sin(ang_rad)
        nx_nom, ny_nom = -math.sin(ang_rad), math.cos(ang_rad)

        # lado do nome
        if nome_side_mode == "oposto_dim":
            side = +1.0  # dimensAes usam -1.0 => nome no lado oposto
        else:
            # modo "auto": heurAstica com lotes/usuArio
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

        # aplica tambAm deslocamento AO LONGO do eixo
        base_x = float(mid_all.x + tx * float(nome_shift_along_m))
        base_y = float(mid_all.y + ty * float(nome_shift_along_m))

        try:
            _place_text_center(
                ms, nome,
                base_x + side * nx_nom * nome_offset_total,
                base_y + side * ny_nom * nome_offset_total,
                float(texto_altura),
                rotation_deg=ang_nome, layer=layer_via_nome, style=style_texto,
            )
        except Exception:
            pass

        # ---------- amostragem para medir ----------
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

        for eixo in trechos:
            for frac in amostras_fracs:
                frac = max(0.05, min(0.95, float(frac)))
                mid = eixo.interpolate(frac, normalized=True)
                ang_loc = _tangent_angle_at_frac(eixo, frac, delta_norm=0.02)
                ang_rad = math.radians(ang_loc)
                nx, ny = -math.sin(ang_rad), math.cos(ang_rad)

                # 1) guia a guia
                hits_guias = _closest_hits(Point(mid.x, mid.y), (nx, ny), span * 0.5, guia_union)
                if not hits_guias:
                    continue
                g_neg, g_pos = _round_pt(hits_guias[0]), _round_pt(hits_guias[1])
                if g_neg == g_pos:
                    continue

                # 2) lote a lote
                hits_lotes = _closest_hits(Point(mid.x, mid.y), (nx, ny), span, lotes_boundary) if lotes_boundary is not None else None
                l_neg = l_pos = None
                if hits_lotes:
                    l_neg, l_pos = _round_pt(hits_lotes[0]), _round_pt(hits_lotes[1])

                # a) GuiaaGuia
                _add_dim_aligned(ms, g_neg, g_pos, -1.0, float(offset_texto_m),
                                 layer_via_medida, float(texto_altura),
                                 arrow_size_factor * float(texto_altura), 2, dim_text_style)

                # b) GuiaaLote
                if l_neg is not None:
                    _add_dim_aligned(ms, g_neg, l_neg, -1.0, float(offset_texto_m),
                                     layer_via_medida, float(texto_altura),
                                     arrow_size_factor * float(texto_altura), 2, dim_text_style)
                if l_pos is not None:
                    _add_dim_aligned(ms, g_pos, l_pos, -1.0, float(offset_texto_m),
                                     layer_via_medida, float(texto_altura),
                                     arrow_size_factor * float(texto_altura), 2, dim_text_style)

                # c) LoteaLote (offset adicional)
                if (l_neg is not None) and (l_pos is not None) and (l_neg != l_pos):
                    dist_ll = float(offset_texto_m) + float(offset_lote_lote_extra_m)
                    _add_dim_aligned(ms, l_neg, l_pos, -1.0, dist_ll,
                                     layer_via_medida, float(texto_altura),
                                     arrow_size_factor * float(texto_altura), 2, dim_text_style)





