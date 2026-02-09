"""
DXF I/O utilities.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

import os
import math
from typing import List, Set

import ezdxf
from shapely.geometry import Polygon, LineString, MultiLineString

from reurb.utils.logging_utils import REURBLogger
from reurb.config.layers import (
    STYLE_TEXTO,
    LAYER_SETAS_SAIDA,
    LAYER_SOLEIRA_BLOCO,
    LAYER_SOLEIRA_NUM_PAV,
    LAYER_SOLEIRA_AREA,
    LAYER_VERTICE_PTO,
    LAYER_VERTICE_TXT,
    LAYER_TABELA,
    LAYER_ORDINATE,
    LAYER_CURVA_INTER,
    LAYER_CURVA_MESTRA,
    LAYER_CURVA_ROTULO,
    LAYER_VIA_MEDIDA,
    LAYER_VIA_NOME,
    LAYER_LOTES,
    LAYER_EDIF,
    LAYER_PERIMETRO,
    LAYER_EIXO_SETAS,
    ROTATION_LAYERS,
)

logger = REURBLogger(__name__, verbose=False)

# ---------------- Abertura / salvamento ----------------

def abrir_dxf_simbologia(path_dxf: str):
    doc = ezdxf.readfile(path_dxf)
    return doc, doc.modelspace()


def salvar_dxf(doc, pasta_saida: str, nome_arquivo: str) -> str:
    os.makedirs(pasta_saida, exist_ok=True)
    path = os.path.join(pasta_saida, nome_arquivo)
    doc.saveas(path)
    return path


# ---------------- Garantias de estilo/layers ----------------

def garantir_estilos_blocos(doc, params=None):
    # estilo de texto
    for style_name, font in ((STYLE_TEXTO, "arial.ttf"), ("Cota_Rua", "arial.ttf"), ("Cota_Lote", "arial.ttf")):
        try:
            if style_name not in doc.styles:
                try:
                    doc.styles.add(style_name, font=font)
                except Exception as e:
                    doc.styles.new(style_name)
                    logger.warning(f"Falha ao adicionar estilo {style_name}: {e}")
        except Exception as e:
            logger.warning(f"Erro ao garantir estilo {style_name}: {e}")

    needed_layers = {
        LAYER_SETAS_SAIDA,
        LAYER_SOLEIRA_BLOCO,
        LAYER_SOLEIRA_NUM_PAV,
        LAYER_SOLEIRA_AREA,
        LAYER_VERTICE_PTO,
        LAYER_VERTICE_TXT,
        LAYER_TABELA,
        LAYER_ORDINATE,
        LAYER_CURVA_INTER,
        LAYER_CURVA_MESTRA,
        LAYER_CURVA_ROTULO,
        # adicoes para as medicoes de via
        LAYER_VIA_MEDIDA,
        LAYER_VIA_NOME,
    }
    for ln in needed_layers:
        try:
            if ln not in doc.layers:
                doc.layers.add(ln)
        except Exception as e:
            logger.warning(f"Erro ao garantir layer {ln}: {e}")


# ---------------- Utilidades internas ----------------

def _ensure_doc_msp(path_dxf: str, doc=None, msp=None):
    if msp is not None:
        return doc, msp
    if doc is not None:
        return doc, doc.modelspace()
    doc = ezdxf.readfile(path_dxf)
    return doc, doc.modelspace()


# ---------------- Conversoes c/ bulges ----------------

def _sample_arc(center, radius: float, start_deg: float, end_deg: float, step_deg: float = 12.0):
    cx, cy = float(center.x), float(center.y)
    da = (end_deg - start_deg) % 360.0
    if da == 0.0:
        da = 360.0
    n = max(2, int(math.ceil(da / max(1e-3, step_deg))))
    pts = []
    for k in range(n + 1):
        a = math.radians(start_deg + da * (k / n))
        pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return pts


def _lwpoly_to_coords(pl, arc_step_deg: float = 10.0) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    first = True
    for ent in pl.virtual_entities():
        t = ent.dxftype()
        if t == "LINE":
            if first:
                coords.append((ent.dxf.start.x, ent.dxf.start.y))
                first = False
            coords.append((ent.dxf.end.x, ent.dxf.end.y))
        elif t == "ARC":
            if first:
                a0 = math.radians(ent.dxf.start_angle)
                sx = ent.dxf.center.x + ent.dxf.radius * math.cos(a0)
                sy = ent.dxf.center.y + ent.dxf.radius * math.sin(a0)
                coords.append((sx, sy))
                first = False
            coords.extend(
                _sample_arc(
                    ent.dxf.center,
                    ent.dxf.radius,
                    ent.dxf.start_angle,
                    ent.dxf.end_angle,
                    step_deg=arc_step_deg,
                )[1:]
            )
    if getattr(pl, "closed", False) and coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def _poly2d_to_coords(pl) -> list[tuple[float, float]]:
    pts = [(v.dxf.location.x, v.dxf.location.y) for v in pl.vertices]  # type: ignore[attr-defined]
    if getattr(pl, "is_closed", False) and pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


# ---------------- Carregadores por layer ----------------

def carregar_poligonos_por_layer(path_dxf: str, layer_name: str, *, doc=None, msp=None) -> List[Polygon]:
    doc, msp = _ensure_doc_msp(path_dxf, doc, msp)
    polys: List[Polygon] = []

    for pl in msp.query("LWPOLYLINE"):
        if pl.dxf.layer != layer_name:
            continue
        coords = _lwpoly_to_coords(pl, arc_step_deg=8.0)
        if len(coords) < 3:
            continue
        try:
            pg = Polygon(coords)
            if (not pg.is_valid) or pg.area <= 0.0:
                pg = pg.buffer(0)
            if pg.is_valid and pg.area > 1e-6:
                polys.append(pg)
        except Exception:
            continue

    for pl in msp.query("POLYLINE"):
        if pl.dxf.layer != layer_name:
            continue
        if not getattr(pl, "is_2d_polyline", False):
            continue
        coords = _poly2d_to_coords(pl)
        if len(coords) < 3:
            continue
        try:
            pg = Polygon(coords)
            if (not pg.is_valid) or pg.area <= 0.0:
                pg = pg.buffer(0)
            if pg.is_valid and pg.area > 1e-6:
                polys.append(pg)
        except Exception:
            continue

    return polys


def _carregar_linhas_por_layer(path_dxf: str, layer_name: str, *, doc=None, msp=None) -> List[LineString]:
    doc, msp = _ensure_doc_msp(path_dxf, doc, msp)
    lines: List[LineString] = []

    for e in msp.query("LINE"):
        if e.dxf.layer != layer_name:
            continue
        p1 = (e.dxf.start.x, e.dxf.start.y)
        p2 = (e.dxf.end.x, e.dxf.end.y)
        if p1 != p2:
            lines.append(LineString([p1, p2]))

    for e in msp.query("ARC"):
        if e.dxf.layer != layer_name:
            continue
        pts = _sample_arc(e.dxf.center, e.dxf.radius, e.dxf.start_angle, e.dxf.end_angle, step_deg=12.0)
        if len(pts) >= 2:
            lines.append(LineString(pts))

    for pl in msp.query("LWPOLYLINE"):
        if pl.dxf.layer != layer_name:
            continue
        coords = _lwpoly_to_coords(pl, arc_step_deg=12.0)
        if len(coords) >= 2:
            lines.append(LineString(coords))

    for pl in msp.query("POLYLINE"):
        if pl.dxf.layer != layer_name:
            continue
        if not getattr(pl, "is_2d_polyline", False):
            continue
        coords = _poly2d_to_coords(pl)
        if len(coords) >= 2:
            lines.append(LineString(coords))

    return lines


def carregar_linhas_por_layers(path_dxf: str, layers: Set[str], *, doc=None, msp=None) -> List[LineString]:
    doc, msp = _ensure_doc_msp(path_dxf, doc, msp)
    out: List[LineString] = []
    for ln in layers:
        out.extend(_carregar_linhas_por_layer(path_dxf, ln, doc=doc, msp=msp))
    return out


def carregar_camadas_dados(path_dxf_dados: str):
    doc, msp = abrir_dxf_simbologia(path_dxf_dados)
    kwargs = {"doc": doc, "msp": msp}
    lotes = carregar_poligonos_por_layer(path_dxf_dados, LAYER_LOTES, **kwargs)
    edificacoes = carregar_poligonos_por_layer(path_dxf_dados, LAYER_EDIF, **kwargs)
    perimetros = carregar_poligonos_por_layer(path_dxf_dados, LAYER_PERIMETRO, **kwargs)

    via_lines_setas = _carregar_linhas_por_layer(path_dxf_dados, LAYER_EIXO_SETAS, **kwargs)

    via_lines_geral: List[LineString] = []
    for ln in ROTATION_LAYERS:
        via_lines_geral.extend(_carregar_linhas_por_layer(path_dxf_dados, ln, **kwargs))

    return lotes, edificacoes, via_lines_setas, via_lines_geral, perimetros


__all__ = [
    "abrir_dxf_simbologia",
    "salvar_dxf",
    "garantir_estilos_blocos",
    "carregar_poligonos_por_layer",
    "carregar_linhas_por_layers",
    "carregar_camadas_dados",
]
