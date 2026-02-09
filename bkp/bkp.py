from __future__ import annotations
# Arquivo consolidado automaticamente a partir dos m?dulos em REURB


# === module: config.py ===
# config.py
from dataclasses import dataclass

# Verbosidade
VERBOSE = True

# Simbologia fixa
USE_FIXED_SIMBOLOGIA  = True
FIXED_SIMBOLOGIA_PATH = r"\\192.168.2.29\d\2304_REURB_SP\SIMBOLOGIA.dxf"

SIMBOLOGIA_DEFAULT_PATH = FIXED_SIMBOLOGIA_PATH

# ---------------- Layers de dados (DXF de entrada) ----------------
LAYER_LOTES        = "LOTE"
LAYER_EDIF         = "EDIF"
LAYER_EIXO_SETAS   = "EIXO"      # para setas de drenagem
LAYER_PERIMETRO    = "PER_INTERESSE"
LAYER_PER_LEVANTAMENTO = "PER_LEVANTAMENTO"  # recorte das curvas de nível

# Eixos de via (alias explícito)
LAYER_EIXO_VIA     = "EIXO"

# Layers com linhas que podem orientar rotação de blocos/textos
ROTATION_LAYERS = {
    "VIA", "EIXO",
}

# Bordas de pista (meio-fio). Para medições, usamos a COM_GUIA.
LAYERS_PISTA_BORDA = {"VIA"}

# ---------------- Layers de saída ----------------
LAYER_SETAS_SAIDA   = "TOP_SISTVIA"
LAYER_SOLEIRA_BLOCO = "EDIF_COTA_SOLEIRA"
LAYER_SOLEIRA_NUM_PAV = "TOP_EDIF_NUM_PAV"
LAYER_SOLEIRA_AREA  = "TOP_AREA_LOTE"

# Curvas de nível
LAYER_CURVA_INTER   = "HM_CURVA_NIV_INTERMEDIARIA"
LAYER_CURVA_MESTRA  = "HM_CURVA_NIV_MESTRA"
LAYER_CURVA_ROTULO  = "TOP_CURVA_NIV"

# Vértices/Tabela/Ordinate
LAYER_VERTICE_PTO   = "HM_VERTICES_PTO"
LAYER_VERTICE_TXT   = "TOP_VERTICE"
LAYER_TABELA        = "TOP_TABELA"
LAYER_ORDINATE      = "TOP_AZIMUTE"

# Medições de vias
LAYER_VIA_MEDIDA = "TOP_COTAS_VIARIO"
LAYER_VIA_NOME   = "TOP_SISTVIA"

# Layers REURB específicos

# ---------------- Blocos / Estilos ----------------
STYLE_TEXTO       = "Arial"
BLOCO_SETA        = "SETA_VIA"
BLOCO_SOLEIRA_POS = "SOLEIRA1"
BLOCO_VERTICE     = "VERTICE"

# ---------------- Mapeamentos (se usados em processing.py) ----------------
# ATENÇÃO: o leitor uppercasa o 'type', então use SOMENTE chaves em MAIÚSCULAS.
TYPE_TO_LAYER = {
    # iluminação / energia (exemplos seus)
    'PA': 'ELET_POSTE_ALTA_TENSAO_LUMINARIA',
    'PI': 'ELET_POSTE_ALTA_TENSAO_LUMINARIA',
    'PFI': 'ELET_POSTE_ALTA_TENSAO',
    # TUBO DE TELEFONIA / TV (exemplos seus)
    'PVTEL': 'INFRA_PVT', 'PVT': 'INFRA_PVT', 'AEPVTEL': 'INFRA_PVT',
    # PV de água pluvial
    'PVA':'INFRA_PVAP', 'PVAP':'INFRA_PVAP', 'AEPVA':'INFRA_PVAP',
    # >>> CORREÇÃO: AEPVE deve ir para INFRA_PVE (antes estava PVAP)
    'AEPVE': 'INFRA_PVE',  'ES': 'INFRA_PVE',  'PVE':'INFRA_PVE',
    'AEPVPD':'INFRA_PVE',  'PV': 'INFRA_PVE',
    # Boca de lobo (variações)
    'AEBO': 'INFRA_BOCA_LOBO', 'AEBO1': 'INFRA_BOCA_LOBO', 'AEBO2': 'INFRA_BOCA_LOBO', 'AEBO3': 'INFRA_BOCA_LOBO',
    'AEBO.1': 'INFRA_BOCA_LOBO', 'AEBO.2': 'INFRA_BOCA_LOBO',
    'BL1': 'INFRA_BOCA_LOBO', 'BL2': 'INFRA_BOCA_LOBO', 'BL3': 'INFRA_BOCA_LOBO',
    # Boca de leão (variações)
    'AEBE': 'INFRA_BOCA_LEAO', 'AEBE1': 'INFRA_BOCA_LEAO', 'AEBE2': 'INFRA_BOCA_LEAO', 'AEBE3': 'INFRA_BOCA_LEAO',
    'AEBE.1': 'INFRA_BOCA_LEAO', 'AEBE.2': 'INFRA_BOCA_LEAO', 'AEBE.3': 'INFRA_BOCA_LEAO', 'AEB': 'INFRA_BOCA_LEAO',
    # >>> NOVO: trate AEBL genérico (sem número)
    'AEBL':  'INFRA_BOCA_LOBO', 'AEBL1': 'INFRA_BOCA_LOBO', 'AEBL2': 'INFRA_BOCA_LOBO', 'AEBL3': 'INFRA_BOCA_LOBO',
    # Árvores (normalização no wrapper para 'ARVORE')
    'ARVORE': 'VEG_ARVORE_ISOLADA',
}
LAYER_TO_BLOCK = {
    'INFRA_PVE':  'INFRA_PVE',
    'INFRA_PVAP':'INFRA_PVAP',
    'INFRA_PVT': 'INFRA_PVT',
    'INFRA_BOCA_LOBO': 'BOCA_LOBO',
    'INFRA_BOCA_LEAO': 'BOCA_DE_LEAO',
    'ELET_POSTE_ALTA_TENSAO_LUMINARIA': 'POSTE_ILUMI',
    'ELET_POSTE_ALTA_TENSAO': 'POSTE_TENSAO',
    'MOB_MOBILIARIO_URBANO': 'MOB_URBANO_PT_ONIBUS',
    'INFRA_BUEIRO': 'COD119',
    'VEG_ARVORE_ISOLADA': 'ARVORE',
}


IGNORAR_SEM_LOG = {"DIVL", "VIELA"}

# ---------------- Parâmetros ----------------
@dataclass
class Params:
    # Setas de drenagem
    min_seg_len: float      = 20.0
    offset_seta: float      = 0.4
    delta_interp: float     = 0.1
    dist_busca_rot: float   = 8.0
    setas_por_trecho: int   = 2   # <<< agora este valor é respeitado no desenho
    
    # Controle inteligente de setas de drenagem
    setas_buffer_distancia: float = 5.0      # Buffer mínimo entre setas (metros)
    setas_seg_curto_max: int = 3             # Máximo de setas para segmentos curtos
    setas_seg_medio_max: int = 4             # Máximo de setas para segmentos médios
    setas_seg_longo_max: int = 5             # Máximo de setas para segmentos longos
    setas_seg_curto_threshold: float = 30.0  # Limite para segmento curto (metros)
    setas_seg_medio_threshold: float = 60.0  # Limite para segmento médio (metros)

    # Textos / Nº / Pav / Área do lote
    altura_texto_soleira: float = 0.75
    altura_texto_area: float    = 0.75
    line_spacing_factor: float  = 1.2

    # Geometria / regras
    buffer_lote_edif: float = 0.20
    max_dist_lote: float    = 5.0

    # Perímetro & anotações
    altura_texto_P: float       = 1.0
    p_label_offset_m: float   = 1.8
    p_label_offset_step: float = 0.5
    altura_texto_tabela: float  = 2.0
    tabela_cell_w: float        = 35.0
    area_table_cell_w: float    = 75.0
    tabela_cell_h: float        = 6.0
    tabela_header_h: float      = 12.0
    tabela_col_w_segmento: float  = 25.0
    tabela_col_w_distancia: float = 25.0
    tabela_col_w_azimute: float   = 25.0
    tabela_col_w_ponto: float     = 25.0
    tabela_col_w_e: float         = 25.0
    tabela_col_w_n: float         = 25.0
    tabela_offset_x: float      = 120.0
    tabela_offset_y: float      = 0.0
    dimtxt_ordinate: float      = 0.5
    dimasz_ordinate: float      = 0.2
    lote_dim_offset_m: float    = 0.60
    lote_dim_min_len_m: float   = 0.0
    lote_dim_min_spacing_m: float = 0.0
    lote_dim_snap_tol_m: float  = 0.20

    # Curvas de nível
    curva_equidist: float        = 1.0     # m
    curva_mestra_cada: int       = 5       # mestra a cada N m
    altura_texto_curva: float    = 0.75     # altura do rótulo
    curva_char_w_factor: float   = 0.60    # largura ~ h * fator * n_chars
    curva_gap_margin: float      = 0.50    # margem auxiliar (compatibilidade)
    curva_label_step_m: float    = 80.0    # distância entre rótulos na mestra (m)
    curva_min_len: float         = 10.0    # descarta linhas curtas (m)
    curva_min_area: float        = 20.0    # descarta anéis muito pequenos (m²)
    curva_smooth_sigma_px: float = 1.0     # suavização (sigma em pixels; 0 desliga)
    curva_label_offset_m: float  = 0.25    # afastamento perpendicular do texto
    curva_label_gap_enabled: bool = False  # evita cortar a curva sob o texto

    # Medições de vias
    altura_texto_via: float = 0.75
    via_offset_texto: float = 0.50
    via_cross_span: float   = 80.0
    via_dim_gap_m: float    = 0.60
    via_dim_min_len_m: float = 12.0
    via_dim_min_spacing_m: float = 25.0
    via_dim_max_por_trecho: int = 2
    via_dim_max_dist_m: float = 20.0
    via_dim_min_sep_area_m: float = 10.0
    via_dim_equal_tol_m: float = 0.05
    
    # Configurações gerais de soleira/numeração
    rotacionar_numero_casa: bool = False


# Objeto global de parâmetros (sem painel UI)
GLOBAL_PARAMS = Params()

# === module: geom_utils.py ===
# geom_utils.py
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.strtree import STRtree

def clamp(val, lo, hi): return max(lo, min(hi, val))

def encontrar_rotacao_por_via(ponto_xy, via_lines, dist_busca=8.0, delta=0.1):
    p = Point(ponto_xy); best, best_d = None, float('inf')
    for l in via_lines:
        d = l.distance(p)
        if d < best_d: best, best_d = l, d
    if best is None or best_d > dist_busca: return 0.0
    proj = best.project(p); L = best.length; proj2 = clamp(proj + delta, 0.0, L)
    a = best.interpolate(proj).coords[0]; b = best.interpolate(proj2).coords[0]
    dx, dy = (b[0]-a[0]), (b[1]-a[1])
    if dx == 0 and dy == 0: return 0.0
    return float(np.degrees(np.arctan2(dy, dx)) % 360.0)

def encontrar_rotacao_por_lote(ponto_xy, lotes, delta=0.1, raio=12.0):
    p = Point(ponto_xy); alvo, dmin = None, float("inf")
    for lp in lotes:
        d = lp.exterior.distance(p)
        if d < dmin: alvo, dmin = lp, d
    if alvo is None or dmin > raio: return None
    exterior = list(alvo.exterior.coords)
    best_seg, best_d = None, float("inf")
    for i in range(len(exterior)-1):
        a,b = exterior[i], exterior[i+1]
        seg = LineString([a,b]); d = seg.distance(p)
        if d < best_d: best_d, best_seg = d, seg
    if best_seg is None: return None
    proj = best_seg.project(p); L = best_seg.length; proj2 = clamp(proj + delta, 0.0, L)
    a = best_seg.interpolate(proj).coords[0]; b = best_seg.interpolate(proj2).coords[0]
    dx, dy = (b[0]-a[0]), (b[1]-a[1])
    if dx == 0 and dy == 0: return None
    return float(np.degrees(np.arctan2(dy, dx)) % 360.0)

def calcular_offset(p1, p2, dist=0.5):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]; L = np.hypot(dx,dy)
    if L == 0: return (0.0,0.0)
    ndx, ndy = dx/L, dy/L; nx, ny = -ndy, ndx
    return (nx*dist, ny*dist)

def segmentos_ordenados_por_proximidade(poly: Polygon, ponto_xy):
    p = Point(ponto_xy); exterior = list(poly.exterior.coords); ordenados=[]
    for i in range(len(exterior)-1):
        a,b = exterior[i], exterior[i+1]
        seg = LineString([a,b]); d = seg.distance(p)
        proj_xy = seg.interpolate(seg.project(p)).coords[0]
        ordenados.append((i, d, proj_xy, seg))
    ordenados.sort(key=lambda t: t[1])
    return ordenados

# === module: dxf_utils.py ===
# dxf_utils.py
import os
import math
from typing import List, Set
import ezdxf
from shapely.geometry import Polygon, LineString, MultiLineString

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
                except Exception:
                    doc.styles.new(style_name)
        except Exception:
            pass

    needed_layers = {
        LAYER_SETAS_SAIDA, LAYER_SOLEIRA_BLOCO, LAYER_SOLEIRA_NUM_PAV, LAYER_SOLEIRA_AREA,
        LAYER_VERTICE_PTO, LAYER_VERTICE_TXT, LAYER_TABELA, LAYER_ORDINATE,
        LAYER_CURVA_INTER, LAYER_CURVA_MESTRA, LAYER_CURVA_ROTULO,
        # >>> adições para as medições de via
        LAYER_VIA_MEDIDA, LAYER_VIA_NOME,
        # LAYER_VIA_TEXTO,  # se mantiver esse layer, pode descomentar
    }
    for ln in needed_layers:
        try:
            if ln not in doc.layers:
                doc.layers.add(ln)
        except Exception:
            pass

# ---------------- Utilidades internas ----------------
def _ensure_doc_msp(path_dxf: str, doc=None, msp=None):
    if msp is not None:
        return doc, msp
    if doc is not None:
        return doc, doc.modelspace()
    doc = ezdxf.readfile(path_dxf)
    return doc, doc.modelspace()

# ---------------- Conversões c/ bulges ----------------
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
                coords.append((ent.dxf.start.x, ent.dxf.start.y)); first = False
            coords.append((ent.dxf.end.x, ent.dxf.end.y))
        elif t == "ARC":
            if first:
                a0 = math.radians(ent.dxf.start_angle)
                sx = ent.dxf.center.x + ent.dxf.radius * math.cos(a0)
                sy = ent.dxf.center.y + ent.dxf.radius * math.sin(a0)
                coords.append((sx, sy)); first = False
            coords.extend(_sample_arc(ent.dxf.center, ent.dxf.radius,
                                      ent.dxf.start_angle, ent.dxf.end_angle,
                                      step_deg=arc_step_deg)[1:])
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
        if pl.dxf.layer != layer_name: continue
        coords = _lwpoly_to_coords(pl, arc_step_deg=8.0)
        if len(coords) < 3: continue
        try:
            pg = Polygon(coords)
            if (not pg.is_valid) or pg.area <= 0.0: pg = pg.buffer(0)
            if pg.is_valid and pg.area > 1e-6: polys.append(pg)
        except Exception:
            continue

    for pl in msp.query("POLYLINE"):
        if pl.dxf.layer != layer_name: continue
        if not getattr(pl, "is_2d_polyline", False): continue
        coords = _poly2d_to_coords(pl)
        if len(coords) < 3: continue
        try:
            pg = Polygon(coords)
            if (not pg.is_valid) or pg.area <= 0.0: pg = pg.buffer(0)
            if pg.is_valid and pg.area > 1e-6: polys.append(pg)
        except Exception:
            continue

    return polys

def _carregar_linhas_por_layer(path_dxf: str, layer_name: str, *, doc=None, msp=None) -> List[LineString]:
    doc, msp = _ensure_doc_msp(path_dxf, doc, msp)
    lines: List[LineString] = []

    for e in msp.query("LINE"):
        if e.dxf.layer != layer_name: continue
        p1 = (e.dxf.start.x, e.dxf.start.y); p2 = (e.dxf.end.x, e.dxf.end.y)
        if p1 != p2: lines.append(LineString([p1, p2]))

    for e in msp.query("ARC"):
        if e.dxf.layer != layer_name: continue
        pts = _sample_arc(e.dxf.center, e.dxf.radius, e.dxf.start_angle, e.dxf.end_angle, step_deg=12.0)
        if len(pts) >= 2: lines.append(LineString(pts))

    for pl in msp.query("LWPOLYLINE"):
        if pl.dxf.layer != layer_name: continue
        coords = _lwpoly_to_coords(pl, arc_step_deg=12.0)
        if len(coords) >= 2: lines.append(LineString(coords))

    for pl in msp.query("POLYLINE"):
        if pl.dxf.layer != layer_name: continue
        if not getattr(pl, "is_2d_polyline", False): continue
        coords = _poly2d_to_coords(pl)
        if len(coords) >= 2: lines.append(LineString(coords))

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

def _noded_lote_segments(lotes: list, snap_tol: float = 0.05) -> list[LineString]:
    if not lotes:
        return []
    try:
        boundaries = [p.boundary for p in lotes if p is not None and not p.is_empty]
        union_lines = unary_union(boundaries)
        if snap_tol:
            union_lines = snap(union_lines, union_lines, snap_tol)
        noded = unary_union(union_lines)
    except Exception:
        return []

    segs: list[LineString] = []
    def _add_segs_from_line(line: LineString):
        try:
            coords = list(line.coords)
        except Exception:
            return
        for i in range(len(coords) - 1):
            a, b = coords[i], coords[i + 1]
            if a != b:
                segs.append(LineString([a, b]))

    if hasattr(noded, "geoms"):
        for g in noded.geoms:
            if isinstance(g, LineString):
                _add_segs_from_line(g)
    elif isinstance(noded, LineString):
        _add_segs_from_line(noded)

    # dedup by rounded endpoints
    seen = set()
    unique: list[LineString] = []
    for s in segs:
        try:
            a, b = list(s.coords)[0], list(s.coords)[-1]
        except Exception:
            continue
        ax, ay = int(round(a[0] * 1000)), int(round(a[1] * 1000))
        bx, by = int(round(b[0] * 1000)), int(round(b[1] * 1000))
        key = (ax, ay, bx, by) if (ax, ay) <= (bx, by) else (bx, by, ax, ay)
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)

    # merge segmentos muito pequenos (<= 1 cm) ao segmento alinhado adjacente
    def _ang_deg(a, b) -> float:
        return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
    def _ang_diff(a, b) -> float:
        d = abs(a - b)
        return d if d <= 90.0 else 180.0 - d
    def _pt_key(p, tol):
        return (round(p[0] / tol) * tol, round(p[1] / tol) * tol)

    merge_tol = 0.01  # 1 cm
    ang_tol = 5.0
    key_tol = max(1e-3, float(snap_tol) if snap_tol else 1e-3)

    changed = True
    it = 0
    segs = list(unique)
    while changed and it < 5:
        it += 1
        changed = False
        endpoints = {}
        for idx, s in enumerate(segs):
            try:
                a, b = list(s.coords)[0], list(s.coords)[-1]
            except Exception:
                continue
            ka = _pt_key(a, key_tol)
            kb = _pt_key(b, key_tol)
            endpoints.setdefault(ka, []).append((idx, 0))
            endpoints.setdefault(kb, []).append((idx, 1))

        to_remove = set()
        to_add: list[LineString] = []

        for i, s in enumerate(segs):
            if i in to_remove:
                continue
            try:
                a, b = list(s.coords)[0], list(s.coords)[-1]
            except Exception:
                continue
            if math.hypot(b[0] - a[0], b[1] - a[1]) > merge_tol:
                continue
            ang_s = _ang_deg(a, b)
            ka = _pt_key(a, key_tol)
            kb = _pt_key(b, key_tol)

            neigh_a = [(idx, end) for (idx, end) in endpoints.get(ka, []) if idx != i and idx not in to_remove]
            neigh_b = [(idx, end) for (idx, end) in endpoints.get(kb, []) if idx != i and idx not in to_remove]

            def _neighbor_far(idx, end):
                na, nb = list(segs[idx].coords)[0], list(segs[idx].coords)[-1]
                return nb if end == 0 else na

            cand_a = []
            for idx, end in neigh_a:
                na, nb = list(segs[idx].coords)[0], list(segs[idx].coords)[-1]
                if _ang_diff(ang_s, _ang_deg(na, nb)) <= ang_tol:
                    cand_a.append((idx, end))
            cand_b = []
            for idx, end in neigh_b:
                na, nb = list(segs[idx].coords)[0], list(segs[idx].coords)[-1]
                if _ang_diff(ang_s, _ang_deg(na, nb)) <= ang_tol:
                    cand_b.append((idx, end))

            if cand_a and cand_b:
                ia, enda = cand_a[0]
                ib, endb = cand_b[0]
                fa = _neighbor_far(ia, enda)
                fb = _neighbor_far(ib, endb)
                if fa != fb:
                    to_remove.update({i, ia, ib})
                    to_add.append(LineString([fa, fb]))
                    changed = True
                    continue
            elif cand_a:
                ia, enda = cand_a[0]
                fa = _neighbor_far(ia, enda)
                if fa != b:
                    to_remove.update({i, ia})
                    to_add.append(LineString([fa, b]))
                    changed = True
                    continue
            elif cand_b:
                ib, endb = cand_b[0]
                fb = _neighbor_far(ib, endb)
                if fb != a:
                    to_remove.update({i, ib})
                    to_add.append(LineString([a, fb]))
                    changed = True
                    continue

        if changed:
            segs = [s for idx, s in enumerate(segs) if idx not in to_remove]
            segs.extend(to_add)

    return segs

import rasterio

def make_get_elevation(path_mdt):
    src = rasterio.open(path_mdt)
    transform = src.transform; arr = src.read(1); nodata = src.nodata
    def get_el(x,y):
        c,r = ~transform * (x,y); c,r = int(c),int(r)
        if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
            v=float(arr[r,c]); 
            if nodata is not None and np.isclose(v, nodata): return None
            return round(v,3)
        return None
    return src, get_el

# === module: txt_utils.py ===
# === txt_utils.py ===
import io
import re
import csv
from typing import Optional, Tuple
import pandas as pd


def _read_text(path: str) -> Tuple[str, str]:
    """Lê o arquivo como texto (tenta UTF-8 e cai para Latin-1)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), "utf-8"
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read(), "latin-1"


def _first_nonempty_line(txt: str) -> str:
    for ln in txt.splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def _detect_sep(txt: str) -> str:
    """
    Detecta o separador via csv.Sniffer + heurísticas.
    Preferência: ';', '\\t', ',', '|', espaço.
    Fallback: ';'
    """
    sample = txt[:20000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t| ")
        if getattr(dialect, "delimiter", None):
            return dialect.delimiter
    except Exception:
        pass

    first = _first_nonempty_line(sample).lower().replace(" ", "")
    if first in ("type;e;n;z", "tipo;e;n;z"):
        return ";"
    if first in ("type,e,n,z", "tipo,e,n,z"):
        return ","

    cands = [";", "\t", ",", "|", " "]
    lines = [ln for ln in sample.splitlines() if ln.strip()][:40]
    for cand in cands:
        counts = []
        for ln in lines[:12]:
            parts = re.split(r"\s+", ln.strip()) if cand == " " else ln.split(cand)
            parts = [p for p in parts if p != ""]
            counts.append(len(parts))
        if counts and len(set(counts)) == 1 and counts[0] >= 2:
            return cand

    return ";"


def ler_bruto(path_txt: str, sep: Optional[str] = None) -> pd.DataFrame:
    """
    Lê o TXT em DataFrame com colunas padronizadas:
      - **MAIÚSCULAS**: TYPE, E, N, Z (compatível com processing_wrapper)
      - cria **aliases minúsculos**: type, e, n, z
    Aceita com/sem cabeçalho; converte vírgula decimal; ignora linhas vazias.
    """
    txt, _enc = _read_text(path_txt)
    sep_eff = sep or _detect_sep(txt)
    sio = io.StringIO(txt)

    header_line = _first_nonempty_line(txt).strip().lower()

    # Heurística de cabeçalho: reconhece cabeçalhos em qualquer ordem e com sinônimos
    def _norm_colname(s: str) -> str:
        return (
            s.strip()
             .lower()
             .replace(" ", "")
             .replace("\t", "")
             .replace("-", "")
             .replace("_", "")
             .replace(".", "")
        )

    synonyms = {
        "type": {"type", "tipo", "tp", "classe", "feature", "feat"},
        "e": {"e", "x", "east", "este", "utme", "easting", "coorde", "coordx"},
        "n": {"n", "y", "north", "norte", "utmn", "northing", "coordn", "coordy"},
        "z": {"z", "alt", "altitude", "cota", "elev", "elevacao", "elevation", "h"},
    }

    # detecta separador efetivo e tokeniza 1a linha
    tokens = [t for t in re.split(r"\s+" if (sep or sep_eff) == " " else (sep or sep_eff), header_line) if t]
    tokens_norm = [_norm_colname(t) for t in tokens]
    recognized = 0
    for t in tokens_norm:
        if any(t in syns for syns in synonyms.values()):
            recognized += 1
    has_header = recognized >= 2  # ao menos 2 campos reconhecidos

    try:
        if has_header:
            df = pd.read_csv(sio, sep=sep_eff, engine="python", comment="#", skip_blank_lines=True)
            # normaliza e mapeia nomes lidos para base_cols
            cols_in = [str(c) for c in df.columns]
            mapped: dict[str, str] = {}
            for c in cols_in:
                cn = _norm_colname(c)
                target = None
                for base, syns in synonyms.items():
                    if cn in syns:
                        target = base; break
                mapped[c] = target or c
            df.rename(columns=mapped, inplace=True)
        else:
            # Detecta rapidamente o número de colunas na primeira linha de dados
            first_line = _first_nonempty_line(txt)
            parts = [p for p in (re.split(r"\s+", first_line) if sep_eff == " " else first_line.split(sep_eff)) if p != ""]
            if len(parts) >= 5:
                names = ["idx", "type", "e", "n", "z"]
            else:
                names = ["type", "e", "n", "z"]
            df = pd.read_csv(
                sio, sep=sep_eff, engine="python", header=None,
                names=names,
                comment="#", skip_blank_lines=True,
            )
    except Exception:
        # último recurso: força ';'
        sio2 = io.StringIO(txt)
        df = pd.read_csv(
            sio2, sep=";", engine="python",
            header=0 if has_header else None,
            names=None if has_header else ["type", "e", "n", "z"],
            comment="#", skip_blank_lines=True,
        )

    # Garante 4 colunas e ordem (ordem-agnóstica)
    # Descarta índice caso presente
    if "idx" in df.columns:
        try:
            idx_numeric_ratio = pd.to_numeric(df["idx"], errors="coerce").notna().mean()
            if idx_numeric_ratio > 0.8:
                df = df.drop(columns=["idx"])
        except Exception:
            df = df.drop(columns=["idx"], errors="ignore")

    base_cols = ["type", "e", "n", "z"]
    for c in base_cols:
        if c not in df.columns:
            df[c] = None
    df = df[base_cols]

    # Normalizações
    # >>> CORREÇÃO AQUI: usar .str.strip() <<<
    df["type"] = df["type"].astype(str).str.strip().str.upper()

    for col in ("e", "n", "z"):
        df[col] = (
            df[col].astype(str)
                   .str.replace(",", ".", regex=False)
                   .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Heurística: se apenas uma das colunas está na casa do milhão,
    # considere-a como Northing (N) e a outra como Easting (E).
    try:
        med_e = pd.to_numeric(df["e"], errors="coerce").median(skipna=True)
        med_n = pd.to_numeric(df["n"], errors="coerce").median(skipna=True)
        e_is_million = (med_e is not None) and pd.notna(med_e) and (float(med_e) >= 1_000_000)
        n_is_million = (med_n is not None) and pd.notna(med_n) and (float(med_n) >= 1_000_000)
        # Caso típico: se 'e' estiver na casa do milhão e 'n' não, eles vieram trocados
        if e_is_million and not n_is_million:
            df[["e", "n"]] = df[["n", "e"]].values
    except Exception:
        pass

    # Remove linhas sem coordenadas
    df = df.dropna(subset=["e", "n"]).reset_index(drop=True)

    # Padroniza nomes em MAIÚSCULO (compatível com processing_wrapper)
    df.columns = [c.upper() for c in df.columns]

    # Aliases minúsculos para compatibilidade
    for up, low in zip(["TYPE", "E", "N", "Z"], ["type", "e", "n", "z"]):
        if low not in df.columns:
            df[low] = df[up]

    return df


def ler_txt(path_txt: str) -> pd.DataFrame:
    """Função pública usada pelo run."""
    return ler_bruto(path_txt)

# === module: symbology_profiles.py ===
# === symbology_profiles.py ===
import json, os
from typing import Dict, Iterable

# chaves semanticas usadas pelo run e pelos modulos
SEM_KEYS = [
    "via_nome", "via_med",
    "per_tab", "per_vert",
    "curva_i", "curva_m", "curva_txt",
    "drenagem",
    # extras para TXT
    "txt_grande",     # no/pav/area
    "txt_soleira",    # soleira/cota
]

# defaults (padrao REURB)
DEFAULTS: Dict[str, str] = {
    "via_nome":  "TOP_SISTVIA",
    "via_med":   "TOP_COTAS_VIARIO",
    "per_tab":   "TOP_TABELA",
    "per_vert":  "HM_VERTICE",
    "curva_i":   "HM_CURVA_NIV_INTERM_LIN",
    "curva_m":   "HM_CURVA_NIV_MESTRA_LIN",
    "curva_txt": "TOP_CURVA_NIV",
    "drenagem":  "TOP_DRENAGEM",
    "txt_grande":"TOP_TEXTO",   # se o seu pipeline nao usar, e ignorado
    "txt_soleira":"TOP_COTA",
}

# perfil específico para simbologia REURB
PROFILE_REURB: Dict[str, str] = {
    "via_nome":  "TOP_SISTVIA",
    "via_med":   "TOP_COTAS_VIARIO",
    "per_tab":   "TOP_TABELA",
    "per_vert":  "HM_VERTICE",
    "curva_i":   "HM_CURVA_NIV_INTERM_LIN",
    "curva_m":   "HM_CURVA_NIV_MESTRA_LIN",
    "curva_txt": "TOP_CURVA_NIV",
    "drenagem":  "TOP_DRENAGEM",
    "txt_grande":"TOP_TEXTO",
    "txt_soleira":"TOP_COTA",
    # Layers específicos REURB
    "curva_i_reurb": "HM_CURVA_NIV_INTERM_LIN",
    "curva_m_reurb": "HM_CURVA_NIV_MESTRA_LIN",
}

# perfil fixo para a simbologia CDHU
PROFILE_CDHU: Dict[str, str] = {
    "per_tab":   "Top-Tabela",
    "per_vert":  "Top-Poligonal",
    "via_med":   "Top-Txt-Pequeno",
    "via_nome":  "Top-Txt-Grande",
    "drenagem":  "Top-Agua",
    "curva_i":   "Top-Curva1",
    "curva_m":   "Top-Curva5",
    "curva_txt": "Top-Curva5",
    "txt_grande":"Top-Txt-Grande",  # no/pav/area
    "txt_soleira":"Top-Cota",
}

def _norm(s: str) -> str:
    t = s.strip().lower()
    t = (t.replace("a","a").replace("a","a").replace("a","a").replace("a","a")
           .replace("e","e").replace("e","e").replace("i","i")
           .replace("o","o").replace("o","o").replace("o","o")
           .replace("u","u").replace("c","c"))
    return t

def load_profile_sidecar(simb_path: str) -> dict | None:
    base, _ = os.path.splitext(simb_path)
    sidecar = base + ".layers.json"
    if not os.path.isfile(sidecar):
        return None
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            data = json.load(f)
        # filtra somente chaves conhecidas, strings nao vazias
        out = {k: v for k, v in data.items() if k in SEM_KEYS and isinstance(v, str) and v.strip()}
        return out or None
    except Exception:
        return None

def build_layer_profile(doc, simb_path: str) -> Dict[str, str]:
    """
    Gera o perfil de layers:
      1) se houver sidecar JSON (<simb>.layers.json), usa-o;
      2) se for SIMBOLOGIA_CDHU, aplica PROFILE_CDHU;
      3) do contrario, usa DEFAULTS.
    """
    prof = dict(DEFAULTS)

    # 1) sidecar tem prioridade absoluta
    side = load_profile_sidecar(simb_path)
    if side:
        prof.update(side)
        return prof

    # 2) perfil fixo CDHU
    bn = os.path.basename(simb_path).lower()
    if "simbol" in bn and "cdhu" in bn:
        prof.update(PROFILE_CDHU)
        return prof
    
    # 3) perfil REURB
    if "simbol" in bn and "reurb" in bn:
        prof.update(PROFILE_REURB)
        return prof

    # 4) padrao
    return prof

# === module: shp_names.py ===
from typing import List, Tuple, Optional

from osgeo import ogr, osr
from shapely.geometry import LineString, Point
from shapely.ops import linemerge


def _pick_name_field(layer: ogr.Layer) -> Optional[str]:
    cand_subs = ("name", "nome", "logr", "logradouro", "nm_")
    def norm(s: str) -> str:
        return s.strip().lower()
    for i in range(layer.GetLayerDefn().GetFieldCount()):
        fld = layer.GetLayerDefn().GetFieldDefn(i)
        fn = norm(fld.GetName())
        if any(sub in fn for sub in cand_subs):
            return fld.GetName()
    return None


class ShpNameProvider:
    def __init__(self, lines: List[Tuple[LineString, str]]):
        self._geoms = [(ls, nm) for (ls, nm) in lines if (ls is not None and not ls.is_empty and nm)]

    def get(self, x: float, y: float, max_dist_m: float = 15.0) -> Optional[str]:
        if not self._geoms:
            return None
        p = Point(float(x), float(y))
        best_nm, best_d = None, 1e30
        for ls, nm in self._geoms:
            try:
                d = ls.distance(p)
            except Exception:
                continue
            if d < best_d:
                best_d = d; best_nm = nm
        return best_nm if (best_nm and best_d <= max_dist_m) else None

    def stats(self) -> str:
        return f"{len(self._geoms)} vias carregadas do SHP"


def build_shp_name_provider(path_shp: str, epsg_local: int) -> Optional[ShpNameProvider]:
    try:
        ds = ogr.Open(path_shp)
        if ds is None:
            return None
        lyr = ds.GetLayer(0)
        if lyr is None:
            return None
        # fonte SRS
        srs_src = lyr.GetSpatialRef()
        srs_dst = osr.SpatialReference(); srs_dst.ImportFromEPSG(int(epsg_local))
        transform = None
        if srs_src is not None and not srs_src.IsSame(srs_dst):
            transform = ogr.osr.CoordinateTransformation(srs_src, srs_dst)  # type: ignore[attr-defined]
        name_field = _pick_name_field(lyr)
        if not name_field:
            return None

        lines: List[Tuple[LineString, str]] = []
        feat = lyr.GetNextFeature()
        while feat:
            nm = feat.GetField(name_field)
            if nm:
                geom = feat.GetGeometryRef()
                if geom is not None:
                    try:
                        geom_l = geom.Clone()
                        if transform is not None:
                            geom_l.Transform(transform)
                        t = geom_l.GetGeometryType()
                        if t == ogr.wkbLineString or t == ogr.wkbLineString25D:
                            pts = [(geom_l.GetPoint(i)[0], geom_l.GetPoint(i)[1]) for i in range(geom_l.GetPointCount())]
                            if len(pts) >= 2:
                                lines.append((LineString(pts), str(nm)))
                        elif t == ogr.wkbMultiLineString:
                            for i in range(geom_l.GetGeometryCount()):
                                g = geom_l.GetGeometryRef(i)
                                pts = [(g.GetPoint(j)[0], g.GetPoint(j)[1]) for j in range(g.GetPointCount())]
                                if len(pts) >= 2:
                                    lines.append((LineString(pts), str(nm)))
                    except Exception:
                        pass
            feat = lyr.GetNextFeature()

        if not lines:
            return None

        # mescla geometrias com mesmo nome para reduzir fragmentação
        by_name: dict[str, List[LineString]] = {}
        for ls, nm in lines:
            by_name.setdefault(nm, []).append(ls)
        merged: List[Tuple[LineString, str]] = []
        for nm, lst in by_name.items():
            try:
                ml = linemerge(lst)
                if ml is None:
                    continue
                if ml.geom_type == "LineString":
                    merged.append((ml, nm))
                else:
                    for g in ml.geoms:
                        merged.append((g, nm))
            except Exception:
                for g in lst:
                    merged.append((g, nm))

        return ShpNameProvider(merged)
    except Exception:
        return None


# === module: processing.py ===
# processing.py
import re, math, numpy as np, pandas as pd
from shapely.geometry import Point, LineString

# ---- padroes classicos de soleira (mantidos) ----
RE_SOLEIRA_NUM = re.compile(r'^\s*(?P<pav>\d)\s*PV\s*(?P<num>[0-9A-Z]+)\s*$', re.IGNORECASE)
RE_SOLEIRA_SN  = re.compile(r'^\s*(?P<pav>\d)\s*PVSN\b', re.IGNORECASE)
RE_SOLEIRA_NUM_OLD = re.compile(r'^\s*(?P<pav>\d)\s*PV\s*(?P<num>[0-9A-Z]+)\s*$', re.IGNORECASE)
RE_SOLEIRA_SN_OLD  = re.compile(r'^\s*YPVSN\b', re.IGNORECASE)
RE_E_PATTERN = re.compile(r"^E\s*\d+\s*--\s*([0-9A-Za-z]+|SN)\s*--\s*([0-9O])\s*--\s*([A-Za-z]+)\s*$", flags=re.IGNORECASE)
RE_ARV_ALIAS = re.compile(r"^ARV[0-9A-Z]*$", flags=re.IGNORECASE)

def _log(msg:str):
    if VERBOSE: print(msg)

def add_centered_text(ms, content:str, x:float, y:float, height:float, style:str, layer:str, rot:float|None=None):
    t = ms.add_text(content, dxfattribs={'height':height,'style':style,'layer':layer})
    t.dxf.insert = (x,y)
    t.dxf.align_point = (x,y)
    t.dxf.halign = 1  # center
    t.dxf.valign = 2  # middle
    if rot is not None:
        t.dxf.rotation = float(rot)
    return t

def _place_mtext_middle_center(mt, x: float, y: float, rot: float | None = None):
    try:
        mt.dxf.attachment_point = 5  # Middle Center
    except Exception:
        pass
    try:
        mt.set_location((x, y, 0.0), rotation=(float(rot) if rot is not None else 0.0))
    except Exception:
        try:
            mt.dxf.insert = (x, y)
            if rot is not None:
                mt.dxf.rotation = float(rot)
        except Exception:
            pass
    return mt

def _upright(rot_deg: float) -> float:
    """Mantem texto 'em pe': se 90270, soma 180."""
    r = (rot_deg or 0.0) % 360.0
    if 90.0 < r < 270.0:
        r = (r + 180.0) % 360.0
    return r

def _normal_from_rotation(rot_deg: float) -> tuple[float, float]:
    """Normal unitaria a direcao 'rot_deg' (graus)."""
    rad = math.radians(rot_deg or 0.0)
    nx, ny = -math.sin(rad), math.cos(rad)
    nrm = math.hypot(nx, ny) or 1.0
    return (nx / nrm, ny / nrm)

def _point_in_limit(limite, x, y=None) -> bool:
    if limite is None:
        return True
    try:
        if y is None:
            if hasattr(x, 'x') and hasattr(x, 'y'):
                px, py = float(x.x), float(x.y)
            elif isinstance(x, (tuple, list)) and len(x) >= 2:
                px, py = float(x[0]), float(x[1])
            else:
                px = float(x)
                py = float(0.0)
        else:
            px, py = float(x), float(y)
        return limite.contains(Point(px, py))
    except Exception:
        return True

def _segment_vec_by_index(poly, idx_seg:int) -> tuple[float,float]:
    """Retorna vetor do segmento 'idx_seg' do anel exterior do poligono."""
    coords = list(poly.exterior.coords)
    i2 = (idx_seg + 1) % (len(coords) - 1)  # ultimo repete o 1o
    p1, p2 = coords[idx_seg], coords[i2]
    return (p2[0] - p1[0], p2[1] - p1[1])

def inserir_setas_drenagem(ms, doc, linhas_eixo, get_elevation, params: Params):
    buffer_min = float(getattr(params, "setas_buffer_distancia", 0.0))
    setas_pts: list[tuple[float, float]] = []
    for l in linhas_eixo:
        pts = list(l.coords)
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            seg_len = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            if seg_len < params.min_seg_len:
                continue
            z1, z2 = get_elevation(*p1), get_elevation(*p2)
            if (z1 is None) or (z2 is None):
                continue
            fluxo = (p2[0] - p1[0], p2[1] - p1[1]) if z2 < z1 else (p1[0] - p2[0], p1[1] - p2[1])
            ang = float(np.degrees(np.arctan2(fluxo[1], fluxo[0])))
            # quantidade de setas por segmento (reduz em trechos curtos)
            n_base = 1
            if seg_len < float(getattr(params, "setas_seg_curto_threshold", 30.0)):
                n_base = 1
                n_cap = int(getattr(params, "setas_seg_curto_max", 1))
            elif seg_len < float(getattr(params, "setas_seg_medio_threshold", 60.0)):
                n_base = 2
                n_cap = int(getattr(params, "setas_seg_medio_max", 2))
            else:
                n_base = 3
                n_cap = int(getattr(params, "setas_seg_longo_max", 3))

            n_req = int(getattr(params, "setas_por_trecho", 1))
            n_setas = max(1, min(n_req, n_cap, n_base))
            if n_setas == 1:
                fracs = [0.5]
            else:
                fracs = np.linspace(0.20, 0.80, n_setas)
            for f in fracs:
                px = p1[0] + f * (p2[0] - p1[0]); py = p1[1] + f * (p2[1] - p1[1])
                offx, offy = calcular_offset(p1, p2, dist=params.offset_seta)
                final_pt = (px + offx, py + offy)
                if buffer_min > 0.0:
                    skip = False
                    for sx, sy in setas_pts:
                        if math.hypot(final_pt[0] - sx, final_pt[1] - sy) < buffer_min:
                            skip = True
                            break
                    if skip:
                        continue
                ms.add_blockref(BLOCO_SETA, final_pt, dxfattribs={'rotation': ang, 'layer': LAYER_SETAS_SAIDA})
                if buffer_min > 0.0:
                    setas_pts.append(final_pt)

def _lote_mais_proximo(lotes, ponto, buffer_lote):
    lote_idx, lote_poly = None, None; min_d = float('inf')
    for i, lp in enumerate(lotes):
        d = 0.0 if lp.buffer(buffer_lote).contains(ponto) else lp.distance(ponto)
        if d < min_d: min_d, lote_idx, lote_poly = d, i, lp
    return lote_idx, lote_poly, min_d

def _maior_edificacao_no_lote(edificacoes, lote_poly):
    """Maior intersecao de edif com o lote (area)."""
    best = None; best_area = 0.0
    for e in edificacoes:
        try:
            inter = e.intersection(lote_poly)
            if inter.is_empty: continue
            if hasattr(inter, "geoms"):
                for g in inter.geoms:
                    a = g.area
                    if a > best_area: best_area, best = a, g
            else:
                a = inter.area
                if a > best_area: best_area, best = a, inter
        except Exception:
            continue
    return best

def processar_registros(df: pd.DataFrame, ms, doc, params: Params,
                        lotes, edificacoes, via_lines_setas, via_lines_geral, get_elevation):
    line_gap = params.altura_texto_soleira * params.line_spacing_factor
    style_texto = getattr(params, "style_texto", STYLE_TEXTO)
    type_to_layer = getattr(params, "type_to_layer", TYPE_TO_LAYER)
    layer_to_block = getattr(params, "layer_to_block", LAYER_TO_BLOCK)
    layer_num_pav = getattr(params, "layer_soleira_num_pav", LAYER_SOLEIRA_NUM_PAV)
    layer_pav = layer_num_pav
    escrever_area_lote = bool(getattr(params, "escrever_area_lote", True))
    usar_mtext_num_pav = bool(getattr(params, "soleira_num_pav_mtext", False))
    rotacionar_numero_casa = bool(getattr(params, "rotacionar_numero_casa", True))
    segmentos_usados_por_lote = {}
    lotes_com_numero = set()

    def _lote_key(idx, poly):
        if poly is None:
            return None
        try:
            return poly.wkb
        except Exception:
            return idx

    def _calc_area_offset(n_lines: int = 2) -> float:
        # Offset seguro para evitar sobreposicao entre bloco de numero/pav e texto de area.
        h_soleira = float(getattr(params, "altura_texto_soleira", 0.75))
        h_area = float(getattr(params, "altura_texto_area", 0.6))
        lg = max(float(line_gap), h_soleira)
        text_block = h_soleira + max(0, n_lines - 1) * lg
        return 0.5 * text_block + 0.5 * h_area + 0.5 * lg

    perimetro_limite = getattr(params, "perimetro_levantamento_geom", None)
    try:
        perimetro_limite = perimetro_limite.buffer(0) if perimetro_limite is not None else None
    except Exception:
        pass

    if "type" in df.columns:
        df = df.copy()
        df["type"] = df["type"].fillna("").astype(str).str.strip()

        # normaliza PVSNM para PVSN (pavimento sem numero)
        mask_pvsnm = df["type"].str.match(r"^\s*\d\s*PVSNM\s*$", flags=re.IGNORECASE)
        if mask_pvsnm.any():
            df.loc[mask_pvsnm, "type"] = df.loc[mask_pvsnm, "type"].str.replace(
                r"^\s*(\d)\s*PVSNM\s*$", r"\1 PVSN", regex=True
            )

        # normaliza padrao "1pv765c8fwm" -> "1 PV 765c8fw" (remove sufixo 'm')
        mask_pv_m = df["type"].str.match(r"^\s*\d\s*PV[0-9A-Za-z]+M\s*$", flags=re.IGNORECASE)
        if mask_pv_m.any():
            df.loc[mask_pv_m, "type"] = df.loc[mask_pv_m, "type"].str.replace(
                r"^\s*(\d)\s*PV\s*([0-9A-Za-z]+)M\s*$", r"\1 PV \2", regex=True
            )

        # normaliza padrao "1pvsn765c2c4m" -> "1 PV 765c2c4" (usa numero e remove 'm')
        mask_pvsn_num_m = df["type"].str.match(r"^\s*\d\s*PVSN[0-9A-Za-z]+M\s*$", flags=re.IGNORECASE)
        if mask_pvsn_num_m.any():
            df.loc[mask_pvsn_num_m, "type"] = df.loc[mask_pvsn_num_m, "type"].str.replace(
                r"^\s*(\d)\s*PVSN\s*([0-9A-Za-z]+)M\s*$", r"\1 PV \2", regex=True
            )

        # normaliza padrao "3pv1001aa" -> "3 PV 1001A" e ignora sufixo final "x"
        mask_pv_suf = df["type"].str.match(r"^\s*\d\s*PV\d+[A-Za-z]+X?\s*$", flags=re.IGNORECASE)
        if mask_pv_suf.any():
            def _pv_suf_norm(val: str) -> str:
                m = re.match(r"^\s*(\d)\s*PV(\d+)([A-Za-z]+)X?\s*$", val.strip(), flags=re.IGNORECASE)
                if not m:
                    return val
                pav, num, suf = m.group(1), m.group(2), m.group(3)
                suf_ch = suf[0].upper() if (suf and len(suf) >= 2) else ""
                return f"{pav} PV {num}{suf_ch}"
            df.loc[mask_pv_suf, "type"] = df.loc[mask_pv_suf, "type"].map(_pv_suf_norm)

        def _map_e_pattern(s: str) -> str:
            m = RE_E_PATTERN.match(s.strip())
            if not m:
                return s
            g1, g2, g3 = m.group(1).upper(), m.group(2).upper(), m.group(3).upper()
            if g3 == "TV":
                return "TV"
            try:
                pav = int(g2)
            except Exception:
                pav = 1
            if g1 == "SN":
                return f"{pav} PVSN"
            return f"{pav} PV {g1}"

        df["type"] = df["type"].map(_map_e_pattern)

        mask_arv = df["type"].str.match(RE_ARV_ALIAS)
        if mask_arv.any():
            df.loc[mask_arv, "type"] = "ARVORE"

    for _, r in df.iterrows():
        tipo_raw = str(r['type']).strip()
        tipo = tipo_raw.upper()
        x, y = float(r['E']), float(r['N'])
        z_mdt = get_elevation(x, y)

        # --- TV (mantido) ---
        if tipo == 'TV':
            p = Point(x,y); lote_idx, lote_poly, min_d = _lote_mais_proximo(lotes, p, params.buffer_lote_edif)
            alvo_pt = (x,y)
            if lote_poly:
                edif_sel = _maior_edificacao_no_lote(edificacoes, lote_poly)
                if edif_sel is not None: alvo_pt = edif_sel.representative_point().coords[0]
                else: alvo_pt = lote_poly.representative_point().coords[0]
            add_centered_text(ms, "TV", alvo_pt[0], alvo_pt[1], params.altura_texto_soleira, style_texto, LAYER_SOLEIRA_NUM_PAV)
            continue

        # --- Soleiras ---
        pav=None; sn=False
        m_sn = RE_SOLEIRA_SN.search(tipo)
        if m_sn: pav=int(m_sn.group('pav')); sn=True
        elif RE_SOLEIRA_SN_OLD.search(tipo): pav=1; sn=True

        m_num = RE_SOLEIRA_NUM.search(tipo) or RE_SOLEIRA_NUM_OLD.search(tipo)
        is_soleira = bool(sn or m_num)
        if is_soleira:
            if pav is None:
                # 'pav' so existe no RE_SOLEIRA_NUM (nao no OLD), trata default=1
                try:
                    pav = int(m_num.group('pav'))
                except Exception:
                    pav = 1

            # monta numero/sufixo com seguranca
            if sn:
                numero_str = None
            else:
                gi = getattr(m_num.re, 'groupindex', {})
                num_val = m_num.group('num') if ('num' in gi) else ''
                suf_val = m_num.group('suf') if ('suf' in gi) else ''
                num_val = (num_val or '').strip()
                suf_val = (suf_val or '').strip()
                numero_str = (f"{num_val}{suf_val}".upper() if (num_val or suf_val) else None)

            p_txt = Point(x,y)
            lote_idx, lote_poly, min_d = _lote_mais_proximo(lotes, p_txt, params.buffer_lote_edif)
            lote_key = _lote_key(lote_idx, lote_poly)
            if lote_key is not None and lote_key in lotes_com_numero:
                # procura outro lote disponivel
                candidatos = []
                for j, lp in enumerate(lotes):
                    key_j = _lote_key(j, lp)
                    if key_j is not None and key_j in lotes_com_numero:
                        continue
                    try:
                        d = 0.0 if lp.buffer(params.buffer_lote_edif).contains(p_txt) else lp.distance(p_txt)
                    except Exception:
                        continue
                    candidatos.append((d, j, lp))
                candidatos.sort(key=lambda t: t[0])
                lote_idx = lote_poly = None
                min_d = float("inf")
                for d, j, lp in candidatos:
                    lote_idx, lote_poly, min_d = j, lp, d
                    break
                lote_key = _lote_key(lote_idx, lote_poly)
            numero_label = "S/N" if numero_str is None else f"N {numero_str}"
            pav_label = f"{pav} PV"
            if (lote_poly is None) or (min_d > params.max_dist_lote):
                ms.add_blockref(BLOCO_SOLEIRA_POS, (x, y), dxfattribs={'rotation': 0.0, 'layer': LAYER_SOLEIRA_BLOCO})
                base_rot = 0.0
                permitir_texto = _point_in_limit(perimetro_limite, x, y)
                if not permitir_texto:
                    continue
                if usar_mtext_num_pav and layer_num_pav:
                    mt = ms.add_mtext('\\P'.join([numero_label, pav_label]), dxfattribs={'layer': layer_num_pav, 'style': style_texto})
                    try:
                        mt.dxf.char_height = params.altura_texto_soleira
                    except Exception:
                        pass
                    _place_mtext_middle_center(mt, x, y, (base_rot if rotacionar_numero_casa else 0.0))
                else:
                    rot_text = base_rot if rotacionar_numero_casa else 0.0
                    add_centered_text(ms, numero_label,
                                      x, y + 0.5*line_gap, params.altura_texto_soleira, style_texto, layer_num_pav, rot_text)
                    add_centered_text(ms, pav_label,
                                      x, y - 0.5*line_gap, params.altura_texto_soleira, style_texto, layer_pav, rot_text)
                continue
            if lote_key is not None:
                lotes_com_numero.add(lote_key)

            # 1) escolhe segmento de testada e projeta: SOLEIRA EXATA NA LINHA
            ordenados = segmentos_ordenados_por_proximidade(lote_poly, (x,y))
            usados = segmentos_usados_por_lote.setdefault(lote_idx, set())
            pos_soleira=(x,y); seg_escolhido=None
            for idx_seg, _, proj_xy, _ in ordenados:
                if idx_seg not in usados:
                    seg_escolhido=idx_seg; pos_soleira=proj_xy; break
            if seg_escolhido is None and ordenados:
                seg_escolhido, _, pos_soleira, _ = ordenados[0]
            usados.add(seg_escolhido)

            # bloco de soleira exatamente na borda
            ms.add_blockref(BLOCO_SOLEIRA_POS, pos_soleira, dxfattribs={'rotation':0.0,'layer':LAYER_SOLEIRA_BLOCO})

            # 2) rotacao para textos: via proxima (fallback: tangente da testada), com upright
            rot_txt = encontrar_rotacao_por_via(pos_soleira, via_lines_geral, params.dist_busca_rot, params.delta_interp)
            if not rot_txt:
                vx, vy = _segment_vec_by_index(lote_poly, seg_escolhido)
                rot_txt = math.degrees(math.atan2(vy, vx))
            rot_txt = _upright(rot_txt)

            # 3) ponto dos textos: dentro da MAIOR EDIFICACAO; senao centro do lote
            e_sel = _maior_edificacao_no_lote(edificacoes, lote_poly)
            if e_sel is not None:
                tx_pt = e_sel.representative_point().coords[0]
            else:
                tx_pt = lote_poly.representative_point().coords[0]
            base_x, base_y = float(tx_pt[0]), float(tx_pt[1])

            permitir_texto = (_point_in_limit(perimetro_limite, pos_soleira)
                               and _point_in_limit(perimetro_limite, base_x, base_y))
            if not permitir_texto:
                continue

            # 4) escreve alinhado e com ordem: Numero (topo), Pav (meio), Area (embaixo)
            #    Mantem centralizado independente do comprimento do texto.
            rot_text = rot_txt if rotacionar_numero_casa else 0.0
            nux, nuy = _normal_from_rotation(rot_text)

            if usar_mtext_num_pav and layer_num_pav:
                mt = ms.add_mtext('\\P'.join([numero_label, pav_label]), dxfattribs={'layer': layer_num_pav, 'style': style_texto})
                try:
                    mt.dxf.char_height = params.altura_texto_soleira
                except Exception:
                    pass
                _place_mtext_middle_center(mt, base_x, base_y, (rot_txt if rotacionar_numero_casa else 0.0))
                # Area embaixo de tudo, com margem segura
                if escrever_area_lote:
                    area_offset = _calc_area_offset(2)
                    add_centered_text(
                        ms, f"{lote_poly.area:.2f} m2",
                        base_x + nux * -area_offset, base_y + nuy * -area_offset,
                        params.altura_texto_area, style_texto, LAYER_SOLEIRA_AREA, rot_text
                    )
            else:
                # Numero (topo)
                add_centered_text(
                    ms, numero_label,
                    base_x + nux * +1.0 * line_gap, base_y + nuy * +1.0 * line_gap,
                    params.altura_texto_soleira, style_texto, layer_num_pav, rot_text
                )
                # Pav (meio)
                add_centered_text(
                    ms, pav_label,
                    base_x, base_y,
                    params.altura_texto_soleira, style_texto, layer_pav, rot_text
                )
                # Area (embaixo)
                if escrever_area_lote:
                    area_offset = _calc_area_offset(2)
                    add_centered_text(
                        ms, f"{lote_poly.area:.2f} m2",
                        base_x + nux * -area_offset, base_y + nuy * -area_offset,
                        params.altura_texto_area, style_texto, LAYER_SOLEIRA_AREA, rot_text
                    )
            continue

        # --- Demais blocos (infra etc.) ---
        associada = type_to_layer.get(tipo)
        if not associada: continue
        bloco = layer_to_block.get(associada)
        if not bloco: continue

        rot = encontrar_rotacao_por_via((x,y), via_lines_geral, params.dist_busca_rot, params.delta_interp)
        if not rot:
            rot_fb = encontrar_rotacao_por_lote((x,y), lotes, params.delta_interp, raio=max(params.dist_busca_rot*2, 8.0))
            if rot_fb is not None: rot = rot_fb
        rot = _upright(rot or 0.0)

        if bloco in {"INFRA_PVE", "INFRA_PVAP"}:
            rot = 0.0
        if "BOCA" in bloco:
            import re as _re
            try: qtd = int(_re.findall(r'(\d+)$', tipo)[-1])
            except: qtd = 1
            dx = 1.0*math.cos(math.radians(rot)); dy = 1.0*math.sin(math.radians(rot))
            for i in range(qtd):
                xi, yi = x + i*dx, y + i*dy
                ms.add_blockref(bloco, (xi, yi), dxfattribs={'rotation': rot, 'layer': associada})
        else:
            ms.add_blockref(bloco, (x, y), dxfattribs={'rotation': rot, 'layer': associada})

        if bloco == 'INFRA_PVE':
            add_centered_text(ms, "PVE", x, y+0.7, 0.4, style_texto, associada, rot)
        elif bloco == 'INFRA_PVAP':
            add_centered_text(ms, "PVAP", x, y+0.7, 0.4, style_texto, associada, rot)

        if z_mdt is not None:
            add_centered_text(
                ms, f"{float(z_mdt):.3f}",
                x, y - 1.0, 0.4, style_texto, associada, rot
            )

    # setas de drenagem (mesma regra de antes)
    inserir_setas_drenagem(ms, doc, via_lines_setas, get_elevation, params)

# === module: perimetro.py ===
# perimetro.py
import math
from typing import List, Tuple
import numpy as np
from shapely.geometry import Polygon, Point

def _vertices_do_poligono(poly: Polygon) -> List[Tuple[float,float]]:
    """Mesma lA3gica do Colab: usa a sequAancia do contorno EXTERNO invertida
    e remove o Aoltimo vArtice duplicado."""
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
    # normaliza segundos=60
    if s == 60: s=0; m+=1
    if m == 60: m=0; d=(d+1)%360
    deg = "\u00B0"  # símbolo de grau
    return f"{d:02d}{deg}{m:02d}'{s:02d}''"

def _dist(p1, p2) -> float:
    return float(math.hypot(p2[0]-p1[0], p2[1]-p1[1]))

def _azimute(p1, p2) -> float:
    dE = p2[0]-p1[0]; dN = p2[1]-p1[1]
    ang = math.degrees(math.atan2(dE, dN))
    if ang < 0: ang += 360.0
    return float(ang)

def _bbox_max(vertices):
    xs=[p[0] for p in vertices]; ys=[p[1] for p in vertices]
    return max(xs), max(ys)

def _add_text_center(ms, text, cx, cy, h, layer, style=STYLE_TEXTO):
    t = ms.add_text(text, dxfattribs={'height':h, 'style':style, 'layer':layer})
    t.dxf.insert=(cx,cy); t.dxf.align_point=(cx,cy); t.dxf.halign=1; t.dxf.valign=2
    return t

def _draw_cell(ms, x, y, w, h, layer):
    ms.add_lwpolyline([(x,y),(x+w,y),(x+w,y-h),(x,y-h),(x,y)], close=True, dxfattribs={'layer': layer})

def _table_header(ms, x0, y0, col_ws, header_h, ch, txt_h, layer, style):
    # Linha 1: Segmento | Distancia | Azimute | Ponto | [Coordenadas (E+N)]
    headers = ["Segmento", "Distância (m)", "Azimute", "Ponto"]
    x = x0
    for i, htxt in enumerate(headers):
        w = col_ws[i]
        # cabecalho com altura de header_h (mesma soma de Coordenadas+E/N)
        _draw_cell(ms, x, y0, w, header_h, layer)
        _add_text_center(ms, htxt, x + w/2, y0 - header_h/2, txt_h, layer, style)
        x += w
    w_e = col_ws[4]
    w_n = col_ws[5]
    coord_y0 = y0
    _draw_cell(ms, x, coord_y0, w_e + w_n, ch, layer)
    _add_text_center(ms, "Coordenadas", x + (w_e + w_n) / 2, coord_y0 - ch/2, txt_h, layer, style)
    y1 = y0 - ch
    _draw_cell(ms, x, y1, w_e, ch, layer); _add_text_center(ms, "E", x + w_e/2, y1 - ch/2, txt_h, layer, style)
    _draw_cell(ms, x + w_e, y1, w_n, ch, layer); _add_text_center(ms, "N", x + w_e + w_n/2, y1 - ch/2, txt_h, layer, style)


def _table_row(ms, row_idx, x0, y0, col_ws, header_h, ch, txt_h, layer, style, seg, dist, az, pid, e, n):
    # y de inicio da linha deste conteudo (após cabeçalhos)
    y = y0 - header_h - (row_idx * ch)
    cols = [seg, f"{dist:.2f}", _dms_str(az), pid, f"{e:.4f}", f"{n:.4f}"]
    x = x0
    for i, val in enumerate(cols):
        w = col_ws[i]
        _draw_cell(ms, x, y, w, ch, layer)
        _add_text_center(ms, val, x + w/2, y - ch/2, txt_h, layer, style)
        x += w


def _add_ordinate(ms, midpoint, texto, layer, params: Params, style: str):
    dim = ms.add_ordinate_dim(midpoint, (0.5,-1.0), 0,
                              origin=midpoint, text=texto, rotation=0,
                              dxfattribs={'layer': layer},
                              override={"dimtxt": params.dimtxt_ordinate,
                                        "dimasz": params.dimasz_ordinate,
                                        "dimtsz": 0, "dimtad": 0,
                                        "dimtxsty": style})
    dim.render()

def processar_lotes_dimensoes(ms, doc, params: Params, lotes: list):
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
    ang_tol = 5.0
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
            dx = b[0] - a[0]; dy = b[1] - a[1]
            ang = math.degrees(math.atan2(dy, dx))
            L = math.hypot(dx, dy)
            ux, uy = dx / L, dy / L
            t1 = a[0] * ux + a[1] * uy
            t2 = b[0] * ux + b[1] * uy
            tmin, tmax = (t1, t2) if t1 <= t2 else (t2, t1)
            idx = len(seg_meta)
            seg_meta.append({"a": a, "b": b, "len": seg_len, "mx": mx, "my": my,
                             "ang": ang, "ux": ux, "uy": uy, "tmin": tmin, "tmax": tmax})
            ax, ay = int(round(a[0] * 1000)), int(round(a[1] * 1000))
            bx, by = int(round(b[0] * 1000)), int(round(b[1] * 1000))
            key = (ax, ay, bx, by) if (ax, ay) <= (bx, by) else (bx, by, ax, ay)
            seg_key_map[key] = idx
        except Exception:
            continue

    def _ang_diff_180(a: float, b: float) -> float:
        d = abs((a - b) % 180.0)
        return d if d <= 90.0 else 180.0 - d

    # remove large segments when smaller segments cover it
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
            dxm = sj["mx"] - si["mx"]; dym = sj["my"] - si["my"]
            dist_line = abs(dxm * (-si["uy"]) + dym * (si["ux"]))
            if dist_line > line_tol:
                continue
            # projeta sj no eixo de si (direcao do segmento maior)
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
        except Exception:
            continue
        seg_len = float(math.hypot(b[0] - a[0], b[1] - a[1]))
        # evita cotas que viram 0.00 na impressao (dimdec=2)
        if seg_len < min_len or round(seg_len, 2) == 0.0:
            continue
        mid = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        # remove duplicadas proximas com mesmo valor
        dup = False
        for mx, my, mlen in kept_mid_pts:
            if abs(mlen - seg_len) <= dim_val_tol:
                dx = mid[0] - mx; dy = mid[1] - my
                if (dx * dx + dy * dy) <= dim_close_tol * dim_close_tol:
                    dup = True
                    break
        if dup:
            continue
        # skip large segment covered by smaller segments
        if seg_meta:
            ax, ay = int(round(a[0] * 1000)), int(round(a[1] * 1000))
            bx, by = int(round(b[0] * 1000)), int(round(b[1] * 1000))
            key = (ax, ay, bx, by) if (ax, ay) <= (bx, by) else (bx, by, ax, ay)
            idx = seg_key_map.get(key)
            if idx is not None and idx in skip_idx:
                continue
        if min_spacing2 > 0.0:
            too_close = False
            for px, py in placed_pts:
                dx = mid[0] - px; dy = mid[1] - py
                if (dx * dx + dy * dy) < min_spacing2:
                    too_close = True
                    break
            if too_close:
                continue

        try:
            _add_dim_aligned(
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
        except Exception:
            pass

def processar_perimetros(ms, doc, params: Params, perimetros: list, perimetro_limite=None):
    """Usa o primeiro polAgono de PER_INTERESSE."""
    if not perimetros: return
    poly = perimetros[0]
    style_texto = getattr(params, "style_texto", STYLE_TEXTO)
    V = _vertices_do_poligono(poly)  # P1..Pn

    # 1) Inserir VERTICE e rA3tulos Pn
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
        # Caixa aproximada do texto, centrada em (cx, cy)
        w = max(0.01, len(str(text)) * float(h) * float(w_factor))
        hh = float(h)
        return (cx - w / 2.0, cy - hh / 2.0, cx + w / 2.0, cy + hh / 2.0)

    def _boxes_overlap(a, b) -> bool:
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

    def _find_label_pos(px, py, dir_x, dir_y, text, h):
        # tenta manter perto do vertice, sem sobrepor outros textos
        perp_x, perp_y = -dir_y, dir_x
        offsets = [0.0, step_m, -step_m, 2*step_m, -2*step_m]
        dist = offset_m
        while dist <= max_dist + 1e-6:
            for lat in offsets:
                cx = px + dir_x * dist + perp_x * lat
                cy = py + dir_y * dist + perp_y * lat
                if math.hypot(cx - px, cy - py) > max_dist + 1e-6:
                    continue
                box = _text_box(cx, cy, text, h)
                if any(_boxes_overlap(box, b) for b in placed_boxes):
                    continue
                placed_boxes.append(box)
                return cx, cy
            dist += step_m
        # fallback: aceita o offset minimo
        cx = px + dir_x * offset_m
        cy = py + dir_y * offset_m
        placed_boxes.append(_text_box(cx, cy, text, h))
        return cx, cy

    for i, (x, y) in enumerate(V, start=1):
        ms.add_blockref(BLOCO_VERTICE, (x, y), dxfattribs={'layer': LAYER_VERTICE_PTO})
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
            dist = offset_m
            candidate = Point(px + dir_x * dist, py + dir_y * dist)
            tries = 0
            while boundary_geom is not None and (boundary_geom.contains(candidate) or boundary_geom.distance(candidate) < 0.05) and tries < 40:
                dist += step_m
                if dist > max_dist:
                    break
                candidate = Point(px + dir_x * dist, py + dir_y * dist)
                tries += 1
            px, py = candidate.x, candidate.y
            label_x, label_y = _find_label_pos(px, py, dir_x, dir_y, f"P{i}", params.altura_texto_P)
        else:
            dir_x, dir_y = 1.0, 0.0
            label_x, label_y = _find_label_pos(px, py, dir_x, dir_y, f"P{i}", params.altura_texto_P)
        _add_text_center(ms, f"P{i}", label_x, label_y, params.altura_texto_P, LAYER_VERTICE_TXT, style_texto)

    # 2) Tabela (Segmento / DistAncia / Azimute / Ponto / E / N)
    max_x, max_y = _bbox_max(V)
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

    _table_header(ms, x0, y0, col_ws, header_h, ch, th, LAYER_TABELA, style_texto)

    n = len(V)
    for i in range(n):
        a = V[i]; b = V[(i+1) % n]
        seg = f"{i+1} - {((i+1)%n)+1}"
        dist = _dist(a,b)
        az = _azimute(a,b)
        pid = f"P{i+1}"
        e, ncoord = a[0], a[1]
        _table_row(ms, i, x0, y0, col_ws, header_h, ch, th, LAYER_TABELA, style_texto, seg, dist, az, pid, e, ncoord)

    # Passa âncora onde a tabela de áreas deverá começar (logo abaixo da tabela)
    try:
        setattr(params, "area_table_anchor", (x0, y0 - (2 + n) * ch - ch))
    except Exception:
        pass
    # 3) Ordinate dimension no meio de cada segmento (texto adist / Az. XXAYY'ZZ''a empilhado)
    for i in range(n):
        a = V[i]; b = V[(i+1)%n]
        mid = (0.5*(a[0]+b[0]), 0.5*(a[1]+b[1]))
        dist = _dist(a,b)
        az = _azimute(a,b)
        texto = f"\\S{dist:.2f}/ Az. {_dms_str(az)};"
        _add_ordinate(ms, mid, texto, LAYER_ORDINATE, params, style_texto)



# === module: curvas_nivel.py ===
# === curvas_nivel.py ===
import math
from typing import Optional, Iterable, Tuple, List

import ezdxf
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import substring
from osgeo import gdal, ogr

try:
    CFG_STYLE_TEXTO = STYLE_TEXTO
except Exception:
    CFG_STYLE_TEXTO = "SIMPLEX"  # padrao

# Fallback names (serao normalmente substituidos em run pelo perfil de simbologia)
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
    """Aceita caminho ou dataset e devolve GDAL Dataset."""
    if isinstance(mdt_src, gdal.Dataset):
        return mdt_src
    try:
        # talvez seja um path string
        return gdal.Open(mdt_src)
    except Exception:
        return mdt_src  # torce para ja ser um dataset compativel

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
    # mantem texto em pe:
    while ang <= -180: ang += 360
    while ang > 180: ang -= 360
    if ang > 90: ang -= 180
    if ang <= -90: ang += 180
    return ang

def _calculate_text_position_outside(ls: LineString, dist: float, offset_m: float = 2.0) -> Tuple[float, float, float]:
    """
    Calcula posição do texto fora da curva (modo REURB).
    Retorna (x, y, angle_deg) com offset perpendicular à curva.
    """
    # Ponto na curva
    pt = ls.interpolate(dist)
    
    # Ângulo da tangente
    angle_deg = _tangent_angle_deg(ls, dist)
    angle_rad = math.radians(angle_deg)
    
    # Vetor perpendicular (normal) apontando para fora
    # Rotaciona 90 graus no sentido anti-horário
    perp_x = -math.sin(angle_rad)
    perp_y = math.cos(angle_rad)
    
    # Posição final com offset
    final_x = pt.x + perp_x * offset_m
    final_y = pt.y + perp_y * offset_m
    
    return final_x, final_y, angle_deg

def _format_elev(v: float, prec: int = 1) -> str:
    # 1 casa decimal e tipico
    s = f"{v:.{prec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s

def _text_len_m(content: str, h: float, char_w_factor: float) -> float:
    return max(0.01, len(content) * h * float(char_w_factor))

def _cut_line_by_gaps(ls: LineString, gaps: List[Tuple[float, float]]) -> List[LineString]:
    """Corta a LineString removendo intervalos [s0, s1] (em metros ao longo da linha)."""
    if not gaps:
        return [ls]
    L = float(ls.length)
    # normaliza e ordena
    norm = []
    for s0, s1 in gaps:
        a, b = sorted((max(0.0, s0), min(L, s1)))
        if b - a > 1e-6:
            norm.append((a, b))
    if not norm:
        return [ls]
    norm.sort()
    # funde gaps sobrepostos
    merged = []
    ca, cb = norm[0]
    for a, b in norm[1:]:
        if a <= cb + 1e-6:
            cb = max(cb, b)
        else:
            merged.append((ca, cb))
            ca, cb = a, b
    merged.append((ca, cb))
    # cria segmentos restantes
    segs = []
    start = 0.0
    for a, b in merged:
        if a - start > 1e-6:
            segs.append(substring(ls, start, a))
        start = b
    if L - start > 1e-6:
        segs.append(substring(ls, start, L))
    # shapely.substring pode retornar GeometryCollection/LineString
    out: List[LineString] = []
    for s in segs:
        if s.is_empty: 
            continue
        if s.geom_type == "LineString":
            out.append(s)  # type: ignore
        elif s.geom_type == "MultiLineString":
            out.extend([g for g in s.geoms if g.length > 0])
    return out

def gerar_curvas_nivel(ms,
                       mdt_src,
                       params,
                       clip_poly: Optional[Polygon] = None):
    """
    Gera curvas de nivel (intermediaria/mestra) e rotulos.
    Regras pedidas:
      - rotular **apenas** as curvas mestras (por padrao);
      - cortar a linha no local do numero (abertura sob o texto).
    """
    eq = float(getattr(params, "curva_equidist", 1.0))
    mestra_cada = int(getattr(params, "curva_mestra_cada", 5))
    min_len = float(getattr(params, "curva_min_len", 10.0))
    min_area = float(getattr(params, "curva_min_area", 20.0))  # nao usado aqui (aplica-se a poligonos)
    h_text = float(getattr(params, "altura_texto_curva", 0.4))
    char_w_factor = float(getattr(params, "curva_char_w_factor", 0.6))
    gap_margin = float(getattr(params, "curva_gap_margin", 0.5))
    step_m = float(getattr(params, "curva_label_step_m", 80.0))
    ends_only = bool(getattr(params, "curva_label_ends_only", False))
    only_master_labels = bool(getattr(params, "curva_label_only_master", True))
    label_precision = int(getattr(params, "curva_label_precision", 1))
    label_offset_m = max(0.0, float(getattr(params, "curva_label_offset_m", 0.25)))
    label_gap_enabled = bool(getattr(params, "curva_label_gap_enabled", False))

    style_name = getattr(params, "style_texto", CFG_STYLE_TEXTO) or "SIMPLEX"
    layer_curva_i = getattr(params, "layer_curvas", DEF_LYR_CURVA_I)
    layer_curva_m = getattr(params, "layer_curvas_mestra", DEF_LYR_CURVA_M)
    layer_txt = getattr(params, "layer_curvas_txt", DEF_LYR_CURVA_TXT)

    doc = ms.doc
    _ensure_text_style(doc, style_name)
    for lyr in (layer_curva_i, layer_curva_m, layer_txt):
        _ensure(doc, lyr)

    # GDAL contours
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
            # fallback: polyline 2D
            pl = ms.add_polyline2d(coords, dxfattribs={"layer": layer})
        # Define a elevação da curva no próprio entity, quando possível
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
            t.dxf.halign = 1  # center
            t.dxf.valign = 2  # middle
        except Exception:
            pass
    
    feat = lyr.GetNextFeature()
    while feat:
        elev = float(feat.GetFieldAsDouble(elev_field_index))
        geom = feat.GetGeometryRef()
        lines = _ogr_lines_to_shapely(geom)
        if clip_poly is not None:
            lines = _clip_lines_to_poly(lines, clip_poly)
        # tipo: mestra ou intermediaria?
        idx = int(round(elev / eq))
        is_master = (idx % mestra_cada == 0)

        for ls in lines:
            if ls.length < min_len:
                continue

            # posicoes de rotulo (se for mestra; ou tambem intermediaria caso only_master_labels=False)
            label_positions: List[float] = []
            if is_master or not only_master_labels:
                if ends_only:
                    if ls.length <= 0.01:
                        label_positions = [0.0]
                    else:
                        label_positions = [0.0, float(ls.length)]
                else:
                    # posicione ao longo da curva a cada step_m
                    nlab = max(1, int(ls.length // max(step_m, 1.0)))
                    for i in range(nlab):
                        s = (i + 0.5) * (ls.length / nlab)
                        label_positions.append(s)

            # cortes sob o texto apenas quando habilitado e não for modo extremidades
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

            # desenha os textos
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

# === module: vias_medidas.py ===
# === vias_medidas.py ===
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
                          dim_gap_m: float = 0.60,                  # separacao entre cotas para nao sobrepor
                          dim_min_len_m: float = 12.0,              # comprimento minimo do trecho para cotar
                          dim_min_spacing_m: float = 25.0,          # espacamento minimo entre cotas no mesmo trecho
                          dim_max_por_trecho: int = 2,              # maximo de cotas por trecho
                          dim_max_dist_m: float = 20.0,             # distancia maxima para cotar
                          dim_min_sep_area_m: float = 10.0,         # raio minimo para evitar cotas muito proximas
                          dim_equal_tol_m: float = 0.05,            # tolerancia para considerar cotas iguais
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
        if eixo_full.length >= 5.0:
            # escolhe o maior segmento do eixo para posicionar o nome
            try:
                coords = list(eixo_full.coords)
                best_len = -1.0
                best_seg = None
                for i in range(len(coords) - 1):
                    a = coords[i]; b = coords[i + 1]
                    seg_len = float(math.hypot(b[0] - a[0], b[1] - a[1]))
                    if seg_len > best_len:
                        best_len = seg_len
                        best_seg = (a, b)
                if best_seg is not None:
                    mid_all = Point((best_seg[0][0] + best_seg[1][0]) * 0.5,
                                    (best_seg[0][1] + best_seg[1][1]) * 0.5)
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

            if 'best_seg' in locals() and best_seg is not None:
                ang_nome = math.degrees(math.atan2(best_seg[1][1] - best_seg[0][1], best_seg[1][0] - best_seg[0][0]))
            else:
                ang_nome = _tangent_angle_at_frac(eixo_full, 0.5, delta_norm=0.02)
            ang_rad = math.radians(ang_nome)
            # tangente e normal
            tx, ty = math.cos(ang_rad), math.sin(ang_rad)
            nx_nom, ny_nom = -math.sin(ang_rad), math.cos(ang_rad)

            # lado do nome
            if nome_side_mode == "oposto_dim":
                side = +1.0  # dimensoes usam -1.0 => nome no lado oposto
            else:
                # modo "auto": heuristica com lotes/usuario
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

            # aplica tambem deslocamento AO LONGO do eixo
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
                        dx = mx - px; dy = my - py
                        if (dx * dx + dy * dy) < min_sep2:
                            too_close = True
                            break
                    if too_close:
                        continue
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

                gap = max(0.1, float(dim_gap_m))
                off_gg = float(offset_texto_m)
                off_gl = float(offset_texto_m) + gap
                off_ll = float(offset_texto_m) + 2.0 * gap + float(offset_lote_lote_extra_m)

                max_dist = float(dim_max_dist_m)
                dist_gg = float(Point(g_neg).distance(Point(g_pos)))
                eq_tol = float(dim_equal_tol_m)
                if dist_gg > max_dist:
                    continue

                # a) GuiaaGuia
                _add_dim_aligned(ms, g_neg, g_pos, -1.0, off_gg,
                                 layer_via_medida, float(texto_altura),
                                 arrow_size_factor * float(texto_altura), 2, dim_text_style)

                # b) GuiaaLote
                if l_neg is not None:
                    dist_gl_neg = float(Point(g_neg).distance(Point(l_neg)))
                    if dist_gl_neg <= max_dist:
                        _add_dim_aligned(ms, g_neg, l_neg, -1.0, off_gl,
                                         layer_via_medida, float(texto_altura),
                                         arrow_size_factor * float(texto_altura), 2, dim_text_style)
                if l_pos is not None:
                    dist_gl_pos = float(Point(g_pos).distance(Point(l_pos)))
                    if dist_gl_pos <= max_dist:
                        _add_dim_aligned(ms, g_pos, l_pos, -1.0, off_gl,
                                         layer_via_medida, float(texto_altura),
                                         arrow_size_factor * float(texto_altura), 2, dim_text_style)

                # c) LoteaLote (offset adicional)
                if (l_neg is not None) and (l_pos is not None) and (l_neg != l_pos):
                    dist_ll = float(Point(l_neg).distance(Point(l_pos)))
                    if dist_ll <= max_dist and abs(dist_ll - dist_gg) > eq_tol:
                        _add_dim_aligned(ms, l_neg, l_pos, -1.0, off_ll,
                                         layer_via_medida, float(texto_altura),
                                         arrow_size_factor * float(texto_altura), 2, dim_text_style)
                if min_sep2 > 0.0:
                    placed_mid_pts.append((float(mid.x), float(mid.y)))






# === module: area_table.py ===
# === area_table.py ===
import math
from typing import Optional, Dict, Any
from shapely.geometry import Polygon
import ezdxf

def format_area_br_m2(area_m2: float, casas: int = 3) -> str:
    """Formata área com milhar '.' e decimal ',' seguido de m²."""
    try:
        s = f"{float(area_m2):,.{casas}f}"
    except Exception:
        return f"{area_m2} m²"
    s = s.replace(",", "|").replace(".", ",").replace("|", ".")
    return f"{s} m²"

def create_area_table(ms, 
                     per_interesse: Optional[Polygon] = None,
                     per_levantamento: Optional[Polygon] = None,
                     params: Any = None) -> bool:
    """
    Cria tabela com áreas de levantamento e núcleo (PER_INTERESSE).
    
    Args:
        ms: ModelSpace do DXF
        per_interesse: Polígono do perímetro de interesse (núcleo)
        per_levantamento: Polígono do perímetro de levantamento
        params: Parâmetros de configuração
        
    Returns:
        bool: True se a tabela foi criada com sucesso
    """
    try:
        # Configurações
        layer_tabela = getattr(params, "layer_tabela", "TOP_TABELA")
        altura_texto = getattr(params, "altura_texto_tabela", 2.0)
        style_texto = getattr(params, "style_texto", "SIMPLEX")
        
        # Posição da tabela
        anchor = getattr(params, "area_table_anchor", None)
        offset_x = getattr(params, "tabela_offset_x", 120.0)
        offset_y = getattr(params, "tabela_offset_y", 0.0)
        cell_width = getattr(params, "area_table_cell_w", getattr(params, "tabela_cell_w", 25.0))
        cell_height = getattr(params, "tabela_cell_h", 6.0)
        
        # Calcular áreas
        area_nucleo = None
        area_levantamento = None
        
        if per_interesse and not per_interesse.is_empty:
            try:
                area_nucleo = float(per_interesse.area)
            except Exception:
                pass
                
        if per_levantamento and not per_levantamento.is_empty:
            try:
                area_levantamento = float(per_levantamento.area)
            except Exception:
                pass
        
        # Se não há áreas válidas, não criar tabela
        if area_nucleo is None and area_levantamento is None:
            return False
        
        # Posição inicial da tabela
        if isinstance(anchor, tuple) and len(anchor) == 2:
            start_x, start_y = float(anchor[0]), float(anchor[1])
        else:
            start_x = offset_x
            start_y = offset_y
        
        # Desenho no formato duas células lado a lado, conforme exemplo
        total_w = cell_width * 2.0
        total_h = cell_height

        # Retângulo externo
        try:
            ms.add_lwpolyline([(start_x, start_y), (start_x + total_w, start_y),
                               (start_x + total_w, start_y - total_h), (start_x, start_y - total_h),
                               (start_x, start_y)], close=True, dxfattribs={"layer": layer_tabela})
            # Linha divisória central
            ms.add_line((start_x + total_w / 2.0, start_y), (start_x + total_w / 2.0, start_y - total_h),
                        dxfattribs={"layer": layer_tabela})
        except Exception:
            pass

        # Textos centralizados em cada c?lula (TEXT, nao MTEXT)
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

        # fallback se veio apenas um lado
        if left_data and right_data:
            add_center(left_data[0], left_data[1], start_x + total_w * 0.25)
            add_center(right_data[0], right_data[1], start_x + total_w * 0.75)
        elif left_data:
            add_center(left_data[0], left_data[1], start_x + total_w * 0.5)
        elif right_data:
            add_center(right_data[0], right_data[1], start_x + total_w * 0.5)

        return True
        
    except Exception as e:
        print(f"[WARN] Falha ao criar tabela de áreas: {e}")
        return False

# === module: ui.py ===
# === ui.py ===
import os
import json
import sys
import threading
import queue
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

_CFG = os.path.join(os.path.expanduser("~"), ".itesp_reurb_ui.json")

def _load_state():
    try:
        with open(_CFG, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_state(state: dict):
    try:
        with open(_CFG, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

class ToolTip:
    def __init__(self, widget, text, wrap=56):
        self.widget = widget; self.text = text; self.wrap = wrap; self.tip = None
        widget.bind("<Enter>", self._show); widget.bind("<Leave>", self._hide)
    def _show(self, *_):
        if self.tip or not self.text: return
        x, y = self.widget.winfo_pointerxy()
        tw = tk.Toplevel(self.widget); self.tip = tw
        tw.wm_overrideredirect(True); tw.wm_geometry(f"+{x+14}+{y+14}")
        lbl = tk.Label(tw, text=self.text, justify="left", background="#111", foreground="#fff",
                       relief="solid", borderwidth=1, padx=8, pady=6, wraplength=self.wrap*8, font=("Segoe UI", 9))
        lbl.pack()
    def _hide(self, *_):
        if self.tip: self.tip.destroy(); self.tip = None

def _row(parent, r, label, var, width=12, unit="", tip=""):
    ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=4)
    ent = ttk.Entry(parent, textvariable=var, width=width)
    ent.grid(row=r, column=1, sticky="we", padx=6, pady=4)
    ttk.Label(parent, text=unit).grid(row=r, column=2, sticky="w", padx=2)
    if tip: ToolTip(ent, tip)
    return ent

def _browse_entry(parent, r, label, var, kind="file", patterns=(("Todos","*.*"),), tip=""):
    ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=4)
    ent = ttk.Entry(parent, textvariable=var, width=64)
    ent.grid(row=r, column=1, sticky="we", padx=6, pady=4, columnspan=2)
    def go():
        init = os.path.dirname(var.get()) if var.get() else _load_state().get("lastdir", os.path.expanduser("~"))
        path = filedialog.askopenfilename(parent=parent, initialdir=init, filetypes=patterns) if kind=="file" \
               else filedialog.askdirectory(parent=parent, initialdir=init)
        if path:
            var.set(path); _save_state({"lastdir": os.path.dirname(path)})
    ttk.Button(parent, text="...", width=3, command=go).grid(row=r, column=3, padx=4)
    if tip: ToolTip(ent, tip)
    return ent



def abrir_ui(params_default, on_execute=None) -> dict:
    root = tk.Tk()
    root.title("REURB - Exportador")
    root.geometry("1200x650")

    nb = ttk.Notebook(root)
    tab_geral = ttk.Frame(nb)
    tab_textos = ttk.Frame(nb)
    nb.add(tab_geral, text="Geral")
    nb.add(tab_textos, text="Textos")
    nb.pack(fill="both", expand=True, padx=10, pady=8)

    # === Arquivos ===
    v_nome = tk.StringVar(value="PROJETO")
    v_txt = tk.StringVar(value="E:/PROJETO_NEIDE/teste/junq/EDITADO.txt")
    v_dados = tk.StringVar(value="E:/PROJETO_NEIDE/teste/junq/sistvia.dxf")
    v_mdt = tk.StringVar(value="E:/PROJETO_NEIDE/teste/junq/JD_JUNQUEIROPOLIS.tif")
    v_out = tk.StringVar(value="E:/PROJETO_NEIDE/teste/junq")

    tab_geral.columnconfigure(0, weight=3)
    tab_geral.columnconfigure(1, weight=2)
    tab_geral.rowconfigure(0, weight=0)
    tab_geral.rowconfigure(1, weight=0)
    tab_geral.rowconfigure(2, weight=0)
    tab_geral.rowconfigure(3, weight=1)

    frm_files = ttk.LabelFrame(tab_geral, text="Arquivos")
    frm_files.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=10, pady=8)
    _browse_entry(frm_files, 0, "TXT (blocos, opcional):", v_txt, "file", (("TXT","*.txt"),))
    _browse_entry(frm_files, 1, "DXF Dados (opcional):", v_dados, "file", (("DXF","*.dxf"),))
    _browse_entry(frm_files, 2, "MDT (GeoTIFF, opcional):", v_mdt, "file", (("GeoTIFF","*.tif;*.tiff"),))
    _browse_entry(frm_files, 3, "Pasta de saida:", v_out, "dir")

    ttk.Label(frm_files, text="Nome do projeto/area:").grid(row=4, column=0, sticky="w", padx=6, pady=6)
    ttk.Entry(frm_files, textvariable=v_nome, width=32).grid(row=4, column=1, sticky="w", padx=6, pady=6)

    # === Simbologia fixa ===
    v_simb_tipo = tk.StringVar(value="SIMBOLOGIA")

    # === O que exportar ===
    frm_opts = ttk.LabelFrame(tab_geral, text="O que exportar")
    frm_opts.grid(row=0, column=1, sticky="nsew", padx=10, pady=8)
    v_do_txt = tk.BooleanVar(value=False)
    v_do_per = tk.BooleanVar(value=False)
    v_do_cur = tk.BooleanVar(value=False)
    v_do_dren = tk.BooleanVar(value=False)
    v_do_vias = tk.BooleanVar(value=False)
    v_do_lotes_dim = tk.BooleanVar(value=False)
    exports_vars = (v_do_txt, v_do_per, v_do_cur, v_do_dren, v_do_vias, v_do_lotes_dim)
    ttk.Checkbutton(frm_opts, text="TXT (blocos)", variable=v_do_txt).grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frm_opts, text="Perimetro", variable=v_do_per).grid(row=0, column=1, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frm_opts, text="Curvas de nivel", variable=v_do_cur).grid(row=1, column=0, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frm_opts, text="Setas de drenagem (MDT)", variable=v_do_dren).grid(row=1, column=1, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frm_opts, text="Vias (medidas)", variable=v_do_vias).grid(row=2, column=0, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frm_opts, text="Lotes (dimensoes)", variable=v_do_lotes_dim).grid(row=2, column=1, sticky="w", padx=6, pady=4)
    def _select_all():
        for var in exports_vars:
            var.set(True)
    ttk.Button(frm_opts, text="Selecionar todos", command=_select_all).grid(row=0, column=3, rowspan=2, padx=6, pady=4)

    # === Fonte do perimetro ===
    frm_per_src = ttk.LabelFrame(tab_geral, text="Perimetro: fonte para Tabela e Vertices")
    frm_per_src.grid(row=1, column=1, sticky="nsew", padx=10, pady=8)
    v_per_src = tk.StringVar(value="")
    ttk.Radiobutton(frm_per_src, text="PER_INTERESSE", variable=v_per_src, value="interesse").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ttk.Radiobutton(frm_per_src, text="PER_LEVANTAMENTO", variable=v_per_src, value="levantamento").grid(row=0, column=1, sticky="w", padx=6, pady=4)

    # === Curvas de nivel ===
    frm_curvas = ttk.LabelFrame(tab_geral, text="Curvas de nivel")
    frm_curvas.grid(row=2, column=1, sticky="nsew", padx=10, pady=8)
    v_label_mode = tk.StringVar(value="ends" if bool(getattr(params_default, "curva_label_ends_only", False)) else "padrao")
    ttk.Radiobutton(frm_curvas, text="Padrao (ao longo da curva)", variable=v_label_mode, value="padrao").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ttk.Radiobutton(frm_curvas, text="Somente nas extremidades (inicial/final)", variable=v_label_mode, value="ends").grid(row=1, column=0, sticky="w", padx=6, pady=4)

    # === TEXTOS ===
    def _get(p, d):
        try:
            return float(getattr(params_default, p))
        except Exception:
            return d

    tab_textos.columnconfigure(0, weight=1)
    frm_txt = ttk.LabelFrame(tab_textos, text="Textos e Dimensoes")
    frm_txt.grid(row=0, column=0, sticky="nsew", padx=10, pady=8)
    v_h_soleira = tk.DoubleVar(value=_get("altura_texto_soleira", 0.75))
    v_h_area = tk.DoubleVar(value=_get("altura_texto_area", 0.60))
    v_h_p = tk.DoubleVar(value=_get("altura_texto_P", 0.50))
    v_h_tab = tk.DoubleVar(value=_get("altura_texto_tabela", 2.00))
    v_h_curva = tk.DoubleVar(value=_get("altura_texto_curva", 0.40))
    v_h_via = tk.DoubleVar(value=_get("altura_texto_via", 0.40))
    v_dimtxt_ord_via = tk.DoubleVar(value=_get("dimtxt_ordinate_via", 0.50))
    v_dimtxt_ord_per = tk.DoubleVar(value=_get("dimtxt_ordinate_perim", 0.50))
    _row(frm_txt, 0, "Texto Soleira (m):", v_h_soleira)
    _row(frm_txt, 1, "Texto Areas (m):", v_h_area)
    _row(frm_txt, 2, "Texto P (m):", v_h_p)
    _row(frm_txt, 3, "Texto Tabela (m):", v_h_tab)
    _row(frm_txt, 4, "Texto Curva (m):", v_h_curva)
    _row(frm_txt, 5, "Texto Via (m):", v_h_via)
    _row(frm_txt, 6, "DIM Text (Ordinate) - VIA:", v_dimtxt_ord_via, tip="Altura do texto das cotas do tipo Ordinate usadas em vias (se aplicavel).")
    _row(frm_txt, 7, "DIM Text (Ordinate) - PERIMETRO:", v_dimtxt_ord_per, tip="Altura do texto das cotas Ordinate do perimetro (azimute/distancia).")

    # === Log ===
    frm_log = ttk.LabelFrame(tab_geral, text="Log")
    frm_log.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=8)
    log_text = tk.Text(frm_log, height=10, wrap="word")
    log_scroll = ttk.Scrollbar(frm_log, command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_text.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
    log_scroll.pack(side="right", fill="y", padx=(0, 6), pady=6)

    class _UILogStream:
        def __init__(self, widget: tk.Text):
            self.widget = widget
        def write(self, s: str):
            if not s:
                return
            try:
                self.widget.insert("end", s)
                self.widget.see("end")
                self.widget.update_idletasks()
            except Exception:
                pass
        def flush(self):
            pass

    class _QueueLogStream:
        def __init__(self, q: queue.Queue):
            self.q = q
        def write(self, s: str):
            if s:
                self.q.put(s)
        def flush(self):
            pass

    log_q: queue.Queue = queue.Queue()

    def _poll_log():
        try:
            while True:
                msg = log_q.get_nowait()
                log_text.insert("end", msg)
                log_text.see("end")
        except queue.Empty:
            pass
        root.after(100, _poll_log)

    _poll_log()

    # === Rodape ===
    frm_bottom = ttk.Frame(root)
    frm_bottom.pack(fill="x", padx=10, pady=10)
    out = {}

    def _ok():
        if not v_out.get():
            messagebox.showerror("Atencao", "Pasta de saida e obrigatoria.")
            return
        per_src_val = v_per_src.get().strip() or None
        vias_cfg = {
            "via_offset_texto": float(getattr(params_default, "via_offset_texto", 0.50)),
            "via_cross_span": float(getattr(params_default, "via_cross_span", 80.0)),
            "via_nome_maiusculas": bool(getattr(params_default, "via_nome_maiusculas", False)),
            "via_offset_lote_lote_extra_m": float(getattr(params_default, "via_offset_lote_lote_extra_m", 0.60)),
        }
        curvas_cfg = {
            "curva_equidist": float(getattr(params_default, "curva_equidist", 1.0)),
            "curva_mestra_cada": int(getattr(params_default, "curva_mestra_cada", 5)),
            "curva_char_w_factor": float(getattr(params_default, "curva_char_w_factor", 0.60)),
            "curva_gap_margin": float(getattr(params_default, "curva_gap_margin", 0.50)),
            "curva_label_step_m": float(getattr(params_default, "curva_label_step_m", 80.0)),
            "curva_min_len": float(getattr(params_default, "curva_min_len", 10.0)),
            "curva_min_area": float(getattr(params_default, "curva_min_area", 20.0)),
            "curva_smooth_sigma_px": float(getattr(params_default, "curva_smooth_sigma_px", 1.0)),
            "curva_label_ends_only": (v_label_mode.get() == "ends"),
        }
        setas_cfg = {
            "setas_buffer_distancia": float(getattr(params_default, "setas_buffer_distancia", 5.0)),
            "setas_seg_curto_max": int(getattr(params_default, "setas_seg_curto_max", 3)),
            "setas_seg_medio_max": int(getattr(params_default, "setas_seg_medio_max", 4)),
            "setas_seg_longo_max": int(getattr(params_default, "setas_seg_longo_max", 5)),
        }
        out.update({
            "nome_area": v_nome.get().strip() or "PROJETO",
            "paths": {
                "txt": v_txt.get() or None,
                "simb_tipo": v_simb_tipo.get(),
                "dados": v_dados.get() or None,
                "mdt": v_mdt.get() or None,
                "saida": v_out.get(),
            },
            "exports": {
                "txt": bool(v_do_txt.get()),
                "perimetros": bool(v_do_per.get()),
                "curvas": bool(v_do_cur.get()),
                "drenagem": bool(v_do_dren.get()),
                "vias": bool(v_do_vias.get()),
                "lotes_dim": bool(v_do_lotes_dim.get()),
                "per_source": per_src_val,
            },
            "textos": {
                "altura_texto_soleira": float(v_h_soleira.get()),
                "altura_texto_area": float(v_h_area.get()),
                "altura_texto_P": float(v_h_p.get()),
                "altura_texto_tabela": float(v_h_tab.get()),
                "altura_texto_curva": float(v_h_curva.get()),
                "altura_texto_via": float(v_h_via.get()),
                "dimtxt_ordinate_via": float(v_dimtxt_ord_via.get()),
                "dimtxt_ordinate_perim": float(v_dimtxt_ord_per.get()),
            },
            "vias": vias_cfg,
            "curvas": curvas_cfg,
            "setas": setas_cfg,
        })
        if on_execute:
            def _run():
                old_stdout, old_stderr = sys.stdout, sys.stderr
                q_logger = _QueueLogStream(log_q)
                sys.stdout = q_logger
                sys.stderr = q_logger
                try:
                    on_execute(out)
                except Exception as e:
                    log_q.put(f"[ERRO] {e}\n")
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
            threading.Thread(target=_run, daemon=True).start()
            return
        root.destroy()

    ttk.Button(frm_bottom, text="Cancelar", command=root.destroy).pack(side="right", padx=6)
    ttk.Button(frm_bottom, text="Executar", command=_ok).pack(side="right", padx=6)

    root.mainloop()
    return out

# === module: run_bloco_reurb.py ===
# run_bloco_reurb.py

import math
import copy
from shapely.ops import unary_union, split, snap
from shapely.geometry import MultiLineString

from osgeo import gdal, osr

try:
    LAYER_SOLEIRA_AREA
except Exception:
    # fallback so para nao quebrar em tempo de execucao se o linter nao achar
    LAYER_SOLEIRA_AREA = "TOP_AREA_LOTE"



# caminhos fixos de simbologia (ver config.py)

# layers de ENTRADA (agora vindo do config)
LYR_EIXO = LAYER_EIXO_VIA
# Se não houver separação entre pistas com/sem guia no seu DXF, estas listas podem ficar vazias.
LYR_GUIA_COM = "SISTVIA_PAV_COM_GUIA"
LYR_GUIA_SEM = "SISTVIA_PAV_SEM_GUIA"
LYR_DIV_FIS_LOTE = LAYER_LOTES
LYR_EDIF = LAYER_EDIF
LYR_PER_INTERESSE = LAYER_PERIMETRO
LYR_PER_LEVANT = LAYER_PER_LEVANTAMENTO


def _epsg_from_gdal_dataset(ds):
    if ds is None:
        return None, "sem dataset"
    try:
        wkt = ds.GetProjection()
        if not wkt:
            return None, "sem WKT"
        srs = osr.SpatialReference(); srs.ImportFromWkt(wkt)
        try:
            srs.AutoIdentifyEPSG()
        except Exception:
            pass
        code = srs.GetAuthorityCode("PROJCS") or srs.GetAuthorityCode("GEOGCS")
        return (int(code) if code else None), ("authority" if code else "indeterminado")
    except Exception as e:
        return None, f"erro:{e}"

def _rotular_areas_lote(ms, lotes, edificacoes, vias, params):
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
    def _normal_from_rotation(deg):
        if deg is None:
            return (0.0, 1.0)
        rad = math.radians(deg)
        return (-math.sin(rad), math.cos(rad))

    line_gap = max(float(getattr(params, "altura_texto_soleira", 0.5)), 0.50)

    for lote_poly in (lotes or []):
        # texto da area
        try:
            area_txt = f"{float(lote_poly.area):.2f} m2"
        except Exception:
            continue

        # base: maior edificacao (centro representativo)  lote (fallback)
        try:
            e_sel = None
            if edificacoes:
                max_a = -1.0
                for e in edificacoes:
                    try:
                        c = e.representative_point()
                        if not lote_poly.contains(c):
                            continue
                        a = float(e.area)
                        if a > max_a:
                            max_a = a; e_sel = e
                    except Exception:
                        continue
            if e_sel is not None:
                base_x, base_y = e_sel.representative_point().coords[0]
            else:
                base_x, base_y = lote_poly.representative_point().coords[0]
        except Exception:
            continue

        # rotacao: testada do lote  via (fallback)
        rot_txt = None
        try:
            rot_txt = encontrar_rotacao_por_lote((base_x, base_y), [lote_poly], delta=getattr(params,"delta_interp",2.0), raio=12.0)
        except Exception:
            rot_txt = None
        if rot_txt is None:
            try:
                rot_txt = encontrar_rotacao_por_via((base_x, base_y), vias or [], getattr(params, "dist_busca_rot", 50.0), getattr(params, "delta_interp", 2.0))
            except Exception:
                rot_txt = None
        rot_txt = _upright(rot_txt) if rot_txt is not None else None
        nux, nuy = _normal_from_rotation(rot_txt)

        # posicao final: "em cima" da base
        tx_x = float(base_x) + nux * (1.0 * line_gap)
        tx_y = float(base_y) + nuy * (1.0 * line_gap)

        try:
            t = ms.add_text(area_txt, dxfattribs={"height": float(getattr(params, "altura_texto_area", 0.6)),
                                                  "style": STYLE_TEXTO, "layer": LAYER_SOLEIRA_AREA})
            t.dxf.insert = (tx_x, tx_y)
            try:
                t.dxf.halign = 1  # center
                t.dxf.valign = 2  # middle
                t.dxf.align_point = (tx_x, tx_y)
            except Exception:
                pass
            try:
                t.dxf.rotation = 0.0
            except Exception:
                pass
        except Exception:
            continue

def _executar(settings):
    if not settings:
        return

    import time
    nome_area = settings["nome_area"]
    P = settings["paths"]
    X = settings["exports"]
    T = settings["textos"]
    V = settings["vias"]
    C = settings["curvas"]
    S = settings["setas"]

    # === 2) Simbologia fixa ===
    simb_path = FIXED_SIMBOLOGIA_PATH
    print(f"[INFO] DXF de simbologia (fixo): {simb_path}")

    # === 3) Abrir doc de saida e perfil de camadas ===
    doc, ms = abrir_dxf_simbologia(simb_path)
    garantir_estilos_blocos(doc, GLOBAL_PARAMS)
    perfil = build_layer_profile(doc, simb_path)
    print("[INFO] Perfil de camadas:", perfil)

    # === 4) Params runtime ===
    params = copy.deepcopy(GLOBAL_PARAMS)
    for k, v in T.items():
        setattr(params, k, v)
    for k, v in C.items():
        setattr(params, k, v)
    for k, v in S.items():
        setattr(params, k, v)
    setattr(params, "style_texto", STYLE_TEXTO)
    setattr(params, "via_dim_style", "Cota_Rua")
    setattr(params, "layer_curvas", perfil["curva_i"])
    setattr(params, "layer_curvas_mestra", perfil["curva_m"])
    setattr(params, "layer_curvas_txt", perfil["curva_txt"])
    setattr(params, "layer_per_out_tab", perfil["per_tab"])
    setattr(params, "layer_per_out_vert", perfil["per_vert"])
    setattr(params, "layer_txt_grande", perfil.get("txt_grande"))
    setattr(params, "layer_txt_soleira", perfil.get("txt_soleira"))
    setattr(params, "layer_soleira_num_pav", LAYER_SOLEIRA_NUM_PAV)
    setattr(params, "type_to_layer", dict(TYPE_TO_LAYER))
    setattr(params, "layer_to_block", dict(LAYER_TO_BLOCK))
    setattr(params, "escrever_area_lote", True)
    setattr(params, "soleira_num_pav_mtext", True)
    setattr(params, "rotacionar_numero_casa", False)

# === 5) Entradas ===
    print("[INFO] Arquivos selecionados:")
    print(f"   TXT: {P['txt'] or ''}")
    print(f"   DXF SIMB: {simb_path}")
    print(f"   DXF DADOS: {P['dados'] or ''}")
    print(f"   MDT: {P['mdt'] or ''}")
    print(f"   SAIDA: {P['saida']}")

    need_mdt_reasons = []
    if X.get("txt"):
        need_mdt_reasons.append("TXT (blocos)")
    if X.get("curvas"):
        need_mdt_reasons.append("curvas de nivel")
    if X.get("drenagem"):
        need_mdt_reasons.append("setas de drenagem")
    need_mdt = bool(need_mdt_reasons)

    if need_mdt and not P.get("mdt"):
        print("[ERROR] MDT obrigatorio para: " + ", ".join(need_mdt_reasons))
        return

    df = None
    if X.get("txt") and P.get("txt"):
        df = ler_txt(P["txt"])
        print(f"[INFO] TXT lido: {len(df)} linhas.")
    elif X.get("txt"):
        print("[WARN] TXT marcado sem arquivo  pulado.")

    dados_path = P.get("dados")
    dados_doc = dados_msp = None
    dados_kwargs: dict = {}
    if dados_path:
        try:
            dados_doc, dados_msp = abrir_dxf_simbologia(dados_path)
            dados_kwargs = {"doc": dados_doc, "msp": dados_msp}
        except Exception as e:
            print(f"[WARN] DXF dados: falha ao abrir {dados_path}: {e}")
            dados_path = None
            dados_doc = dados_msp = None
            dados_kwargs = {}

    lotes = carregar_poligonos_por_layer(dados_path, LYR_DIV_FIS_LOTE, **dados_kwargs) if dados_path else []
    edificacoes = carregar_poligonos_por_layer(dados_path, LYR_EDIF, **dados_kwargs) if dados_path else []
    eixos_via = carregar_linhas_por_layers(dados_path, {LYR_EIXO}, **dados_kwargs) if dados_path else []
    # Bordas de guia: use conjunto do config (ex.: {"VIA"}); fallback aos nomes legados
    guias_layers = set(LAYERS_PISTA_BORDA) if LAYERS_PISTA_BORDA else {LYR_GUIA_COM, LYR_GUIA_SEM}
    guias_via = carregar_linhas_por_layers(dados_path, guias_layers, **dados_kwargs) if dados_path else []
    testadas_abertas = carregar_linhas_por_layers(dados_path, {LYR_DIV_FIS_LOTE}, **dados_kwargs) if dados_path else []
    per_interesse = carregar_poligonos_por_layer(dados_path, LYR_PER_INTERESSE, **dados_kwargs) if dados_path else []
    per_levantamento = carregar_poligonos_por_layer(dados_path, LYR_PER_LEVANT, **dados_kwargs) if dados_path else []

    if dados_path:
        print(f"[INFO] DXF dados  lotes={len(lotes)}, edifs={len(edificacoes)}, "
              f"eixos_via={len(eixos_via)}, guias_via={len(guias_via)}, "
              f"per_interesse={len(per_interesse)}, per_levantamento={len(per_levantamento)}")

    per_interesse_union = None
    if per_interesse:
        try: per_interesse_union = unary_union(per_interesse)
        except Exception: per_interesse_union = per_interesse[0]

    per_levantamento_union = None
    if per_levantamento:
        try: per_levantamento_union = unary_union(per_levantamento)
        except Exception: per_levantamento_union = per_levantamento[0]

    clip_poly = per_levantamento_union
    if clip_poly is None and per_levantamento:
        clip_poly = per_levantamento[0]

    area_info = {}
    if per_interesse_union is not None:
        try: area_info["nucleo"] = float(per_interesse_union.area)
        except Exception:
            pass
    if per_levantamento_union is not None:
        try: area_info["levantamento"] = float(per_levantamento_union.area)
        except Exception:
            pass
    setattr(params, "perimetro_area_info", area_info)
    setattr(params, "perimetro_levantamento_geom", per_levantamento_union)
    setattr(params, "perimetro_interesse_geom", per_interesse_union)
    # MDT (quando necessario): rasterio + GDAL
    mdt_src, get_elevation = None, None
    mdt_gdal = None
    epsg_auto, epsg_note = None, ""
    if P.get("mdt"):
        try:
            mdt_src, get_elevation = make_get_elevation(P["mdt"])
        except Exception as e:
            print(f"[ERROR] MDT: falha ao abrir {P['mdt']}: {e}")
            return
        try:
            mdt_gdal = mdt_src if hasattr(mdt_src, "GetRasterBand") else gdal.Open(P["mdt"])
        except Exception as e_gdal:
            mdt_gdal = None
            print(f"[WARN] MDT: falha ao carregar via GDAL {P['mdt']}: {e_gdal}")
        if mdt_gdal is not None:
            try:
                epsg_auto, epsg_note = _epsg_from_gdal_dataset(mdt_gdal)
            except Exception:
                epsg_auto, epsg_note = None, ""

        if get_elevation is None and need_mdt:
            print("[ERROR] MDT: nao foi possivel obter funcao de cota. Processo interrompido.")
            return

    # EPSG local usado apenas para SHP de nomes (quando existir)
    epsg_choice = int(epsg_auto or 31983)
    print(f"[INFO] EPSG local (auto/default): {epsg_choice}")

    # Provider de nomes: tenta SHP local
    name_provider = None
    if X.get("vias") and eixos_via:
        if per_levantamento_union is not None:
            bbox_local = per_levantamento_union.bounds
        elif per_interesse_union is not None:
            bbox_local = per_interesse_union.bounds
        else:
            bbox_local = MultiLineString(eixos_via).bounds

        # 1) tenta SHP local (se existir)
        shp_path = r"D:\2304_REURB_SP\DOCUMENTOS\SHP\SIRGAS_SHP_logradouronbl_line.shp"
        try:
            import os
            if os.path.isfile(shp_path):
                prov_shp = build_shp_name_provider(shp_path, epsg_choice)
                if prov_shp:
                    print("[INFO] Nomes de via: usando SHP local")
                    name_provider_shp = lambda x, y: prov_shp.get(x, y, max_dist_m=15.0)
                else:
                    name_provider_shp = None
            else:
                name_provider_shp = None
        except Exception as e:
            print(f"[WARN] SHP nomes: falha ao abrir {shp_path}: {e}")
            name_provider_shp = None

        name_provider = name_provider_shp

    # === 8) Execucao por blocos ===
    # 7.5) So AREA do lote (quando marcado e TXT desmarcado)
    if X.get('area_lote') and not X.get('txt'):
        try:
            _rotular_areas_lote(ms, lotes, edificacoes, eixos_via, params)
            print(f"[INFO] Areas de lote rotuladas: {len(lotes)} candidatos")
        except Exception as e:
            print(f"[WARN] Falhou ao rotular areas de lote: {e}")

    try:
        dimtxt_backup = getattr(params, "dimtxt_ordinate", None)

        setas_inseridas_via_processing = False
        # Linhas candidatas para setas: preferir eixo; se vazio, usar guias
        linhas_setas = eixos_via if (eixos_via and len(eixos_via) > 0) else guias_via
        if X.get("txt") and df is not None:
            t0 = time.perf_counter()
            processar_registros(
                df=df, ms=ms, doc=doc, params=params,
                lotes=lotes, edificacoes=edificacoes,
                via_lines_setas=(linhas_setas if X.get("drenagem") else []),
                via_lines_geral=(eixos_via + guias_via),
                get_elevation=get_elevation,
            )
            print(f"[TIME] TXT/processamento: {time.perf_counter() - t0:.2f}s")
            # As setas de drenagem são inseridas dentro de processing
            setas_inseridas_via_processing = bool(X.get("drenagem"))

        # Caso o usuário tenha marcado drenagem sem TXT (ou TXT não processado),
        # garantimos que as setas sejam inseridas aqui também.
        if X.get("drenagem") and (linhas_setas and len(linhas_setas) > 0) and not setas_inseridas_via_processing:
            try:
                if get_elevation is None:
                    print("[WARN] Drenagem: MDT nao carregado  setas nao geradas.")
                else:
                    t0 = time.perf_counter()
                    inserir_setas_drenagem(ms, doc, linhas_setas, get_elevation, params)
                    print(f"[TIME] Drenagem: {time.perf_counter() - t0:.2f}s")
            except Exception as e:
                print(f"[WARN] Drenagem: falha ao inserir setas: {e}")

        if X.get("perimetros") and (per_levantamento or per_interesse):
            per_src = (settings["exports"].get("per_source") or "").lower()
            polys = None
            if per_src == "levantamento":
                polys = per_levantamento; print("[INFO] Perimetro: usando PER_LEVANTAMENTO")
            elif per_src == "interesse":
                polys = per_interesse; print("[INFO] Perimetro: usando PER_INTERESSE")
            elif per_src == "auto":
                if per_levantamento:
                    polys = per_levantamento; print("[INFO] Perimetro: usando PER_LEVANTAMENTO (auto)")
                elif per_interesse:
                    polys = per_interesse; print("[INFO] Perimetro: usando PER_INTERESSE (auto)")
            else:
                # Fallback automático se não especificado ou desconhecido
                if per_interesse:
                    polys = per_interesse; print("[INFO] Perimetro: usando PER_INTERESSE (fallback)")
                elif per_levantamento:
                    polys = per_levantamento; print("[INFO] Perimetro: usando PER_LEVANTAMENTO (fallback)")
            if polys:
                t0 = time.perf_counter()
                setattr(params, "dimtxt_ordinate", float(getattr(params, "dimtxt_ordinate_perim", 0.5)))
                # desloca rotulos para fora do perimetro selecionado
                per_ref = polys[0] if polys else None
                processar_perimetros(ms, doc, params, polys, per_ref)
                print(f"[TIME] Perimetro: {time.perf_counter() - t0:.2f}s")
            else:
                print("[WARN] Nenhum poligono de perimetro encontrado.")
        elif X.get("perimetros"):
            print("[WARN] Perimetro marcado mas sem DXF de dados  pulado.")

        if X.get("lotes_dim") and lotes:
            try:
                t0 = time.perf_counter()
                processar_lotes_dimensoes(ms, doc, params, lotes)
                print(f"[TIME] Lotes dimensoes: {time.perf_counter() - t0:.2f}s")
            except Exception as e:
                print(f"[WARN] Lotes dimensoes: falha ao gerar: {e}")
        elif X.get("lotes_dim"):
            print("[WARN] Lotes dimensoes marcadas mas sem DXF de dados  pulado.")


        if X.get("vias") and eixos_via and guias_via:
            print(f"[INFO] Vias: iniciando medidas (eixos={len(eixos_via)}, guias={len(guias_via)})")
            setattr(params, "dimtxt_ordinate", float(getattr(params, "dimtxt_ordinate_via", 0.5)))
            try:
                t0 = time.perf_counter()
                medir_e_rotular_vias(
                    ms=ms,
                    eixos=eixos_via,
                    bordas_guia=guias_via,
                    lotes_polygons=lotes,
                    testada_extra_lines=testadas_abertas,
                    texto_altura=params.altura_texto_via,
                    offset_texto_m=params.via_offset_texto,
                    cross_span_m=params.via_cross_span,
                    amostras_fracs=(1/3, 2/3),
                    name_provider=name_provider,
                    ativar_testada_testada=True,
                    nome_offset_m=float(getattr(params, "via_nome_offset_m", 0.20)),
                    nome_offset_side="auto",
                    nome_sufixo=" (Asfalto)",
                    style_texto=STYLE_TEXTO,
                    dim_text_style=getattr(params, "via_dim_style", "Cota_Rua"),
                    layer_via_medida=perfil["via_med"],
                    layer_via_nome=perfil["via_nome"],
                    sample_mode="entre_intersecoes",
                    offset_lote_lote_extra_m=float(getattr(params, "via_offset_lote_lote_extra_m", 0.60)),
                    dim_gap_m=float(getattr(params, "via_dim_gap_m", 0.60)),
                    dim_min_len_m=float(getattr(params, "via_dim_min_len_m", 12.0)),
                    dim_min_spacing_m=float(getattr(params, "via_dim_min_spacing_m", 25.0)),
                    dim_max_por_trecho=int(getattr(params, "via_dim_max_por_trecho", 2)),
                    dim_max_dist_m=float(getattr(params, "via_dim_max_dist_m", 20.0)),
                    dim_min_sep_area_m=float(getattr(params, "via_dim_min_sep_area_m", 10.0)),
                    dim_equal_tol_m=float(getattr(params, "via_dim_equal_tol_m", 0.05)),
                    nome_offset_add_dim_m=None,
                    nome_side_mode="oposto_dim",
                    nome_shift_along_m=float(getattr(params, "via_nome_shift_along_m", 6.0)),
                    nome_case="upper" if V.get("via_nome_maiusculas", False) else "as_is",
                )
                print(f"[TIME] Vias: {time.perf_counter() - t0:.2f}s")
                print("[INFO] Vias: medidas concluidas")
            except Exception as e:
                print(f"[WARN] Vias: falha ao medir/rotular vias: {e}")
        elif X.get("vias"):
            faltando = []
            if not eixos_via:
                faltando.append(f"eixo de via (layer {LYR_EIXO})")
            if not guias_via:
                try:
                    layers_guias = ", ".join(sorted(guias_layers))
                except Exception:
                    layers_guias = "bordas de guia"
                faltando.append(f"bordas de guia ({layers_guias})")
            faltando_txt = ", ".join(faltando) if faltando else "dados de via"
            print(f"[WARN] Vias marcadas mas faltam dados no DXF: {faltando_txt}. Medidas nao geradas.")

        # Drenagem: agora e feita dentro do processing (nao duplicar aqui)

        if X.get("curvas") and P.get("mdt"):
            if mdt_gdal is None:
                print("[WARN] Curvas: MDT nao aberto via GDAL  pulado.")
            else:
                t0 = time.perf_counter()
                gerar_curvas_nivel(ms, mdt_gdal, params, clip_poly=clip_poly)
                print(f"[TIME] Curvas: {time.perf_counter() - t0:.2f}s")
        elif X.get("curvas"):
            print("[WARN] Curvas marcadas mas sem MDT  pulado.")

        if dimtxt_backup is not None:
            setattr(params, "dimtxt_ordinate", dimtxt_backup)

        # Gera tabela de áreas sempre que houver perímetros (independente da simbologia)
        try:
            _area_done = getattr(params, "_area_table_done", False)
        except Exception:
            _area_done = False
        if (per_interesse_union is not None or per_levantamento_union is not None) and not _area_done:
            try:
                t0 = time.perf_counter()
                create_area_table(ms, per_interesse_union, per_levantamento_union, params)
                print(f"[TIME] Tabela areas: {time.perf_counter() - t0:.2f}s")
                setattr(params, "_area_table_done", True)
                print("[INFO] Tabela de áreas criada")
            except Exception as e:
                print(f"[WARN] Falha ao criar tabela de áreas: {e}")

        try: doc.purge()
        except Exception: pass
        path_final = salvar_dxf(doc, P["saida"], f"{nome_area}_BLOCOS.dxf")
        print(f" Arquivo salvo em: {path_final}")

    finally:
        try:
            if P.get("mdt") and mdt_gdal:
                try: mdt_gdal.FlushCache()
                except Exception: pass
                mdt_gdal = None
        except Exception:
            pass
        try:
            if P.get("mdt") and mdt_src and hasattr(mdt_src, "close"):
                mdt_src.close()
        except Exception:
            pass

def main():
    def _run(settings):
        _executar(settings)
    abrir_ui(GLOBAL_PARAMS, on_execute=_run)

if __name__ == "__main__":
    main()





