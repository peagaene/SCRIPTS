# perimetro.py
import math
from typing import List, Tuple
import numpy as np
from shapely.geometry import Polygon, Point
from config import (
    Params, STYLE_TEXTO,
    BLOCO_VERTICE, LAYER_VERTICE_PTO, LAYER_VERTICE_TXT,
    LAYER_TABELA, LAYER_ORDINATE,
)

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

def _table_header(ms, x0, y0, cw, ch, txt_h, layer, style):
    # Linha 1: Segmento | DistAncia | Azimute | Ponto | [Coordenadas (largura 2*cw)]
    headers = ["Segmento", "DistAncia (m)", "Azimute", "Ponto"]
    x=x0
    for htxt in headers:
        _draw_cell(ms, x, y0, cw, ch, layer); _add_text_center(ms, htxt, x+cw/2, y0-ch/2, txt_h, layer, style); x += cw
    # cAlula mesclada para "Coordenadas" (2*cw)
    _draw_cell(ms, x, y0, 2*cw, ch, layer)
    _add_text_center(ms, "Coordenadas", x+cw, y0-ch/2, txt_h, layer, style)
    # Linha 2: E | N
    y1 = y0 - ch
    _draw_cell(ms, x, y1, cw, ch, layer); _add_text_center(ms, "E", x+cw/2, y1-ch/2, txt_h, layer, style)
    _draw_cell(ms, x+cw, y1, cw, ch, layer); _add_text_center(ms, "N", x+cw+cw/2, y1-ch/2, txt_h, layer, style)

def _table_row(ms, row_idx, x0, y0, cw, ch, txt_h, layer, style, seg, dist, az, pid, e, n):
    # y de inAcio da linha deste conteAodo (3Aa linha Aotil em diante)
    y = y0 - (2+row_idx)*ch
    cols = [seg, f"{dist:.2f}", _dms_str(az), pid, f"{e:.4f}", f"{n:.4f}"]
    # 4 primeiras cAlulas sAo cw; Aoltimas 2 tambAm cw (jA que cabeAalho aCoordenadasa mesclou)
    for i, val in enumerate(cols):
        # coluna 0..3
        if i < 4:
            x = x0 + i*cw
        else:
            x = x0 + 4*cw + (i-4)*cw
        _draw_cell(ms, x, y, cw, ch, layer)
        _add_text_center(ms, val, x+cw/2, y-ch/2, txt_h, layer, style)

def _add_ordinate(ms, midpoint, texto, layer, params: Params, style: str):
    dim = ms.add_ordinate_dim(midpoint, (0.5,-1.0), 0,
                              origin=midpoint, text=texto, rotation=0,
                              dxfattribs={'layer': layer},
                              override={"dimtxt": params.dimtxt_ordinate,
                                        "dimasz": params.dimasz_ordinate,
                                        "dimtsz": 0, "dimtad": 0,
                                        "dimtxsty": style})
    dim.render()

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
                candidate = Point(px + dir_x * dist, py + dir_y * dist)
                tries += 1
            px, py = candidate.x, candidate.y
        else:
            px += offset_m
            py += offset_m
        _add_text_center(ms, f"P{i}", px, py, params.altura_texto_P, LAYER_VERTICE_TXT, style_texto)

    # 2) Tabela (Segmento / DistAncia / Azimute / Ponto / E / N)
    max_x, max_y = _bbox_max(V)
    x0 = max_x + params.tabela_offset_x
    y0 = max_y + params.tabela_offset_y
    cw, ch, th = params.tabela_cell_w, params.tabela_cell_h, params.altura_texto_tabela

    _table_header(ms, x0, y0, cw, ch, th, LAYER_TABELA, style_texto)

    n = len(V)
    for i in range(n):
        a = V[i]; b = V[(i+1) % n]
        seg = f"{i+1} - {((i+1)%n)+1}"
        dist = _dist(a,b)
        az = _azimute(a,b)
        pid = f"P{i+1}"
        e, ncoord = a[0], a[1]
        _table_row(ms, i, x0, y0, cw, ch, th, LAYER_TABELA, style_texto, seg, dist, az, pid, e, ncoord)

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


