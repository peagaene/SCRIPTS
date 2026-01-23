# processing.py
import re, math, numpy as np, pandas as pd
from shapely.geometry import Point, LineString
from config import (
    VERBOSE, Params, STYLE_TEXTO, BLOCO_SETA, BLOCO_SOLEIRA_POS,
    LAYER_SETAS_SAIDA, LAYER_SOLEIRA_BLOCO, LAYER_SOLEIRA_NUM_PAV, LAYER_SOLEIRA_AREA,
    TYPE_TO_LAYER, LAYER_TO_BLOCK,
)
from geom_utils import (
    encontrar_rotacao_por_via, encontrar_rotacao_por_lote, calcular_offset,
    segmentos_ordenados_por_proximidade,
)

# ---- padroes classicos de soleira (mantidos) ----
RE_SOLEIRA_NUM = re.compile(r'^\s*(?P<pav>\d)\s*PV\s*(?P<num>\d+)(?P<suf>[A-ZCAOUAEIOU]+)?\s*$', re.IGNORECASE)
RE_SOLEIRA_SN  = re.compile(r'^\s*(?P<pav>\d)\s*PVSN\b', re.IGNORECASE)
RE_SOLEIRA_NUM_OLD = re.compile(r'^\s*(?P<pav>\d)\s*PV\s*(?P<num>\d+)\s*$', re.IGNORECASE)
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
            fracs = np.linspace(0.25, 0.75, params.setas_por_trecho) if params.setas_por_trecho > 1 else [0.5]
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

    perimetro_limite = getattr(params, "perimetro_levantamento_geom", None)
    try:
        perimetro_limite = perimetro_limite.buffer(0) if perimetro_limite is not None else None
    except Exception:
        pass

    if "type" in df.columns:
        df = df.copy()
        df["type"] = df["type"].fillna("").astype(str).str.strip()

        mask_tvsn = df["type"].str.upper() == "TVSN"
        if mask_tvsn.any():
            df.loc[mask_tvsn, "type"] = "TV"

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
            _log(f"TV em {alvo_pt}"); continue

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
                    try:
                        mt.set_location((x, y, 0.0), rotation=(base_rot if rotacionar_numero_casa else 0.0))
                    except Exception:
                        mt.dxf.insert = (x, y)
                        if rotacionar_numero_casa:
                            mt.dxf.rotation = base_rot
                else:
                    rot_text = base_rot if rotacionar_numero_casa else 0.0
                    add_centered_text(ms, numero_label,
                                      x, y + 0.5*line_gap, params.altura_texto_soleira, style_texto, layer_num_pav, rot_text)
                    add_centered_text(ms, pav_label,
                                      x, y - 0.5*line_gap, params.altura_texto_soleira, style_texto, layer_pav, rot_text)
                continue

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
            nux, nuy = _normal_from_rotation(rot_txt)

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

            if usar_mtext_num_pav and layer_num_pav:
                mt = ms.add_mtext('\\P'.join([numero_label, pav_label]), dxfattribs={'layer': layer_num_pav, 'style': style_texto})
                try:
                    mt.dxf.char_height = params.altura_texto_soleira
                    # centraliza o MTEXT para garantir alinhamento
                    mt.dxf.attachment_point = 5  # Middle Center
                except Exception:
                    pass
                try:
                    mt.set_location((base_x, base_y, 0.0), rotation=(rot_txt if rotacionar_numero_casa else 0.0))
                except Exception:
                    mt.dxf.insert = (base_x, base_y)
                    if rotacionar_numero_casa:
                        mt.dxf.rotation = rot_txt
                # Area embaixo de tudo, com margem segura
                if escrever_area_lote:
                    add_centered_text(
                        ms, f"{lote_poly.area:.2f} m2",
                        base_x + nux * -2.0 * line_gap, base_y + nuy * -2.0 * line_gap,
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
                    add_centered_text(
                        ms, f"{lote_poly.area:.2f} m2",
                        base_x + nux * -1.0 * line_gap, base_y + nuy * -1.0 * line_gap,
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

        if (z_mdt is not None) and bloco != 'ARVORE':
            add_centered_text(
                ms, f"{float(z_mdt):.3f}",
                x, y - 1.0, 0.4, style_texto, associada, rot
            )

    # setas de drenagem (mesma regra de antes)
    inserir_setas_drenagem(ms, doc, via_lines_setas, get_elevation, params)
