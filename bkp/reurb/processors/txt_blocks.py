"""
TXT blocks processing.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

import re
import math
import numpy as np
import pandas as pd

from shapely.geometry import Point

from reurb.config.dimensions import Params
from reurb.config.layers import (
    STYLE_TEXTO,
    BLOCO_SETA,
    BLOCO_SOLEIRA_POS,
    LAYER_SETAS_SAIDA,
    LAYER_SOLEIRA_BLOCO,
    LAYER_SOLEIRA_NUM_PAV,
    LAYER_SOLEIRA_AREA,
)
from reurb.config.mappings import TYPE_TO_LAYER, LAYER_TO_BLOCK
from reurb.geometry.rotations import encontrar_rotacao_por_via, encontrar_rotacao_por_lote, upright_text_rotation, normal_from_rotation
from reurb.geometry.segments import segmentos_ordenados_por_proximidade, segment_vec_by_index
from reurb.renderers.text_renderer import add_centered_text, place_mtext_middle_center
from reurb.processors.drainage import inserir_setas_drenagem


RE_SOLEIRA_NUM = re.compile(r"^\s*(?P<pav>\d)\s*PV\s*(?P<num>[0-9A-Z]+)\s*$", re.IGNORECASE)
RE_SOLEIRA_SN = re.compile(r"^\s*(?P<pav>\d)\s*PVSN\b", re.IGNORECASE)
RE_SOLEIRA_NUM_OLD = re.compile(r"^\s*(?P<pav>\d)\s*PV\s*(?P<num>[0-9A-Z]+)\s*$", re.IGNORECASE)
RE_SOLEIRA_SN_OLD = re.compile(r"^\s*YPVSN\b", re.IGNORECASE)
RE_E_PATTERN = re.compile(r"^E\s*\d+\s*--\s*([0-9A-Za-z]+|SN)\s*--\s*([0-9O])\s*--\s*([A-Za-z]+)\s*$", flags=re.IGNORECASE)
RE_ARV_ALIAS = re.compile(r"^ARV[0-9A-Z]*$", flags=re.IGNORECASE)


def _log(msg: str):
    pass


def _point_in_limit(limite, x, y=None) -> bool:
    if limite is None:
        return True
    try:
        if y is None:
            if hasattr(x, "x") and hasattr(x, "y"):
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


def _lote_mais_proximo(lotes, ponto, buffer_lote):
    lote_idx, lote_poly = None, None
    min_d = float("inf")
    for i, lp in enumerate(lotes):
        d = 0.0 if lp.buffer(buffer_lote).contains(ponto) else lp.distance(ponto)
        if d < min_d:
            min_d, lote_idx, lote_poly = d, i, lp
    return lote_idx, lote_poly, min_d


def _maior_edificacao_no_lote(edificacoes, lote_poly):
    best = None
    best_area = 0.0
    for e in edificacoes:
        try:
            inter = e.intersection(lote_poly)
            if inter.is_empty:
                continue
            if hasattr(inter, "geoms"):
                for g in inter.geoms:
                    a = g.area
                    if a > best_area:
                        best_area, best = a, g
            else:
                a = inter.area
                if a > best_area:
                    best_area, best = a, inter
        except Exception:
            continue
    return best


def processar_registros(df: pd.DataFrame, ms, doc, params: Params, lotes, edificacoes, via_lines_setas, via_lines_geral, get_elevation):
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

        mask_pvsnm = df["type"].str.match(r"^\s*\d\s*PVSNM\s*$", flags=re.IGNORECASE)
        if mask_pvsnm.any():
            df.loc[mask_pvsnm, "type"] = df.loc[mask_pvsnm, "type"].str.replace(r"^\s*(\d)\s*PVSNM\s*$", r"\1 PVSN", regex=True)

        mask_pv_m = df["type"].str.match(r"^\s*\d\s*PV[0-9A-Za-z]+M\s*$", flags=re.IGNORECASE)
        if mask_pv_m.any():
            df.loc[mask_pv_m, "type"] = df.loc[mask_pv_m, "type"].str.replace(r"^\s*(\d)\s*PV\s*([0-9A-Za-z]+)M\s*$", r"\1 PV \2", regex=True)

        mask_pvsn_num_m = df["type"].str.match(r"^\s*\d\s*PVSN[0-9A-Za-z]+M\s*$", flags=re.IGNORECASE)
        if mask_pvsn_num_m.any():
            df.loc[mask_pvsn_num_m, "type"] = df.loc[mask_pvsn_num_m, "type"].str.replace(r"^\s*(\d)\s*PVSN\s*([0-9A-Za-z]+)M\s*$", r"\1 PV \2", regex=True)

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
        tipo_raw = str(r["type"]).strip()
        tipo = tipo_raw.upper()
        x, y = float(r["E"]), float(r["N"])
        z_mdt = get_elevation(x, y)

        if tipo == "TV":
            p = Point(x, y)
            lote_idx, lote_poly, min_d = _lote_mais_proximo(lotes, p, params.buffer_lote_edif)
            alvo_pt = (x, y)
            if lote_poly:
                edif_sel = _maior_edificacao_no_lote(edificacoes, lote_poly)
                if edif_sel is not None:
                    alvo_pt = edif_sel.representative_point().coords[0]
                else:
                    alvo_pt = lote_poly.representative_point().coords[0]
            add_centered_text(ms, "TV", alvo_pt[0], alvo_pt[1], params.altura_texto_soleira, style_texto, LAYER_SOLEIRA_NUM_PAV)
            continue

        pav = None
        sn = False
        m_sn = RE_SOLEIRA_SN.search(tipo)
        if m_sn:
            pav = int(m_sn.group("pav"))
            sn = True
        elif RE_SOLEIRA_SN_OLD.search(tipo):
            pav = 1
            sn = True

        m_num = RE_SOLEIRA_NUM.search(tipo) or RE_SOLEIRA_NUM_OLD.search(tipo)
        is_soleira = bool(sn or m_num)
        if is_soleira:
            if pav is None:
                try:
                    pav = int(m_num.group("pav"))
                except Exception:
                    pav = 1

            if sn:
                numero_str = None
            else:
                gi = getattr(m_num.re, "groupindex", {})
                num_val = m_num.group("num") if ("num" in gi) else ""
                suf_val = m_num.group("suf") if ("suf" in gi) else ""
                num_val = (num_val or "").strip()
                suf_val = (suf_val or "").strip()
                numero_str = (f"{num_val}{suf_val}".upper() if (num_val or suf_val) else None)

            p_txt = Point(x, y)
            lote_idx, lote_poly, min_d = _lote_mais_proximo(lotes, p_txt, params.buffer_lote_edif)
            lote_key = _lote_key(lote_idx, lote_poly)
            if lote_key is not None and lote_key in lotes_com_numero:
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
                ms.add_blockref(BLOCO_SOLEIRA_POS, (x, y), dxfattribs={"rotation": 0.0, "layer": LAYER_SOLEIRA_BLOCO})
                base_rot = 0.0
                permitir_texto = _point_in_limit(perimetro_limite, x, y)
                if not permitir_texto:
                    continue
                if usar_mtext_num_pav and layer_num_pav:
                    mt = ms.add_mtext("\\P".join([numero_label, pav_label]), dxfattribs={"layer": layer_num_pav, "style": style_texto})
                    try:
                        mt.dxf.char_height = params.altura_texto_soleira
                    except Exception:
                        pass
                    place_mtext_middle_center(mt, x, y, (base_rot if rotacionar_numero_casa else 0.0))
                else:
                    rot_text = base_rot if rotacionar_numero_casa else 0.0
                    add_centered_text(ms, numero_label, x, y + 0.5 * line_gap, params.altura_texto_soleira, style_texto, layer_num_pav, rot_text)
                    add_centered_text(ms, pav_label, x, y - 0.5 * line_gap, params.altura_texto_soleira, style_texto, layer_pav, rot_text)
                continue
            if lote_key is not None:
                lotes_com_numero.add(lote_key)

            ordenados = segmentos_ordenados_por_proximidade(lote_poly, (x, y))
            usados = segmentos_usados_por_lote.setdefault(lote_idx, set())
            pos_soleira = (x, y)
            seg_escolhido = None
            for idx_seg, _, proj_xy, _ in ordenados:
                if idx_seg not in usados:
                    seg_escolhido = idx_seg
                    pos_soleira = proj_xy
                    break
            if seg_escolhido is None and ordenados:
                seg_escolhido, _, pos_soleira, _ = ordenados[0]
            usados.add(seg_escolhido)

            ms.add_blockref(BLOCO_SOLEIRA_POS, pos_soleira, dxfattribs={"rotation": 0.0, "layer": LAYER_SOLEIRA_BLOCO})

            rot_txt = encontrar_rotacao_por_via(pos_soleira, via_lines_geral, params.dist_busca_rot, params.delta_interp)
            if not rot_txt:
                vx, vy = segment_vec_by_index(lote_poly, seg_escolhido)
                rot_txt = math.degrees(math.atan2(vy, vx))
            rot_txt = upright_text_rotation(rot_txt)

            e_sel = _maior_edificacao_no_lote(edificacoes, lote_poly)
            if e_sel is not None:
                tx_pt = e_sel.representative_point().coords[0]
            else:
                tx_pt = lote_poly.representative_point().coords[0]
            base_x, base_y = float(tx_pt[0]), float(tx_pt[1])

            permitir_texto = _point_in_limit(perimetro_limite, pos_soleira) and _point_in_limit(perimetro_limite, base_x, base_y)
            if not permitir_texto:
                continue

            rot_text = rot_txt if rotacionar_numero_casa else 0.0
            nux, nuy = normal_from_rotation(rot_text)

            if usar_mtext_num_pav and layer_num_pav:
                mt = ms.add_mtext("\\P".join([numero_label, pav_label]), dxfattribs={"layer": layer_num_pav, "style": style_texto})
                try:
                    mt.dxf.char_height = params.altura_texto_soleira
                except Exception:
                    pass
                place_mtext_middle_center(mt, base_x, base_y, (rot_txt if rotacionar_numero_casa else 0.0))
                if escrever_area_lote:
                    area_offset = _calc_area_offset(2)
                    add_centered_text(
                        ms,
                        f"{lote_poly.area:.2f} m2",
                        base_x + nux * -area_offset,
                        base_y + nuy * -area_offset,
                        params.altura_texto_area,
                        style_texto,
                        LAYER_SOLEIRA_AREA,
                        rot_text,
                    )
            else:
                add_centered_text(ms, numero_label, base_x + nux * +1.0 * line_gap, base_y + nuy * +1.0 * line_gap, params.altura_texto_soleira, style_texto, layer_num_pav, rot_text)
                add_centered_text(ms, pav_label, base_x, base_y, params.altura_texto_soleira, style_texto, layer_pav, rot_text)
                if escrever_area_lote:
                    area_offset = _calc_area_offset(2)
                    add_centered_text(
                        ms,
                        f"{lote_poly.area:.2f} m2",
                        base_x + nux * -area_offset,
                        base_y + nuy * -area_offset,
                        params.altura_texto_area,
                        style_texto,
                        LAYER_SOLEIRA_AREA,
                        rot_text,
                    )
            continue

        associada = type_to_layer.get(tipo)
        if not associada:
            continue
        bloco = layer_to_block.get(associada)
        if not bloco:
            continue

        rot = encontrar_rotacao_por_via((x, y), via_lines_geral, params.dist_busca_rot, params.delta_interp)
        if not rot:
            rot_fb = encontrar_rotacao_por_lote((x, y), lotes, params.delta_interp, raio=max(params.dist_busca_rot * 2, 8.0))
            if rot_fb is not None:
                rot = rot_fb
        rot = upright_text_rotation(rot or 0.0)

        if bloco in {"INFRA_PVE", "INFRA_PVAP"}:
            rot = 0.0
        if "BOCA" in bloco:
            import re as _re
            try:
                qtd = int(_re.findall(r"(\d+)$", tipo)[-1])
            except Exception:
                qtd = 1
            dx = 1.0 * math.cos(math.radians(rot))
            dy = 1.0 * math.sin(math.radians(rot))
            for i in range(qtd):
                xi, yi = x + i * dx, y + i * dy
                ms.add_blockref(bloco, (xi, yi), dxfattribs={"rotation": rot, "layer": associada})
        else:
            ms.add_blockref(bloco, (x, y), dxfattribs={"rotation": rot, "layer": associada})

        if bloco == "INFRA_PVE":
            add_centered_text(ms, "PVE", x, y + 0.7, 0.4, style_texto, associada, rot)
        elif bloco == "INFRA_PVAP":
            add_centered_text(ms, "PVAP", x, y + 0.7, 0.4, style_texto, associada, rot)

        if z_mdt is not None:
            add_centered_text(ms, f"{float(z_mdt):.3f}", x, y - 1.0, 0.4, style_texto, associada, rot)

    inserir_setas_drenagem(ms, doc, via_lines_setas, get_elevation, params)


__all__ = ["processar_registros"]
