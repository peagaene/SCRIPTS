"""
Symbology profiles.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

import json
import os
from typing import Dict

SEM_KEYS = [
    "via_nome",
    "via_med",
    "per_tab",
    "per_vert",
    "curva_i",
    "curva_m",
    "curva_txt",
    "drenagem",
    "txt_grande",
    "txt_soleira",
]

DEFAULTS: Dict[str, str] = {
    "via_nome": "TOP_SISTVIA",
    "via_med": "TOP_COTAS_VIARIO",
    "per_tab": "TOP_TABELA",
    "per_vert": "HM_VERTICE",
    "curva_i": "HM_CURVA_NIV_INTERM_LIN",
    "curva_m": "HM_CURVA_NIV_MESTRA_LIN",
    "curva_txt": "TOP_CURVA_NIV",
    "drenagem": "TOP_DRENAGEM",
    "txt_grande": "TOP_TEXTO",
    "txt_soleira": "TOP_COTA",
}

PROFILE_REURB: Dict[str, str] = {
    "via_nome": "TOP_SISTVIA",
    "via_med": "TOP_COTAS_VIARIO",
    "per_tab": "TOP_TABELA",
    "per_vert": "HM_VERTICE",
    "curva_i": "HM_CURVA_NIV_INTERM_LIN",
    "curva_m": "HM_CURVA_NIV_MESTRA_LIN",
    "curva_txt": "TOP_CURVA_NIV",
    "drenagem": "TOP_DRENAGEM",
    "txt_grande": "TOP_TEXTO",
    "txt_soleira": "TOP_COTA",
    "curva_i_reurb": "HM_CURVA_NIV_INTERM_LIN",
    "curva_m_reurb": "HM_CURVA_NIV_MESTRA_LIN",
}

PROFILE_CDHU: Dict[str, str] = {
    "per_tab": "Top-Tabela",
    "per_vert": "Top-Poligonal",
    "via_med": "Top-Txt-Pequeno",
    "via_nome": "Top-Txt-Grande",
    "drenagem": "Top-Agua",
    "curva_i": "Top-Curva1",
    "curva_m": "Top-Curva5",
    "curva_txt": "Top-Curva5",
    "txt_grande": "Top-Txt-Grande",
    "txt_soleira": "Top-Cota",
}


def load_profile_sidecar(simb_path: str) -> dict | None:
    base, _ = os.path.splitext(simb_path)
    sidecar = base + ".layers.json"
    if not os.path.isfile(sidecar):
        return None
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            data = json.load(f)
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

    side = load_profile_sidecar(simb_path)
    if side:
        prof.update(side)
        return prof

    bn = os.path.basename(simb_path).lower()
    if "simbol" in bn and "cdhu" in bn:
        prof.update(PROFILE_CDHU)
        return prof

    if "simbol" in bn and "reurb" in bn:
        prof.update(PROFILE_REURB)
        return prof

    return prof


__all__ = ["build_layer_profile", "load_profile_sidecar"]
