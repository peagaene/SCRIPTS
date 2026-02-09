"""
Dimension rendering helpers.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

import math

from reurb.config.dimensions import TOLERANCES
from reurb.utils.logging_utils import REURBLogger

logger = REURBLogger(__name__, verbose=False)

EPS_DIM = float(getattr(TOLERANCES, "DIM_MIN_LENGTH", 1e-4))


def _seg_len(p1, p2) -> float:
    if p1 is None or p2 is None:
        return 0.0
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def add_dim_aligned(ms, p1, p2, offset_sign, offset_mag, layer, text_height, arrow_size, dim_decimals_or_flag, style_texto):
    """Cria dimensao alinhada robusta (pula segmentos degenerados e protege render)."""
    if p1 is None or p2 is None:
        return None
    if _seg_len(p1, p2) < EPS_DIM:
        return None
    try:
        offset_mag = float(offset_mag)
    except Exception as e:
        logger.warning(f"Offset invalido: {e}")
        offset_mag = 0.0
    if not offset_mag or abs(offset_mag) < 1e-6:
        offset_mag = max(float(text_height) if text_height else 0.01, 0.01)
    try:
        dimdec = int(dim_decimals_or_flag) if isinstance(dim_decimals_or_flag, (int, float)) else 2
    except Exception as e:
        logger.warning(f"Dimdec invalido: {e}")
        dimdec = 2

    overrides = {
        "dimtxt": float(text_height) if text_height is not None else 0.25,
        "dimasz": float(arrow_size) if arrow_size is not None else 0.18,
        "dimtxsty": style_texto,
        "dimdec": dimdec,
    }
    try:
        dim = ms.add_aligned_dim(
            p1=p1,
            p2=p2,
            distance=float(offset_sign) * float(offset_mag),
            dxfattribs={"layer": layer},
            override=overrides,
        )
    except Exception as e:
        logger.warning(f"Falha ao criar dimensao: {e}")
        return None

    try:
        dim.render()
    except Exception as e:
        logger.warning(f"Falha ao renderizar dimensao: {e}")
        return None

    return dim


def add_ordinate(ms, midpoint, texto, layer, params, style: str):
    dim = ms.add_ordinate_dim(
        midpoint,
        (0.5, -1.0),
        0,
        origin=midpoint,
        text=texto,
        rotation=0,
        dxfattribs={"layer": layer},
        override={
            "dimtxt": params.dimtxt_ordinate,
            "dimasz": params.dimasz_ordinate,
            "dimtsz": 0,
            "dimtad": 0,
            "dimtxsty": style,
        },
    )
    dim.render()


__all__ = ["add_dim_aligned", "add_ordinate", "EPS_DIM"]
