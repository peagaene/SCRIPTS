"""
Text rendering helpers.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

from reurb.utils.logging_utils import REURBLogger

logger = REURBLogger(__name__, verbose=False)


def add_centered_text(ms, content: str, x: float, y: float, height: float, style: str, layer: str, rot: float | None = None):
    t = ms.add_text(content, dxfattribs={"height": height, "style": style, "layer": layer})
    t.dxf.insert = (x, y)
    t.dxf.align_point = (x, y)
    t.dxf.halign = 1  # center
    t.dxf.valign = 2  # middle
    if rot is not None:
        t.dxf.rotation = float(rot)
    return t


def place_mtext_middle_center(mt, x: float, y: float, rot: float | None = None):
    try:
        mt.dxf.attachment_point = 5  # Middle Center
    except Exception as e:
        logger.warning(f"Falha ao definir attachment_point: {e}")
    try:
        mt.set_location((x, y, 0.0), rotation=(float(rot) if rot is not None else 0.0))
    except Exception:
        try:
            mt.dxf.insert = (x, y)
            if rot is not None:
                mt.dxf.rotation = float(rot)
        except Exception as e:
            logger.warning(f"Falha ao posicionar MText: {e}")
    return mt


def place_text_center(ms, content: str, x: float, y: float, height: float, rotation_deg: float, layer: str, style: str):
    t = ms.add_text(content, dxfattribs={"height": height, "style": style, "layer": layer})
    t.dxf.rotation = float(rotation_deg)
    t.dxf.insert = (x, y)
    t.dxf.align_point = (x, y)
    t.dxf.halign = 1  # center
    t.dxf.valign = 2  # middle
    return t


__all__ = [
    "add_centered_text",
    "place_mtext_middle_center",
    "place_text_center",
]
