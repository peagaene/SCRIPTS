"""
Drainage processing.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

import math
import numpy as np

from reurb.config.layers import BLOCO_SETA, LAYER_SETAS_SAIDA
from reurb.geometry.calculations import calcular_offset
from reurb.config.dimensions import Params


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
                px = p1[0] + f * (p2[0] - p1[0])
                py = p1[1] + f * (p2[1] - p1[1])
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
                ms.add_blockref(BLOCO_SETA, final_pt, dxfattribs={"rotation": ang, "layer": LAYER_SETAS_SAIDA})
                if buffer_min > 0.0:
                    setas_pts.append(final_pt)


__all__ = ["inserir_setas_drenagem"]
