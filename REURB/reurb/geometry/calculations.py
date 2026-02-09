"""
Geometry calculations (pure functions).
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

import math
import numpy as np


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def calcular_offset(p1, p2, dist=0.5):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = np.hypot(dx, dy)
    if L == 0:
        return (0.0, 0.0)
    ndx, ndy = dx / L, dy / L
    nx, ny = -ndy, ndx
    return (nx * dist, ny * dist)


def dist(p1, p2) -> float:
    return float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def azimute(p1, p2) -> float:
    dE = p2[0] - p1[0]
    dN = p2[1] - p1[1]
    ang = math.degrees(math.atan2(dE, dN))
    if ang < 0:
        ang += 360.0
    return float(ang)


def bbox_max(vertices):
    xs = [p[0] for p in vertices]
    ys = [p[1] for p in vertices]
    return max(xs), max(ys)
