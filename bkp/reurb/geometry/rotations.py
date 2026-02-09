"""
Geometry rotations utilities.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon

from reurb.geometry.calculations import clamp


def encontrar_rotacao_por_via(ponto_xy, via_lines, dist_busca=8.0, delta=0.1):
    p = Point(ponto_xy)
    best, best_d = None, float("inf")
    for l in via_lines:
        d = l.distance(p)
        if d < best_d:
            best, best_d = l, d
    if best is None or best_d > dist_busca:
        return 0.0
    proj = best.project(p)
    L = best.length
    proj2 = clamp(proj + delta, 0.0, L)
    a = best.interpolate(proj).coords[0]
    b = best.interpolate(proj2).coords[0]
    dx, dy = (b[0] - a[0]), (b[1] - a[1])
    if dx == 0 and dy == 0:
        return 0.0
    return float(np.degrees(np.arctan2(dy, dx)) % 360.0)


def encontrar_rotacao_por_lote(ponto_xy, lotes, delta=0.1, raio=12.0):
    p = Point(ponto_xy)
    alvo, dmin = None, float("inf")
    for lp in lotes:
        d = lp.exterior.distance(p)
        if d < dmin:
            alvo, dmin = lp, d
    if alvo is None or dmin > raio:
        return None
    exterior = list(alvo.exterior.coords)
    best_seg, best_d = None, float("inf")
    for i in range(len(exterior) - 1):
        a, b = exterior[i], exterior[i + 1]
        seg = LineString([a, b])
        d = seg.distance(p)
        if d < best_d:
            best_d, best_seg = d, seg
    if best_seg is None:
        return None
    proj = best_seg.project(p)
    L = best_seg.length
    proj2 = clamp(proj + delta, 0.0, L)
    a = best_seg.interpolate(proj).coords[0]
    b = best_seg.interpolate(proj2).coords[0]
    dx, dy = (b[0] - a[0]), (b[1] - a[1])
    if dx == 0 and dy == 0:
        return None
    return float(np.degrees(np.arctan2(dy, dx)) % 360.0)


def upright_text_rotation(rot_deg: float) -> float:
    """Mantem texto 'em pe': se 90<r<270, soma 180."""
    r = (rot_deg or 0.0) % 360.0
    if 90.0 < r < 270.0:
        r = (r + 180.0) % 360.0
    return r


def upright_angle_deg(a: float) -> float:
    """Normaliza Angulo para manter o texto 'em pe' (-90..+90)."""
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


def normal_from_rotation(rot_deg: float) -> tuple[float, float]:
    """Normal unitaria a direcao 'rot_deg' (graus)."""
    rad = math.radians(rot_deg or 0.0)
    nx, ny = -math.sin(rad), math.cos(rad)
    nrm = math.hypot(nx, ny) or 1.0
    return (nx / nrm, ny / nrm)


def tangent_angle_pts(p0, p1) -> float:
    dx, dy = (p1[0] - p0[0]), (p1[1] - p0[1])
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def tangent_angle_at_frac(ls: LineString, frac: float, delta_norm: float = 0.01) -> float:
    """
    Angulo (graus) da tangente local do eixo na fracao 'frac' (0..1).
    Usa janela +/- delta_norm para derivar a direcao e devolve 'em pe'.
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
    return upright_angle_deg(ang)


__all__ = [
    "encontrar_rotacao_por_via",
    "encontrar_rotacao_por_lote",
    "upright_text_rotation",
    "upright_angle_deg",
    "normal_from_rotation",
    "tangent_angle_pts",
    "tangent_angle_at_frac",
]
