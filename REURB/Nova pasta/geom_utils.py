# geom_utils.py
import numpy as np
from shapely.geometry import Point, LineString, Polygon

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
