"""
Geometry segment utilities.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

import math
from typing import List

from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union, snap

from reurb.config.dimensions import TOLERANCES

DEDUP_ROUND_SCALE = int(getattr(TOLERANCES, "DEDUP_ROUND_SCALE", 1000))
MERGE_TOL = float(getattr(TOLERANCES, "MERGE_TOLERANCE", 0.01))
ANG_TOL = float(getattr(TOLERANCES, "ANGLE_TOLERANCE_DEG", 5.0))
KEY_TOL_MIN = float(getattr(TOLERANCES, "KEY_TOLERANCE_MIN", 1e-3))
MAX_MERGE_ITERS = int(getattr(TOLERANCES, "MAX_MERGE_ITERS", 5))


def segmentos_ordenados_por_proximidade(poly: Polygon, ponto_xy):
    """Retorna segmentos do poligono ordenados por proximidade ao ponto."""
    p = Point(ponto_xy)
    exterior = list(poly.exterior.coords)
    ordenados = []
    for i in range(len(exterior) - 1):
        a, b = exterior[i], exterior[i + 1]
        seg = LineString([a, b])
        d = seg.distance(p)
        proj_xy = seg.interpolate(seg.project(p)).coords[0]
        ordenados.append((i, d, proj_xy, seg))
    ordenados.sort(key=lambda t: t[1])
    return ordenados


def segment_vec_by_index(poly: Polygon, idx_seg: int) -> tuple[float, float]:
    """Retorna vetor do segmento 'idx_seg' do anel exterior do poligono."""
    coords = list(poly.exterior.coords)
    i2 = (idx_seg + 1) % (len(coords) - 1)
    p1, p2 = coords[idx_seg], coords[i2]
    return (p2[0] - p1[0], p2[1] - p1[1])


def _noded_lote_segments(lotes: list, snap_tol: float = 0.05) -> list[LineString]:
    if not lotes:
        return []
    try:
        boundaries = [p.boundary for p in lotes if p is not None and not p.is_empty]
        union_lines = unary_union(boundaries)
        if snap_tol:
            union_lines = snap(union_lines, union_lines, snap_tol)
        noded = unary_union(union_lines)
    except Exception:
        return []

    segs: list[LineString] = []

    def _add_segs_from_line(line: LineString):
        try:
            coords = list(line.coords)
        except Exception:
            return
        for i in range(len(coords) - 1):
            a, b = coords[i], coords[i + 1]
            if a != b:
                segs.append(LineString([a, b]))

    if hasattr(noded, "geoms"):
        for g in noded.geoms:
            if isinstance(g, LineString):
                _add_segs_from_line(g)
    elif isinstance(noded, LineString):
        _add_segs_from_line(noded)

    # dedup by rounded endpoints
    seen = set()
    unique: list[LineString] = []
    for s in segs:
        try:
            a, b = list(s.coords)[0], list(s.coords)[-1]
        except Exception:
            continue
        ax, ay = int(round(a[0] * DEDUP_ROUND_SCALE)), int(round(a[1] * DEDUP_ROUND_SCALE))
        bx, by = int(round(b[0] * DEDUP_ROUND_SCALE)), int(round(b[1] * DEDUP_ROUND_SCALE))
        key = (ax, ay, bx, by) if (ax, ay) <= (bx, by) else (bx, by, ax, ay)
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)

    # merge segmentos muito pequenos (<= 1 cm) ao segmento alinhado adjacente
    def _ang_deg(a, b) -> float:
        return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0

    def _ang_diff(a, b) -> float:
        d = abs(a - b)
        return d if d <= 90.0 else 180.0 - d

    def _pt_key(p, tol):
        return (round(p[0] / tol) * tol, round(p[1] / tol) * tol)

    merge_tol = MERGE_TOL
    ang_tol = ANG_TOL
    key_tol = max(KEY_TOL_MIN, float(snap_tol) if snap_tol else KEY_TOL_MIN)

    changed = True
    it = 0
    segs = list(unique)
    while changed and it < MAX_MERGE_ITERS:
        it += 1
        changed = False
        endpoints = {}
        for idx, s in enumerate(segs):
            try:
                a, b = list(s.coords)[0], list(s.coords)[-1]
            except Exception:
                continue
            ka = _pt_key(a, key_tol)
            kb = _pt_key(b, key_tol)
            endpoints.setdefault(ka, []).append((idx, 0))
            endpoints.setdefault(kb, []).append((idx, 1))

        to_remove = set()
        to_add: list[LineString] = []

        for i, s in enumerate(segs):
            if i in to_remove:
                continue
            try:
                a, b = list(s.coords)[0], list(s.coords)[-1]
            except Exception:
                continue
            if math.hypot(b[0] - a[0], b[1] - a[1]) > merge_tol:
                continue
            ang_s = _ang_deg(a, b)
            ka = _pt_key(a, key_tol)
            kb = _pt_key(b, key_tol)

            neigh_a = [(idx, end) for (idx, end) in endpoints.get(ka, []) if idx != i and idx not in to_remove]
            neigh_b = [(idx, end) for (idx, end) in endpoints.get(kb, []) if idx != i and idx not in to_remove]

            def _neighbor_far(idx, end):
                na, nb = list(segs[idx].coords)[0], list(segs[idx].coords)[-1]
                return nb if end == 0 else na

            cand_a = []
            for idx, end in neigh_a:
                na, nb = list(segs[idx].coords)[0], list(segs[idx].coords)[-1]
                if _ang_diff(ang_s, _ang_deg(na, nb)) <= ang_tol:
                    cand_a.append((idx, end))
            cand_b = []
            for idx, end in neigh_b:
                na, nb = list(segs[idx].coords)[0], list(segs[idx].coords)[-1]
                if _ang_diff(ang_s, _ang_deg(na, nb)) <= ang_tol:
                    cand_b.append((idx, end))

            if cand_a and cand_b:
                ia, enda = cand_a[0]
                ib, endb = cand_b[0]
                fa = _neighbor_far(ia, enda)
                fb = _neighbor_far(ib, endb)
                if fa != fb:
                    to_remove.update({i, ia, ib})
                    to_add.append(LineString([fa, fb]))
                    changed = True
                    continue
            elif cand_a:
                ia, enda = cand_a[0]
                fa = _neighbor_far(ia, enda)
                if fa != b:
                    to_remove.update({i, ia})
                    to_add.append(LineString([fa, b]))
                    changed = True
                    continue
            elif cand_b:
                ib, endb = cand_b[0]
                fb = _neighbor_far(ib, endb)
                if fb != a:
                    to_remove.update({i, ib})
                    to_add.append(LineString([a, fb]))
                    changed = True
                    continue

        if changed:
            segs = [s for idx, s in enumerate(segs) if idx not in to_remove]
            segs.extend(to_add)

    return segs


__all__ = [
    "segmentos_ordenados_por_proximidade",
    "segment_vec_by_index",
    "_noded_lote_segments",
]
