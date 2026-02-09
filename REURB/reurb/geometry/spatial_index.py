"""
Spatial index helpers.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, List

from shapely.geometry import Point
from shapely.strtree import STRtree


class SpatialIndex:
    """Indice espacial simples para buscas por proximidade."""

    def __init__(self, geometries: List[Any]):
        self.geometries = [g for g in geometries if g is not None]
        self._tree = STRtree(self.geometries) if self.geometries else None
        self._geom_to_idx = {id(g): i for i, g in enumerate(self.geometries)}

    def nearest(self, point: Point, max_distance: float = float("inf")) -> Optional[Tuple[int, Any, float]]:
        if self._tree is None:
            return None
        if max_distance is None or max_distance == float("inf"):
            candidates = self.geometries
        else:
            try:
                candidates = list(self._tree.query(point.buffer(max_distance)))
            except Exception:
                candidates = list(self.geometries)

        if not candidates or len(candidates) == 0:
            return None

        best_idx = None
        best_geom = None
        best_dist = float("inf")
        for geom in candidates:
            try:
                dist = point.distance(geom)
            except Exception:
                continue
            if dist < best_dist and dist <= max_distance:
                best_dist = dist
                best_geom = geom
                best_idx = self._geom_to_idx.get(id(geom))

        return (best_idx, best_geom, best_dist) if best_idx is not None else None

    def query(self, geom, buffer_dist: float | None = None) -> List[Any]:
        """
        Retorna geometrias candidatas que intersectam o envelope de `geom`.
        Se buffer_dist for informado, expande a area de busca.
        """
        if self._tree is None:
            return list(self.geometries)
        try:
            target = geom
            if buffer_dist is not None and buffer_dist > 0:
                target = geom.buffer(buffer_dist)
            return list(self._tree.query(target))
        except Exception:
            return list(self.geometries)


__all__ = ["SpatialIndex"]
