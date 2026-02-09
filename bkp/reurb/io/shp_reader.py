"""
SHP reader utilities.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

from typing import List, Tuple, Optional

from osgeo import ogr, osr
from shapely.geometry import LineString, Point
from shapely.ops import linemerge


def _pick_name_field(layer: ogr.Layer) -> Optional[str]:
    cand_subs = ("name", "nome", "logr", "logradouro", "nm_")

    def norm(s: str) -> str:
        return s.strip().lower()

    for i in range(layer.GetLayerDefn().GetFieldCount()):
        fld = layer.GetLayerDefn().GetFieldDefn(i)
        fn = norm(fld.GetName())
        if any(sub in fn for sub in cand_subs):
            return fld.GetName()
    return None


class ShpNameProvider:
    def __init__(self, lines: List[Tuple[LineString, str]]):
        self._geoms = [(ls, nm) for (ls, nm) in lines if (ls is not None and not ls.is_empty and nm)]

    def get(self, x: float, y: float, max_dist_m: float = 15.0) -> Optional[str]:
        if not self._geoms:
            return None
        p = Point(float(x), float(y))
        best_nm, best_d = None, 1e30
        for ls, nm in self._geoms:
            try:
                d = ls.distance(p)
            except Exception:
                continue
            if d < best_d:
                best_d = d
                best_nm = nm
        return best_nm if (best_nm and best_d <= max_dist_m) else None

    def stats(self) -> str:
        return f"{len(self._geoms)} vias carregadas do SHP"


def build_shp_name_provider(path_shp: str, epsg_local: int) -> Optional[ShpNameProvider]:
    try:
        ds = ogr.Open(path_shp)
        if ds is None:
            return None
        lyr = ds.GetLayer(0)
        if lyr is None:
            return None
        srs_src = lyr.GetSpatialRef()
        srs_dst = osr.SpatialReference()
        srs_dst.ImportFromEPSG(int(epsg_local))
        transform = None
        if srs_src is not None and not srs_src.IsSame(srs_dst):
            transform = ogr.osr.CoordinateTransformation(srs_src, srs_dst)  # type: ignore[attr-defined]
        name_field = _pick_name_field(lyr)
        if not name_field:
            return None

        lines: List[Tuple[LineString, str]] = []
        feat = lyr.GetNextFeature()
        while feat:
            nm = feat.GetField(name_field)
            if nm:
                geom = feat.GetGeometryRef()
                if geom is not None:
                    try:
                        geom_l = geom.Clone()
                        if transform is not None:
                            geom_l.Transform(transform)
                        t = geom_l.GetGeometryType()
                        if t == ogr.wkbLineString or t == ogr.wkbLineString25D:
                            pts = [(geom_l.GetPoint(i)[0], geom_l.GetPoint(i)[1]) for i in range(geom_l.GetPointCount())]
                            if len(pts) >= 2:
                                lines.append((LineString(pts), str(nm)))
                        elif t == ogr.wkbMultiLineString:
                            for i in range(geom_l.GetGeometryCount()):
                                g = geom_l.GetGeometryRef(i)
                                pts = [(g.GetPoint(j)[0], g.GetPoint(j)[1]) for j in range(g.GetPointCount())]
                                if len(pts) >= 2:
                                    lines.append((LineString(pts), str(nm)))
                    except Exception:
                        pass
            feat = lyr.GetNextFeature()

        if not lines:
            return None

        by_name: dict[str, List[LineString]] = {}
        for ls, nm in lines:
            by_name.setdefault(nm, []).append(ls)
        merged: List[Tuple[LineString, str]] = []
        for nm, lst in by_name.items():
            try:
                ml = linemerge(lst)
                if ml is None:
                    continue
                if ml.geom_type == "LineString":
                    merged.append((ml, nm))
                else:
                    for g in ml.geoms:
                        merged.append((g, nm))
            except Exception:
                for g in lst:
                    merged.append((g, nm))

        return ShpNameProvider(merged)
    except Exception:
        return None


__all__ = ["ShpNameProvider", "build_shp_name_provider"]
