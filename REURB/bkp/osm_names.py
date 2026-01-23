# === osm_names.py ===
from __future__ import annotations
import json
import math
from typing import List, Tuple, Optional, Dict

import requests
from shapely.geometry import LineString, shape, mapping, box
from shapely.ops import linemerge
from pyproj import Transformer


OVERPASS_URL_DEFAULT = "https://overpass-api.de/api/interpreter"


def _flatten_ways(geojson: Dict) -> List[LineString]:
    out: List[LineString] = []
    for feat in geojson.get("features", []):
        try:
            geom = shape(feat["geometry"])
            if geom.is_empty:
                continue
            if geom.geom_type == "LineString":
                out.append(geom)
            elif geom.geom_type == "MultiLineString":
                for g in geom.geoms:
                    out.append(g)
        except Exception:
            continue
    return out


class OSMNameProvider:
    """
    Carrega vias do OSM por bbox (em EPSG local), mantém em CRS local e
    responde 'get(x,y,max_dist_m)' com o nome mais próximo (se dentro do raio).
    """

    def __init__(
        self,
        bbox_local: Tuple[float, float, float, float],
        epsg_local: int,
        overpass_url: Optional[str] = None,
        user_agent: str = "itesp-reurb",
        timeout: int = 120,
        inflate_bbox_m: float = 15.0,           # << padrão 15 m
        expansion_steps_m: Tuple[int, ...] = (0,),  # << sem expansão
        fallback_around_m: float = 15.0,        # << raio local 15 m
        verbose: bool = False,
    ):
        self.bbox_local = bbox_local
        self.epsg_local = int(epsg_local)
        self.overpass_url = overpass_url or OVERPASS_URL_DEFAULT
        self.user_agent = user_agent
        self.timeout = timeout
        self.inflate_bbox_m = float(inflate_bbox_m)
        self.expansion_steps_m = tuple(expansion_steps_m)
        self.fallback_around_m = float(fallback_around_m)
        self.verbose = bool(verbose)

        self._geoms_local: List[Tuple[LineString, str]] = []
        self._transform_to_wgs84 = Transformer.from_crs(self.epsg_local, 4326, always_xy=True)
        self._transform_from_wgs84 = Transformer.from_crs(4326, self.epsg_local, always_xy=True)

    # ------------------------------

    def _bbox_local_to_wgs84(self, bbox_local: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Converte bbox no CRS local para WGS84 (lon/lat)."""
        minx, miny, maxx, maxy = bbox_local
        x0, y0 = self._transform_to_wgs84.transform(minx, miny)
        x1, y1 = self._transform_to_wgs84.transform(maxx, maxy)
        lon_min, lon_max = sorted([x0, x1])
        lat_min, lat_max = sorted([y0, y1])
        return (lon_min, lat_min, lon_max, lat_max)

    def _inflate_local_bbox(self, bbox_local, inflate_m) -> Tuple[float, float, float, float]:
        minx, miny, maxx, maxy = bbox_local
        return (minx - inflate_m, miny - inflate_m, maxx + inflate_m, maxy + inflate_m)

    def _query_overpass(self, bbox_wgs84: Tuple[float, float, float, float]) -> Dict:
        """Puxa ways 'highway' com 'name' no bbox WGS84 em GeoJSON."""
        s, w, n, e = bbox_wgs84[1], bbox_wgs84[0], bbox_wgs84[3], bbox_wgs84[2]  # (S,W,N,E)
        q = f"""
        [out:json][timeout:{self.timeout}];
        way
          ["highway"]
          ["name"]
          ({s},{w},{n},{e});
        out geom;
        """
        headers = {"User-Agent": self.user_agent}
        r = requests.post(self.overpass_url, data=q.encode("utf-8"), headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        # Converte para um GeoJSON simplificado
        features = []
        for el in data.get("elements", []):
            if el.get("type") != "way":
                continue
            name = el.get("tags", {}).get("name")
            if not name:
                continue
            coords = [(nd["lon"], nd["lat"]) for nd in el.get("geometry", [])]
            if len(coords) < 2:
                continue
            features.append({
                "type": "Feature",
                "properties": {"name": name},
                "geometry": {"type": "LineString", "coordinates": coords},
            })
        return {"type": "FeatureCollection", "features": features}

    # ------------------------------

    def build(self):
        """Carrega as vias num raio pequeno ao redor do bbox informado."""
        self._geoms_local.clear()

        # usa apenas um inflate curto (15 m) e sem etapas adicionais
        bbox_loc = self._inflate_local_bbox(self.bbox_local, self.inflate_bbox_m)
        bbox_wgs = self._bbox_local_to_wgs84(bbox_loc)

        try:
            gj = self._query_overpass(bbox_wgs)
        except Exception:
            # fallback: circulo em torno do centro
            cx = (self.bbox_local[0] + self.bbox_local[2]) / 2.0
            cy = (self.bbox_local[1] + self.bbox_local[3]) / 2.0
            bb_fb = (cx - self.fallback_around_m, cy - self.fallback_around_m,
                     cx + self.fallback_around_m, cy + self.fallback_around_m)
            gj = self._query_overpass(self._bbox_local_to_wgs84(bb_fb))

        # reprojeta p/ CRS local
        for feat in gj.get("features", []):
            name = feat.get("properties", {}).get("name")
            try:
                ls_w = shape(feat["geometry"])  # lon/lat
                # reprojeta cada vértice p/ local
                xy = [self._transform_from_wgs84.transform(x, y) for x, y in ls_w.coords]
                ls_l = LineString(xy)
                if ls_l.length > 0:
                    self._geoms_local.append((ls_l, name))
            except Exception:
                continue

        # consolida pequenas quebras
        # (não estritamente necessário, mas ajuda na distância)
        merged: Dict[str, List[LineString]] = {}
        for ls, nm in self._geoms_local:
            merged.setdefault(nm, []).append(ls)
        final: List[Tuple[LineString, str]] = []
        for nm, lst in merged.items():
            try:
                ml = linemerge(lst)
                if ml.geom_type == "LineString":
                    final.append((ml, nm))
                else:
                    for g in ml.geoms:
                        final.append((g, nm))
            except Exception:
                for g in lst:
                    final.append((g, nm))

        self._geoms_local = [(ls, nm) for (ls, nm) in final if not ls.is_empty and ls.length > 0.5]

        if self.verbose:
            print(f"[INFO] OSM: {len(self._geoms_local)} vias carregadas (bbox+{int(self.inflate_bbox_m)}m)")

    # ------------------------------

    def get(self, x: float, y: float, max_dist_m: float = 15.0) -> Optional[str]:
        """Retorna o nome da via mais próxima de (x,y) se a distância ≤ max_dist_m."""
        if not self._geoms_local:
            return None
        p = (float(x), float(y))
        best_nm, best_d = None, 1e30
        for ls, nm in self._geoms_local:
            try:
                d = ls.distance(LineString([p, p]))  # distância ponto-linha
            except Exception:
                continue
            if d < best_d:
                best_d = d; best_nm = nm
        return best_nm if (best_nm and best_d <= max_dist_m) else None

    # útil para debug
    def stats(self) -> str:
        return f"{len(self._geoms_local)} vias carregadas (inflate={self.inflate_bbox_m} m)"
