# centerline_mrr.py
# Uso:
#   python centerline_mrr.py IN_POLY.shp OUT_LINE.shp --epsg 31983
#
# Faz uma centerline rápida por polígono usando:
# - minimum_rotated_rectangle -> direção do eixo longo
# - reta pelo centróide paralela ao eixo longo
# - interseção com o polígono -> maior segmento

import argparse
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union

def longest_dir_from_rotated_rect(rect: Polygon):
    """Retorna vetor unitário (ux, uy) na direção do lado mais longo do 'minimum rotated rectangle'."""
    coords = list(rect.exterior.coords)
    # coords tem 5 pontos (fecha no início). Pegamos arestas 0-1,1-2,2-3,3-4
    best_len2 = -1.0
    best_vec = (1.0, 0.0)
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        vx, vy = (x2 - x1), (y2 - y1)
        l2 = vx*vx + vy*vy
        if l2 > best_len2:
            best_len2 = l2
            best_vec = (vx, vy)
    L = (best_vec[0]**2 + best_vec[1]**2) ** 0.5
    if L == 0:
        return (1.0, 0.0)
    return (best_vec[0]/L, best_vec[1]/L)

def centerline_for_polygon(poly: Polygon):
    """Gera uma única centerline para um polígono."""
    if poly.is_empty:
        return None
    # 1) retângulo mínimo orientado
    rect = poly.minimum_rotated_rectangle
    # 2) direção do lado mais longo
    ux, uy = longest_dir_from_rotated_rect(rect)
    # 3) reta longa pelo centróide
    c = poly.centroid
    # comprimento grande o bastante (duas vezes a diagonal do retângulo mínimo)
    minx, miny, maxx, maxy = rect.bounds
    diag = ((maxx - minx)**2 + (maxy - miny)**2) ** 0.5
    L = diag * 2.0
    p1 = (c.x - ux*L, c.y - uy*L)
    p2 = (c.x + ux*L, c.y + uy*L)
    infinite_line = LineString([p1, p2])
    # 4) interseção com o polígono
    inter = poly.intersection(infinite_line)
    # pode retornar LineString, MultiLineString, GeometryCollection...
    # vamos reduzir ao maior LineString
    def pick_longest(ls_or_multi):
        if ls_or_multi.is_empty:
            return None
        if isinstance(ls_or_multi, LineString):
            return ls_or_multi
        # tentar iterar geometrias
        try:
            parts = [g for g in ls_or_multi.geoms if isinstance(g, LineString)]
        except Exception:
            parts = []
        if not parts:
            return None
        return max(parts, key=lambda g: g.length)
    return pick_longest(inter)

def main(in_path, out_path, epsg_target):
    gdf = gpd.read_file(in_path)
    if gdf.crs is None:
        raise ValueError("A camada de entrada não tem CRS. Defina um CRS antes (gdf.set_crs).")
    if epsg_target:
        gdf = gdf.to_crs(epsg=epsg_target)

    out_rows = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            out_rows.append(None)
            continue
        # tratar MultiPolygon: processar cada parte e pegar o maior segmento resultante
        if isinstance(geom, MultiPolygon):
            segs = []
            for p in geom.geoms:
                cl = centerline_for_polygon(p)
                if cl is not None:
                    segs.append(cl)
            if segs:
                longest = max(segs, key=lambda s: s.length)
                out_rows.append(longest)
            else:
                out_rows.append(None)
        elif isinstance(geom, Polygon):
            out_rows.append(centerline_for_polygon(geom))
        else:
            out_rows.append(None)

    out = gdf.copy()
    out.geometry = out_rows
    out = out[out.geometry.notna() & ~out.geometry.is_empty]
    out.to_file(out_path)
    print(f"✅ Centerlines salvas em: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Centerline rápida por polígono (Shapely/GeoPandas, sem raster).")
    ap.add_argument("input", help="Shapefile/GeoPackage de polígonos")
    ap.add_argument("output", help="Shapefile/GeoPackage de saída (linhas)")
    ap.add_argument("--epsg", type=int, default=31983, help="EPSG alvo (default 31983)")
    args = ap.parse_args()
    main(args.input, args.output, args.epsg)
