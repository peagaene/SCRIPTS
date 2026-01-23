# numerar_lotes_clockwise_gui.py
# --------------------------------
# Numeração de lotes por QUADRA:
# • começa no sudoeste
# • perímetro por ordem ao longo do contorno (linear referencing)
# • miolo por heading + camada (distância ao contorno)
# • conversão MultiPolygon/GeometryCollection → Polygon (maior por área)
# • fallbacks para nunca travar
# --------------------------------

import os, re
import numpy as np
import geopandas as gpd
from tkinter import Tk, filedialog, messagebox
from collections import defaultdict
from shapely.ops import linemerge
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

# ========= CONFIG PADRÃO (edite aqui) =========
TARGET_EPSG = 31983              # reprojeta se o dado estiver em lat/long
QUADRA_COL  = "QUADRA"
LOTE_COL    = "LOTE"

ADJ_BUFFER_DEFAULT            = 0.05   # fecha micro-fendas para detectar vizinhos (unidade = CRS)
MIN_SHARED_LEN_RATIO_DEFAULT  = 0.02   # fração do perímetro do menor lote (ex.: 0.02 = 2%)
PERIMETER_FIRST_DEFAULT       = True   # numera primeiro os lotes que tocam o contorno
PERIM_TOUCH_TOL_DEFAULT       = 0.05   # tolerância p/ "tocar contorno" (unidade = CRS)

# Limites de giro (graus)
MAX_CW_DEG_PERIM              = 60
MAX_CW_DEG_INTERIOR           = 120
# ==============================================

# ---------- Utilidades de caminho ----------
def _fix_windows_path(p: str) -> str:
    if not p: return p
    if p.startswith(r"\\"):  # UNC
        return os.path.normpath(p)
    p2 = p.replace("/", "\\")
    m = re.match(r"^\\([a-zA-Z])\\(.*)$", p2)  # ex.: "/f/..." -> "\f\..."
    if m:
        p2 = f"{m.group(1).upper()}:\\{m.group(2)}"
    return os.path.normpath(p2)

# ---------- Sanitize: inválidas + MultiPolygon → Polygon ----------
def _largest_polygon_from_geom(geom):
    """Retorna um Polygon:
       - se Polygon: devolve;
       - se MultiPolygon: maior por área;
       - se GeometryCollection: maior Polygon dentre os componentes;
       - caso contrário: None.
    """
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        try:
            return max(geom.geoms, key=lambda g: g.area)
        except ValueError:
            return None
    if isinstance(geom, GeometryCollection):
        parts = []
        for g in geom.geoms:
            if isinstance(g, Polygon):
                parts.append(g)
            elif isinstance(g, MultiPolygon):
                parts.extend(list(g.geoms))
        if parts:
            return max(parts, key=lambda g: g.area)
        return None
    return None  # outros tipos (LineString, Point etc.)

def sanitize_to_polygons(gdf, fix_invalid=True, verbose=True):
    """
    Corrige geometrias e garante tipo Polygon:
      1) (opcional) buffer(0) para 'make valid'
      2) MultiPolygon/GeometryCollection → maior Polygon
      3) remove linhas sem Polygon (None/empty)
    """
    gdf = gdf.copy()
    if fix_invalid:
        try:
            gdf["geometry"] = gdf.buffer(0)
        except Exception:
            pass
    before = len(gdf)
    try:
        mp_count = int((gdf.geom_type == "MultiPolygon").sum())
    except Exception:
        mp_count = None
    gdf["geometry"] = gdf["geometry"].apply(_largest_polygon_from_geom)
    gdf = gdf[~gdf["geometry"].isna() & ~gdf["geometry"].is_empty].copy()
    gdf.reset_index(drop=True, inplace=True)
    if verbose:
        after = len(gdf)
        conv = f"{mp_count} MultiPolygon" if mp_count is not None else "MultiPolygon"
        print(f"[sanitize] {conv} → Polygon; removidas {(before-after)} geometrias vazias/sem polígono. Total: {after}")
    return gdf

# ---------- Geo helpers ----------
def ensure_projected(gdf: gpd.GeoDataFrame, target_epsg: int = TARGET_EPSG) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("O shapefile não possui CRS definido.")
    if gdf.crs.is_geographic:
        print(f"[info] CRS geográfico detectado ({gdf.crs}). Reprojetando para EPSG:{target_epsg}...")
        gdf = gdf.to_crs(epsg=target_epsg)
    return gdf

def representative_xy(geom):
    try:
        c = geom.centroid; return c.x, c.y
    except Exception:
        rp = geom.representative_point(); return rp.x, rp.y

def clockwise_angles(xs, ys, gx, gy):
    theta = np.arctan2(ys - gy, xs - gx)
    theta = (theta + 2*np.pi) % (2*np.pi)
    return (2*np.pi - theta) % (2*np.pi)

def pick_southwest_index(sub_df):
    order = sub_df.sort_values(by=["_cy", "_cx"], ascending=[True, True])
    return order.index[0]

# ---------- Adjacência com pesos ----------
def _shared_metrics(gi, gj):
    """Comprimento de aresta comum e área de sobreposição (robusto)."""
    try:
        inter_b = gi.boundary.intersection(gj.boundary)
        shared_len = getattr(inter_b, "length", 0.0) or 0.0
    except Exception:
        shared_len = 0.0
    try:
        inter = gi.intersection(gj)
        overlap_area = getattr(inter, "area", 0.0) or 0.0
    except Exception:
        overlap_area = 0.0
    return float(shared_len), float(overlap_area)

def _build_adjacency(sub: gpd.GeoDataFrame,
                     adj_buffer: float,
                     min_shared_len_ratio: float):
    """
    Retorna: adj (dict), wlen (comprimento aresta), warea (área).
    Exige vizinhança real: interseção nas geometrias bufferizadas e,
    nas originais, aresta compartilhada relativa mínima (ou área>0).
    """
    adj = defaultdict(set); wlen, warea = {}, {}
    geoms_raw = sub.geometry
    geoms_adj = geoms_raw.buffer(adj_buffer) if (adj_buffer and adj_buffer > 0) else geoms_raw
    sindex = geoms_adj.sindex
    idx_list = list(sub.index)
    perim = {i: (getattr(geoms_raw[i].boundary, "length", 0.0) or 0.0) for i in idx_list}

    for i in idx_list:
        gi_adj = geoms_adj[i]
        hits_pos = list(sindex.intersection(gi_adj.bounds))
        for j_pos in hits_pos:
            j = idx_list[j_pos]
            if j <= i: continue
            gj_adj = geoms_adj[j]
            try:
                if not gi_adj.intersects(gj_adj): continue
            except Exception:
                pass
            gi_raw = geoms_raw[i]; gj_raw = geoms_raw[j]
            shared_len, overlap_area = _shared_metrics(gi_raw, gj_raw)
            min_per = max(min(perim[i], perim[j]), 1e-9)
            if (shared_len / min_per) < min_shared_len_ratio and overlap_area <= 0:
                continue
            adj[i].add(j); adj[j].add(i)
            wlen[(i,j)] = wlen[(j,i)] = shared_len
            warea[(i,j)] = warea[(j,i)] = overlap_area
    return adj, wlen, warea

# ---------- Contorno e ordem no perímetro ----------
def _merge_boundary(union_poly):
    """Une o contorno numa única LineString (pega a maior se for MultiLineString)."""
    b = union_poly.boundary
    try:
        merged = linemerge(b)
    except Exception:
        merged = b
    if merged.geom_type == "MultiLineString":
        merged = max(list(merged.geoms), key=lambda ls: ls.length)
    return merged

def _perimeter_order(sub, union_poly, touch_tol=0.05):
    """Índices dos lotes de perímetro ordenados pela posição AO LONGO do contorno."""
    boundary_ls = _merge_boundary(union_poly)
    order_items = []  # (measure, idx, shared_len)
    for i, geom in sub.geometry.items():
        try:
            inter = geom.boundary.intersection(boundary_ls.buffer(touch_tol))
        except Exception:
            continue
        shared_len = getattr(inter, "length", 0.0) or 0.0
        if shared_len <= 0:  # não toca contorno
            continue

        # ponto representativo na maior sub-linha
        if inter.geom_type == "LineString":
            rep_pt = inter.interpolate(0.5, normalized=True)
        elif inter.geom_type == "MultiLineString":
            biggest = max(list(inter.geoms), key=lambda ls: ls.length)
            rep_pt = biggest.interpolate(0.5, normalized=True)
        else:
            rep_pt = geom.representative_point()

        try:
            m = boundary_ls.project(rep_pt)
        except Exception:
            m = 0.0
        order_items.append((m, i, shared_len))

    order_items.sort(key=lambda t: (t[0], -t[2]))
    return [i for _, i, _ in order_items]

# ---------- Numeração por quadra (perímetro + heading) ----------
def number_lots_clockwise(
    gdf,
    quadra_col=QUADRA_COL,
    lote_col=LOTE_COL,
    adj_buffer: float = ADJ_BUFFER_DEFAULT,
    min_shared_len_ratio: float = MIN_SHARED_LEN_RATIO_DEFAULT,
    perimeter_first: bool = PERIMETER_FIRST_DEFAULT,
    perim_touch_tol: float = PERIM_TOUCH_TOL_DEFAULT
):
    if quadra_col not in gdf.columns:
        raise KeyError(f"Coluna '{quadra_col}' não encontrada no shapefile.")

    # coords representativas
    xy = gdf.geometry.apply(representative_xy)
    gdf["_cx"] = [p[0] for p in xy]; gdf["_cy"] = [p[1] for p in xy]

    if lote_col in gdf.columns:
        print(f"[aviso] Coluna '{lote_col}' já existe e será sobrescrita.")
    gdf[lote_col] = None

    for quadra, idx in gdf.groupby(quadra_col).groups.items():
        sub = gdf.loc[idx].copy()
        # União robusta
        try:
            union = sub.geometry.union_all()
        except Exception:
            union = sub.geometry.unary_union

        # ----- PERÍMETRO -----
        perim_seq = _perimeter_order(sub, union, touch_tol=perim_touch_tol)
        perim_set = set(perim_seq)

        if perim_seq:
            sw_idx = pick_southwest_index(sub.loc[perim_seq])
            if sw_idx in perim_seq:
                pos = perim_seq.index(sw_idx)
                perim_seq = perim_seq[pos:] + perim_seq[:pos]

        path = []
        if perimeter_first and perim_seq:
            path.extend(perim_seq)

        # ----- MIOLO (heading + camada) -----
        adj, wlen, warea = _build_adjacency(sub,
                                            adj_buffer=adj_buffer,
                                            min_shared_len_ratio=min_shared_len_ratio)

        boundary_ls = _merge_boundary(union)
        dist_layer = {}
        for i, g in sub.geometry.items():
            try:
                dist_layer[i] = g.centroid.distance(boundary_ls)
            except Exception:
                dist_layer[i] = 0.0

        visited = set(path)
        remaining = [i for i in sub.index if i not in visited]

        gx, gy = union.centroid.x, union.centroid.y

        def _cw_delta_from_heading(vx, vy, hx, hy):
            th_h = np.arctan2(hy, hx) % (2*np.pi)
            th_v = np.arctan2(vy, vx) % (2*np.pi)
            return (th_h - th_v) % (2*np.pi)

        def choose_seed():
            # último do perímetro com vizinho não visitado
            for i in reversed(path):
                if any(n not in visited for n in adj.get(i, [])):
                    return i
            if remaining:
                return min(remaining, key=lambda i: dist_layer[i])
            return None

        current = choose_seed()
        if current is not None:
            hx, hy = (sub.loc[current, "_cx"] - gx, sub.loc[current, "_cy"] - gy)
            if hx == 0 and hy == 0: hx, hy = (1.0, 0.0)

        def next_interior(current):
            if current is None:
                return None
            neigh = [n for n in adj.get(current, []) if n not in visited]
            if not neigh:
                return None
            cx, cy = sub.loc[current, ["_cx", "_cy"]].values
            cur_d = dist_layer[current]
            scored = []
            for n in neigh:
                nx, ny = sub.loc[n, ["_cx", "_cy"]].values
                dth = _cw_delta_from_heading(nx - cx, ny - cy, hx, hy)
                same_layer_pen = abs(dist_layer[n] - cur_d)
                shared = wlen.get((current, n), 0.0)
                scored.append((n, same_layer_pen, dth, -shared))
            scored.sort(key=lambda t: (t[1], t[2], t[3]))
            return scored[0][0] if scored else None

        # -------- Loop com fallbacks anti-travar --------
        max_steps = len(sub) * 2  # safety
        steps = 0
        while remaining and steps < max_steps:
            steps += 1

            if current is None or current not in sub.index:
                current = choose_seed()
                if current is None:
                    break
                hx, hy = (sub.loc[current, "_cx"] - gx, sub.loc[current, "_cy"] - gy)
                if hx == 0 and hy == 0: hx, hy = (1.0, 0.0)

            nxt = next_interior(current)

            if nxt is None:
                # Fallback 1: qualquer vizinho não visitado de current
                neighs = [n for n in adj.get(current, []) if n in remaining]
                if neighs:
                    nxt = neighs[0]
                else:
                    # Fallback 2: primeiro restante global
                    nxt = remaining[0]

            # atualiza heading e caminha
            cx, cy = sub.loc[current, ["_cx", "_cy"]].values
            nx, ny = sub.loc[nxt, ["_cx", "_cy"]].values
            hx, hy = (nx - cx, ny - cy) if (nx != cx or ny != cy) else (1.0, 0.0)

            path.append(nxt); visited.add(nxt); remaining.remove(nxt)
            current = nxt

        # Garante início global no sudoeste do conjunto inteiro
        if path:
            sw_all = pick_southwest_index(sub.loc[path])
            if sw_all in path:
                pos0 = path.index(sw_all)
                path = path[pos0:] + path[:pos0]

        # Escreve 1..N
        for n, rid in enumerate(path, start=1):
            gdf.at[rid, lote_col] = int(n)

        print(f"[ok] QUADRA={quadra!r} → {len(path)} lotes "
              f"(perímetro por contorno; miolo por heading + camada + fallbacks)")

    # limpeza
    for col in ["_cx", "_cy", "_ang_cw"]:
        if col in gdf.columns: gdf.drop(columns=[col], inplace=True)
    return gdf

# ---------- GUI / fluxo principal ----------
def main():
    root = Tk(); root.withdraw()
    try: root.attributes("-topmost", True)
    except Exception: pass

    shp_path = filedialog.askopenfilename(
        title="Selecione o shapefile de LOTES",
        filetypes=[("Shapefiles", "*.shp")],
        parent=root
    )
    if not shp_path:
        print("Nenhum arquivo selecionado."); return

    shp_path = _fix_windows_path(shp_path)
    print(f"[debug] Caminho selecionado: {shp_path}")
    if not os.path.exists(shp_path):
        messagebox.showerror("Erro", "Caminho não existe no sistema de arquivos."); return

    base, _ = os.path.splitext(shp_path)
    outp = base + "_numerado.shp"

    try:
        gdf = gpd.read_file(shp_path)

        # Corrige e força Polygon (maior parte por área)
        gdf = sanitize_to_polygons(gdf, fix_invalid=True, verbose=True)

        # Reprojeção se necessário
        gdf = ensure_projected(gdf, target_epsg=TARGET_EPSG)

        # Numeração
        gdf = number_lots_clockwise(
            gdf,
            quadra_col=QUADRA_COL,
            lote_col=LOTE_COL,
            adj_buffer=ADJ_BUFFER_DEFAULT,
            min_shared_len_ratio=MIN_SHARED_LEN_RATIO_DEFAULT,
            perimeter_first=PERIMETER_FIRST_DEFAULT,
            perim_touch_tol=PERIM_TOUCH_TOL_DEFAULT
        )
        gdf[LOTE_COL] = gdf[LOTE_COL].astype("int32")
        gdf.to_file(outp, driver="ESRI Shapefile", encoding="utf-8")
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro:\n{e}")
        raise

    print(f"[salvo] {outp}")
    messagebox.showinfo("Concluído", f"Arquivo salvo em:\n{outp}")

if __name__ == "__main__":
    main()
