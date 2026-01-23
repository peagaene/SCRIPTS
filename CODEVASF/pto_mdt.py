#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extrai pontos cotados estratégicos de um MDT (GeoTIFF) e salva como ESRI Shapefile (.shp):
- Picos (máximos locais)      -> CLASS='peak'
- Depressões (mínimos locais) -> CLASS='pit'
- Alta curvatura (quebras)    -> CLASS='curv'

Recursos:
- Modo interativo (sem parâmetros): diálogos para escolher MDT e saída
- AOI (SHP/GPKG/GeoJSON) para RECORTAR o MDT antes de processar
- mask-points (SHP/GPKG/GeoJSON) para manter APENAS pontos dentro do polígono
- Novos: --mask-mode (within/covers/intersects) e --mask-buffer (m)
- Decimação (--decimate) para prévia rápida
- Logs detalhados

Requisitos: rasterio, numpy, scipy, geopandas, shapely, pandas
"""

import argparse
import os
import sys
import time
import traceback
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import rasterio
import rasterio.mask
from affine import Affine
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter, sobel
from scipy.spatial import cKDTree


# -------------------- Args (com diálogos) --------------------
def parse_or_prompt_args():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    ap = argparse.ArgumentParser(description="Pontos cotados estratégicos a partir de MDT (salva .shp).")
    ap.add_argument("--dem", help="Caminho do MDT (GeoTIFF).")
    ap.add_argument("--out", help="Saída SHP (ex.: C:/dados/pontos.shp).")
    ap.add_argument("--target-epsg", type=int, default=31983, help="EPSG para saída e cálculo de distâncias (default: 31983).")
    ap.add_argument("--win", type=int, default=7, help="Janela ímpar para extremos (default: 7).")
    ap.add_argument("--sigma", type=float, default=1.0, help="Suavização Gaussiana (default: 1.0).")
    ap.add_argument("--min-slope", type=float, default=2.0, help="Declividade mínima (graus) para aceitar candidato.")
    ap.add_argument("--min-dist", type=float, default=100.0, help="Distância mínima entre pontos (m).")
    ap.add_argument("--topn", type=int, default=300, help="Máximo por classe (peak/pit/curv).")
    ap.add_argument("--aoi", help="Polígono (SHP/GPKG/GeoJSON) para RECORTAR o MDT (opcional).")
    ap.add_argument("--mask-points", help="Polígono (SHP/GPKG/GeoJSON) p/ manter APENAS pontos dentro (opcional).")
    ap.add_argument("--mask-mode", choices=["within","covers","intersects"],
                    default="covers",
                    help="Critério espacial para manter pontos: within (estritamente dentro), covers (inclui borda), intersects (qualquer interseção).")
    ap.add_argument("--mask-buffer", type=float, default=0.0,
                    help="Buffer em metros aplicado à máscara antes do filtro (ex.: 2.0 ou -2.0).")
    ap.add_argument("--decimate", type=int, default=1, help="Fator de decimação (1=nativo, 2/4/8...).")
    ap.add_argument("--verbose", action="store_true", help="Logs adicionais (debug).")
    args, _ = ap.parse_known_args()

    # modo interativo (sem parâmetros obrigatórios)
    if not args.dem or not args.out:
        root = tk.Tk(); root.withdraw()

        if not args.dem:
            args.dem = filedialog.askopenfilename(
                title="Selecione o MDT (GeoTIFF)",
                filetypes=[("GeoTIFF", "*.tif *.tiff"), ("Todos os arquivos", "*.*")]
            )
            if not args.dem:
                ap.error("Arquivo MDT não selecionado.")

        if not args.out:
            args.out = filedialog.asksaveasfilename(
                title="Salvar pontos (Shapefile)",
                defaultextension=".shp",
                filetypes=[("ESRI Shapefile", "*.shp")]
            )
            if not args.out:
                ap.error("Arquivo de saída não informado.")

        # Pergunta AOI (recorte real do raster)
        if not args.aoi:
            if messagebox.askyesno("AOI (opcional)", "Deseja selecionar um polígono para RECORTAR o MDT (AOI)?"):
                args.aoi = filedialog.askopenfilename(
                    title="Selecione o polígono AOI (SHP/GPKG/GeoJSON)",
                    filetypes=[("Arquivos vetoriais", "*.shp *.gpkg *.geojson *.json"),
                               ("Shapefile", "*.shp"), ("GeoPackage", "*.gpkg"),
                               ("GeoJSON", "*.geojson *.json"), ("Todos", "*.*")]
                )

        # Pergunta mask-points (filtra só os pontos)
        if not args.mask_points:
            if messagebox.askyesno("Mask Points (opcional)", "Deseja selecionar um polígono para MANTER APENAS pontos dentro do limite?"):
                args.mask_points = filedialog.askopenfilename(
                    title="Selecione o polígono para filtrar pontos (SHP/GPKG/GeoJSON)",
                    filetypes=[("Arquivos vetoriais", "*.shp *.gpkg *.geojson *.json"),
                               ("Shapefile", "*.shp"), ("GeoPackage", "*.gpkg"),
                               ("GeoJSON", "*.geojson *.json"), ("Todos", "*.*")]
                )

    # força extensão .shp
    root_out, ext = os.path.splitext(args.out)
    if ext.lower() != ".shp":
        args.out = root_out + ".shp"

    # win ímpar
    if args.win % 2 == 0:
        args.win += 1

    if args.decimate < 1:
        args.decimate = 1

    return args


def log_step(msg, t0=None):
    if t0 is None:
        print(msg, flush=True); return time.perf_counter()
    else:
        dt = time.perf_counter() - t0
        print(f"{msg} ({dt:.1f}s)", flush=True); return time.perf_counter()


# -------------------- Núcleo --------------------
def read_dem(dem_path, aoi_path=None, decimate=1, verbose=False):
    t0 = log_step("[1/7] Lendo MDT...")
    ds = rasterio.open(dem_path)
    crs = ds.crs
    nodata = ds.nodata
    transform = ds.transform

    if aoi_path:
        gdf_aoi = gpd.read_file(aoi_path)
        if gdf_aoi.crs != crs:
            gdf_aoi = gdf_aoi.to_crs(crs)
        shapes = [geom for geom in gdf_aoi.geometry if geom is not None and not geom.is_empty]
        data, transform = rasterio.mask.mask(ds, shapes, crop=True, filled=True, nodata=nodata)
        arr = data[0]
        z = np.ma.masked_equal(arr, nodata).astype("float32") if nodata is not None else np.ma.masked_invalid(arr).astype("float32")
        log_step(f"      Recorte AOI: {z.shape[1]} x {z.shape[0]} px", t0)
    else:
        z = ds.read(1, masked=True).astype("float32")
        log_step(f"      Tamanho: {ds.width} x {ds.height} px", t0)

    # decimação rápida por fatiamento
    if decimate > 1:
        z = z[::decimate, ::decimate]
        transform = transform * Affine.scale(decimate, decimate)
        print(f"      ↳ Decimação x{decimate} → {z.shape[1]} x {z.shape[0]} px")

    if verbose:
        print(f"      CRS: {crs}, nodata: {nodata}, res: {transform.a:.3f} x {-transform.e:.3f}")

    return z, transform, crs


def horn_slope_degrees(z, xres, yres):
    dzdx = sobel(z, axis=1, mode='nearest') / (8.0 * xres)
    dzdy = sobel(z, axis=0, mode='nearest') / (8.0 * yres)
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    return np.degrees(slope_rad), dzdx, dzdy


def laplacian_curvature(z, xres, yres):
    dzdx = sobel(z, axis=1, mode='nearest') / (8.0 * xres)
    dzdy = sobel(z, axis=0, mode='nearest') / (8.0 * yres)
    d2zdx2 = sobel(dzdx, axis=1, mode='nearest') / (8.0 * xres)
    d2zdy2 = sobel(dzdy, axis=0, mode='nearest') / (8.0 * yres)
    return d2zdx2 + d2zdy2


def local_extrema(masked_arr, win):
    data = np.array(masked_arr, dtype="float32")
    valid = ~np.ma.getmaskarray(masked_arr)
    data[~valid] = np.nan

    neg_inf = np.nanmin(data) - 1.0 if np.isfinite(np.nanmin(data)) else -1e9
    pos_inf = np.nanmax(data) + 1.0 if np.isfinite(np.nanmax(data)) else 1e9
    data_max = np.nan_to_num(data, nan=neg_inf)
    data_min = np.nan_to_num(data, nan=pos_inf)

    foot = np.ones((win, win), dtype=bool)
    loc_max = (data_max == maximum_filter(data_max, footprint=foot, mode='nearest')) & valid
    loc_min = (data_min == minimum_filter(data_min, footprint=foot, mode='nearest')) & valid

    r = win // 2
    for m in (loc_max, loc_min):
        m[:r, :] = False; m[-r:, :] = False; m[:, :r] = False; m[:, -r:] = False
    return loc_max, loc_min


def rc_to_xy(transform, rows, cols):
    xs = transform.c + cols * transform.a + rows * transform.b
    ys = transform.f + cols * transform.d + rows * transform.e
    return np.array(xs), np.array(ys)


def enforce_min_distance(points_xy, scores, min_dist):
    if len(points_xy) == 0:
        return []
    order = np.argsort(-scores)  # maior score primeiro
    selected = []
    cloud = []

    for idx in order:
        p = points_xy[idx]
        if not cloud:
            selected.append(idx); cloud.append(p); continue
        tree = cKDTree(np.vstack(cloud))
        d, _ = tree.query(p, k=1)
        if d >= min_dist:
            selected.append(idx); cloud.append(p)
    return selected


def extract_points(dem_path, out_shp, target_epsg=31983, win=7, sigma=1.0,
                   min_slope_deg=2.0, min_dist_m=100.0, topn=300,
                   aoi_path=None, mask_points_path=None, mask_mode="covers", mask_buffer=0.0,
                   decimate=1, verbose=False):

    T0 = time.perf_counter()
    z, transform, src_crs = read_dem(dem_path, aoi_path=aoi_path, decimate=decimate, verbose=verbose)
    xres, yres = transform.a, -transform.e
    print(f"      Resolução estimada (CRS do raster): {xres:.3f} x {yres:.3f}")

    # [2/7] Suavização
    t = log_step("[2/7] Suavizando...")
    z_s = gaussian_filter(z.filled(np.nan), sigma=sigma)
    z_s = np.ma.masked_invalid(z_s)
    log_step("      OK", t)

    # [3/7] Declividade
    t = log_step("[3/7] Calculando declividade (Horn/Sobel)...")
    slope_deg, _, _ = horn_slope_degrees(z_s, xres, yres)
    log_step("      OK", t)

    # [4/7] Curvatura
    t = log_step("[4/7] Calculando curvatura (Laplaciano aprox.)...")
    curv = laplacian_curvature(z_s, xres, yres)
    log_step("      OK", t)

    # [5/7] Extremos
    t = log_step(f"[5/7] Detectando extremos (win={win})...")
    loc_max, loc_min = local_extrema(z_s, win)
    print(f"      Candidatos brutos → peaks={int(loc_max.sum()):,} | pits={int(loc_min.sum()):,}")
    log_step("      OK", t)

    # [6/7] Seleção por declividade/curvatura
    t = log_step("[6/7] Selecionando candidatos...")
    slope_mask = slope_deg >= min_slope_deg
    peaks_rc = np.where(loc_max & slope_mask)
    pits_rc  = np.where(loc_min & slope_mask)

    curv_mag = np.abs(curv)
    curv_mag = np.where(np.isfinite(curv_mag), curv_mag, 0.0)
    curv_thresh = np.percentile(curv_mag[slope_mask], 98) if np.any(slope_mask) else np.inf
    curv_rc = np.where((curv_mag >= curv_thresh) & slope_mask)

    print(f"      Após declividade (≥{min_slope_deg}°): peaks={len(peaks_rc[0]):,}, pits={len(pits_rc[0]):,}, curv_cands={len(curv_rc[0]):,}")

    # GDFs no CRS de origem
    def build_gdf(rows, cols, elev_arr, cls_label):
        xs, ys = rc_to_xy(transform, rows, cols)
        gdf = gpd.GeoDataFrame(
            {"ELEV": elev_arr[rows, cols].astype("float64"), "CLASS": cls_label},
            geometry=[Point(x, y) for x, y in zip(xs, ys)],
            crs=src_crs
        )
        return gdf

    gdf_peaks = build_gdf(peaks_rc[0], peaks_rc[1], z_s, "peak")
    gdf_pits  = build_gdf(pits_rc[0],  pits_rc[1],  z_s, "pit")
    xs_c, ys_c = rc_to_xy(transform, curv_rc[0], curv_rc[1])
    gdf_curv = gpd.GeoDataFrame(
        {"ELEV": z_s[curv_rc].astype("float64"), "CLASS": "curv"},
        geometry=[Point(x, y) for x, y in zip(xs_c, ys_c)],
        crs=src_crs
    )

    # reprojetar (para min-dist em metros)
    if target_epsg:
        gdf_peaks = gdf_peaks.to_crs(epsg=target_epsg)
        gdf_pits  = gdf_pits.to_crs(epsg=target_epsg)
        gdf_curv  = gdf_curv.to_crs(epsg=target_epsg)

    # min-dist por classe
    def select_by_distance(gdf, score, limit, label):
        if len(gdf) == 0:
            print(f"      {label}: 0 candidatos → 0 mantidos")
            return gdf
        pts = np.vstack([gdf.geometry.x.values, gdf.geometry.y.values]).T
        keep_idx = enforce_min_distance(pts, score, min_dist_m)
        if len(keep_idx) > limit:
            keep_idx = keep_idx[:limit]
        print(f"      {label}: {len(gdf):,} candidatos → {len(keep_idx):,} mantidos (min-dist={min_dist_m} m, topn={limit})")
        return gdf.iloc[keep_idx].copy()

    curv_scores = curv_mag[curv_rc] if len(gdf_curv) else np.array([])
    sel_peaks = select_by_distance(gdf_peaks, gdf_peaks["ELEV"].values, topn, "peaks")
    sel_pits  = select_by_distance(gdf_pits, -gdf_pits["ELEV"].values, topn, "pits")  # menor cota = prioridade
    sel_curv  = select_by_distance(gdf_curv, curv_scores, topn, "curv")

    # juntar classes
    out_geom = pd.concat([sel_peaks.geometry, sel_pits.geometry, sel_curv.geometry], ignore_index=True) if (len(sel_peaks)+len(sel_pits)+len(sel_curv)) else gpd.GeoSeries([], dtype="geometry")
    out_gdf = gpd.GeoDataFrame(
        {
            "ELEV": np.concatenate([
                sel_peaks["ELEV"].values if len(sel_peaks) else np.array([]),
                sel_pits["ELEV"].values  if len(sel_pits)  else np.array([]),
                sel_curv["ELEV"].values  if len(sel_curv)  else np.array([]),
            ], dtype="float64"),
            "CLASS": np.concatenate([
                sel_peaks["CLASS"].values if len(sel_peaks) else np.array([]),
                sel_pits["CLASS"].values  if len(sel_pits)  else np.array([]),
                sel_curv["CLASS"].values  if len(sel_curv)  else np.array([]),
            ], dtype="object"),
        },
        geometry=out_geom,
        crs=f"EPSG:{target_epsg}" if target_epsg else src_crs
    )

    # Filtro opcional (apenas nos pontos) se usuário preferiu mask-points
    if (aoi_path is None) and mask_points_path and len(out_gdf):
        try:
            gdf_mask = gpd.read_file(mask_points_path)
            if gdf_mask.crs is None:
                raise ValueError("Máscara sem CRS definido.")

            print(f"      CRS pontos: {out_gdf.crs} | CRS máscara(orig): {gdf_mask.crs}")

            # reprojeta máscara pro CRS dos pontos
            if str(gdf_mask.crs) != str(out_gdf.crs):
                gdf_mask = gdf_mask.to_crs(out_gdf.crs)

            # explode partes, corrige geometrias e dissolve em um único polígono
            gdf_mask = gdf_mask.explode(index_parts=False, ignore_index=True)
            gdf_mask["geometry"] = gdf_mask.buffer(0)  # corrige self-intersections
            gdf_mask = gdf_mask[~gdf_mask.geometry.is_empty & gdf_mask.geometry.is_valid]

            if gdf_mask.empty:
                raise ValueError("Máscara vazia após correção.")

            mask_geom = gdf_mask.unary_union

            # buffer opcional (m); funciona bem em CRS projetado (UTM)
            if mask_buffer and mask_buffer != 0.0:
                mask_geom = mask_geom.buffer(mask_buffer)

            before = len(out_gdf)
            if mask_mode == "within":
                sel = out_gdf.geometry.within(mask_geom)
            elif mask_mode == "covers":
                sel = out_gdf.geometry.apply(mask_geom.covers)
            else:  # intersects
                sel = out_gdf.geometry.intersects(mask_geom)

            out_gdf = out_gdf.loc[sel].copy()
            print(f"      Filtro --mask-points ({mask_mode}){f' c/ buffer={mask_buffer}m' if mask_buffer else ''}: {before:,} → {len(out_gdf):,}")
        except Exception as e:
            print(f"      [Aviso] Falha ao aplicar --mask-points: {e}")

    log_step("      OK", t)

    # [7/7] Salvar SHP
    t = log_step("[7/7] Salvando Shapefile...")
    out_gdf.to_file(out_shp, driver="ESRI Shapefile", encoding="utf-8")
    log_step(f"      Salvo: {out_shp} | Total pontos={len(out_gdf):,}", t)

    # resumo
    n_peak = int((out_gdf["CLASS"] == "peak").sum()) if len(out_gdf) else 0
    n_pit  = int((out_gdf["CLASS"] == "pit").sum()) if len(out_gdf) else 0
    n_curv = int((out_gdf["CLASS"] == "curv").sum()) if len(out_gdf) else 0
    print(f"[OK] Concluído em {time.perf_counter() - T0:.1f}s → peaks={n_peak:,}, pits={n_pit:,}, curv={n_curv:,}, total={len(out_gdf):,}")


# -------------------- Main --------------------
def main():
    args = parse_or_prompt_args()
    print(f"MDT: {args.dem}")
    print(f"Saída (SHP): {args.out}")
    print(f"Parâmetros → EPSG={args.target_epsg} | win={args.win} | sigma={args.sigma} | min_slope={args.min_slope}° | min_dist={args.min_dist} m | "
          f"topn={args.topn} | decimate={args.decimate} | AOI={args.aoi or 'None'} | mask-points={args.mask_points or 'None'} | "
          f"mask-mode={args.mask_mode} | mask-buffer={args.mask_buffer} m")

    try:
        extract_points(
            dem_path=args.dem,
            out_shp=args.out,
            target_epsg=args.target_epsg,
            win=args.win,
            sigma=args.sigma,
            min_slope_deg=args.min_slope,
            min_dist_m=args.min_dist,
            topn=args.topn,
            aoi_path=args.aoi,
            mask_points_path=args.mask_points,
            mask_mode=args.mask_mode,
            mask_buffer=args.mask_buffer,
            decimate=args.decimate,
            verbose=args.verbose
        )
    except Exception:
        print("\n[ERRO] Falha no processamento:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
