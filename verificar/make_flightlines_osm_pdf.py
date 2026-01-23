# make_flightlines_osm_pdf_interactive.py
# Requer: geopandas, shapely, matplotlib, pyproj, contextily, tkinter
# Rode: python make_flightlines_osm_pdf_interactive.py

import sys
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import matplotlib.pyplot as plt

# --- Interação (seleção de arquivos) ---
import tkinter as tk
from tkinter import filedialog, messagebox

TARGET_EPSG = 31983  # UTM SIRGAS 2000 / 23S
PAGE_INCHES = (11.69, 8.27)  # A4 landscape
LABEL_EVERY = 5
LABEL_FONTSIZE = 8
STROKE_WIDTH = 0.8
DPI = 300

def longest_linestring(geoms):
    ls = [g for g in geoms if isinstance(g, LineString)]
    return max(ls, key=lambda ln: ln.length) if ls else None

def load_reproject_prepare(shp_path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path)
    # Se não tiver CRS, assuma WGS84 (ajuste se necessário):
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    # Reprojetar para EPSG:31983
    gdf = gdf.to_crs(epsg=TARGET_EPSG)

    # Agrupar por FLNUM (ou NAME se não existir) e pegar a maior linha
    group_key = "FLNUM" if "FLNUM" in gdf.columns else "NAME"
    rows = []
    for key, grp in gdf.groupby(group_key, sort=True):
        ln = longest_linestring(grp.geometry)
        if ln is None:
            continue
        # número do strip (int seguro)
        try:
            strip_id = int(float(key))
        except Exception:
            # se NAME não for número, ignore
            continue
        rows.append({"strip": strip_id, "geometry": ln})
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{TARGET_EPSG}").sort_values("strip")
    return out

def draw_pdf_osm_utm(g31983: gpd.GeoDataFrame, out_pdf: str):
    # Basemap OSM (contextily) reprojetado para 31983
    try:
        import contextily as cx
    except Exception as e:
        raise RuntimeError(
            "contextily não encontrado. Instale com: pip install contextily"
        ) from e

    fig = plt.figure(figsize=PAGE_INCHES, dpi=DPI)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.88])  # margens

    # 1) Basemap (OSM padrão) — contextily aceita crs!=3857 e reprojeta
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, crs=g31983.crs)

    # 2) Linhas de voo
    g31983.plot(ax=ax, linewidth=STROKE_WIDTH)

    # 3) Rótulos em 1,5,10… do mesmo lado (borda direita -> maior X do LineString)
    xmin, ymin, xmax, ymax = g31983.total_bounds
    x_offset = 0.006 * (xmax - xmin)  # ajuste fino da distância do rótulo

    for _, row in g31983.iterrows():
        s = int(row["strip"])
        if s == 1 or s % LABEL_EVERY == 0:
            xs, ys = zip(*list(row.geometry.coords))
            i_max = int(np.argmax(xs))  # maior X (leste)
            x_edge, y_edge = xs[i_max], ys[i_max]
            ax.text(
                x_edge + x_offset,
                y_edge,
                f"{s}",
                va="center",
                ha="left",
                fontsize=LABEL_FONTSIZE,
                color="black",
            )

    # 4) Visual limpo
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.01)

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    # Caixa de diálogo para escolher SHP e PDF
    root = tk.Tk()
    root.withdraw()

    shp_path = filedialog.askopenfilename(
        title="Selecione o SHP de linhas de voo",
        filetypes=[("Shapefile", "*.shp"), ("Todos os arquivos", "*.*")],
    )
    if not shp_path:
        messagebox.showwarning("Aviso", "Nenhum arquivo SHP selecionado.")
        return

    out_pdf = filedialog.asksaveasfilename(
        title="Salvar PDF",
        defaultextension=".pdf",
        filetypes=[("PDF", "*.pdf")],
        initialfile="Flight_Lines_OSMstyle_labels_31983.pdf",
    )

    if not out_pdf:
        messagebox.showwarning("Aviso", "Local de saída não informado.")
        return

    try:
        g31983 = load_reproject_prepare(shp_path)
        if g31983.empty:
            raise RuntimeError("Nenhuma linha válida encontrada (verifique FLNUM/NAME).")
        draw_pdf_osm_utm(g31983, out_pdf)
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao gerar PDF:\n{e}")
        raise
    else:
        messagebox.showinfo("Concluído", f"PDF gerado com sucesso:\n{out_pdf}")

if __name__ == "__main__":
    main()
