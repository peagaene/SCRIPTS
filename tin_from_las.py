import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import laspy
except Exception as exc:
    raise SystemExit(f"laspy nao encontrado: {exc}")

try:
    from scipy.spatial import Delaunay
except Exception as exc:
    raise SystemExit(f"scipy nao encontrado: {exc}")

try:
    import geopandas as gpd
    from shapely.geometry import Polygon
except Exception as exc:
    raise SystemExit(f"geopandas/shapely nao encontrado: {exc}")


# Keep it simple: grid-sample points to the MDT resolution and triangulate in XY.

def _grid_sample_min_z(x, y, z, res):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    if np.size(x) == 0:
        return x, y, z

    x0 = x.min()
    y0 = y.min()
    ix = np.floor((x - x0) / res).astype(np.int64)
    iy = np.floor((y - y0) / res).astype(np.int64)

    # Sort by cell, then by z so we keep the lowest point per cell.
    order = np.lexsort((z, iy, ix))
    ix_s = ix[order]
    iy_s = iy[order]

    # First occurrence of each cell after sorting is the min-z point.
    cell_change = np.empty_like(ix_s, dtype=bool)
    cell_change[0] = True
    cell_change[1:] = (ix_s[1:] != ix_s[:-1]) | (iy_s[1:] != iy_s[:-1])
    keep_idx = order[cell_change]

    return x[keep_idx], y[keep_idx], z[keep_idx]


def _get_crs_from_las(las, epsg):
    if epsg:
        try:
            import pyproj
        except Exception as exc:
            raise SystemExit(f"pyproj nao encontrado para --epsg: {exc}")
        return pyproj.CRS.from_epsg(int(epsg))

    try:
        return las.header.parse_crs()
    except Exception:
        return None


def build_tin(las_path: Path, res: float, epsg: int | None, out_path: Path | None, layer: str):
    las = laspy.read(las_path)

    x = las.x
    y = las.y
    z = las.z

    # Filter to ground points if classification is present.
    if hasattr(las, "classification"):
        cls = las.classification
        if np.any(cls == 2):
            mask = cls == 2
            x, y, z = x[mask], y[mask], z[mask]

    # Grid-sample to MDT resolution.
    x_s, y_s, z_s = _grid_sample_min_z(x, y, z, res)

    if np.size(x_s) < 3:
        raise SystemExit("pontos insuficientes para triangulacao")

    pts_xy = np.column_stack((x_s, y_s))
    tri = Delaunay(pts_xy)

    # Build triangle polygons with Z.
    triangles = []
    for a, b, c in tri.simplices:
        triangles.append(
            Polygon([
                (x_s[a], y_s[a], z_s[a]),
                (x_s[b], y_s[b], z_s[b]),
                (x_s[c], y_s[c], z_s[c]),
                (x_s[a], y_s[a], z_s[a]),
            ])
        )

    crs = _get_crs_from_las(las, epsg)
    gdf = gpd.GeoDataFrame({"id": np.arange(len(triangles), dtype=np.int64)}, geometry=triangles, crs=crs)

    if out_path is None:
        out_path = las_path.with_suffix(".gpkg")

    gdf.to_file(out_path, layer=layer, driver="GPKG")
    return out_path, len(triangles), len(x_s)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gera TIN (triangulacao) a partir de LAS/LAZ e exporta para GPKG.")
    ap.add_argument("las", help="Caminho para arquivo .las/.laz")
    ap.add_argument("--res", type=float, default=0.25, help="Resolucao do MDT (padrao 0.25 m)")
    ap.add_argument("--epsg", type=int, default=None, help="EPSG caso o LAS nao tenha CRS")
    ap.add_argument("--out", default=None, help="Arquivo GPKG de saida (padrao: mesmo nome do LAS)")
    ap.add_argument("--layer", default="tin", help="Nome da camada no GPKG")

    args = ap.parse_args(argv)
    las_path = Path(args.las)
    if not las_path.exists():
        raise SystemExit("arquivo LAS/LAZ nao encontrado")

    out_path = Path(args.out) if args.out else None

    out_path, n_tri, n_pts = build_tin(las_path, args.res, args.epsg, out_path, args.layer)
    print(f"TIN gerado: {out_path} | triangulos: {n_tri} | pontos usados: {n_pts}")


def _run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("TIN a partir de LAS/LAZ")
    root.geometry("640x420")

    las_var = tk.StringVar()
    out_var = tk.StringVar()
    res_var = tk.StringVar(value="0.25")
    epsg_var = tk.StringVar()
    layer_var = tk.StringVar(value="tin")
    log_var = tk.StringVar(value="")

    def pick_las():
        path = filedialog.askopenfilename(
            title="Selecionar LAS/LAZ",
            filetypes=[("LAS/LAZ", "*.las *.laz"), ("Todos", "*.*")]
        )
        if path:
            las_var.set(path)
            if not out_var.get():
                out_var.set(str(Path(path).with_suffix(".gpkg")))

    def pick_out():
        path = filedialog.asksaveasfilename(
            title="Salvar GPKG",
            defaultextension=".gpkg",
            filetypes=[("GeoPackage", "*.gpkg")]
        )
        if path:
            out_var.set(path)

    def run():
        las_path = las_var.get().strip()
        out_path = out_var.get().strip()
        if not las_path:
            log("Erro: selecione um arquivo LAS/LAZ.")
            messagebox.showerror("Erro", "Selecione um arquivo LAS/LAZ.")
            return
        if not out_path:
            out_path = str(Path(las_path).with_suffix(".gpkg"))
        try:
            res = float(res_var.get().strip())
        except Exception:
            log("Erro: resolucao invalida.")
            messagebox.showerror("Erro", "Resolucao invalida.")
            return
        epsg_txt = epsg_var.get().strip()
        epsg = int(epsg_txt) if epsg_txt else None
        layer = layer_var.get().strip() or "tin"

        try:
            log(f"Iniciando: {las_path}")
            log(f"Saida: {out_path}")
            log(f"Resolucao: {res} | EPSG: {epsg if epsg is not None else 'auto'} | Layer: {layer}")
            out_path, n_tri, n_pts = build_tin(Path(las_path), res, epsg, Path(out_path), layer)
        except Exception as exc:
            log(f"Erro: {exc}")
            messagebox.showerror("Erro", f"Falha ao gerar TIN:\n{exc}")
            return

        log(f"Concluido: {out_path} | triangulos: {n_tri} | pontos: {n_pts}")
        messagebox.showinfo(
            "Concluido",
            f"TIN gerado:\n{out_path}\nTriangulos: {n_tri}\nPontos usados: {n_pts}"
        )

    pad = 6
    frm = tk.Frame(root, padx=10, pady=10)
    frm.pack(fill="both", expand=True)

    def log(msg):
        log_text.configure(state="normal")
        log_text.insert("end", msg + "\n")
        log_text.see("end")
        log_text.configure(state="disabled")

    tk.Label(frm, text="LAS/LAZ:").grid(row=0, column=0, sticky="w")
    tk.Entry(frm, textvariable=las_var, width=58).grid(row=0, column=1, sticky="we", padx=(0, pad))
    tk.Button(frm, text="...", command=pick_las, width=4).grid(row=0, column=2, sticky="e")

    tk.Label(frm, text="GPKG saida:").grid(row=1, column=0, sticky="w", pady=(pad, 0))
    tk.Entry(frm, textvariable=out_var, width=58).grid(row=1, column=1, sticky="we", padx=(0, pad), pady=(pad, 0))
    tk.Button(frm, text="...", command=pick_out, width=4).grid(row=1, column=2, sticky="e", pady=(pad, 0))

    tk.Label(frm, text="Resolucao (m):").grid(row=2, column=0, sticky="w", pady=(pad, 0))
    tk.Entry(frm, textvariable=res_var, width=12).grid(row=2, column=1, sticky="w", pady=(pad, 0))

    tk.Label(frm, text="EPSG (opcional):").grid(row=3, column=0, sticky="w", pady=(pad, 0))
    tk.Entry(frm, textvariable=epsg_var, width=12).grid(row=3, column=1, sticky="w", pady=(pad, 0))

    tk.Label(frm, text="Layer:").grid(row=4, column=0, sticky="w", pady=(pad, 0))
    tk.Entry(frm, textvariable=layer_var, width=12).grid(row=4, column=1, sticky="w", pady=(pad, 0))

    tk.Button(frm, text="Gerar TIN", command=run, width=16).grid(row=5, column=1, sticky="w", pady=(pad * 2, 0))

    tk.Label(frm, text="Log:").grid(row=6, column=0, sticky="w", pady=(pad, 0))
    log_text = tk.Text(frm, height=8, width=70, state="disabled")
    log_text.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(0, pad))

    frm.columnconfigure(1, weight=1)
    frm.rowconfigure(7, weight=1)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _run_gui()
    else:
        main()
