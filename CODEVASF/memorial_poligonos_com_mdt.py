#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, MultiPolygon, Point, LinearRing
from pyproj import Transformer


# ================== Geometria & Cálculo ==================
def dms_str(angle_deg: float) -> str:
    a = angle_deg % 360.0
    d = int(a)
    m_f = (a - d) * 60.0
    m = int(m_f)
    s = (m_f - m) * 60.0
    s = round(s, 2)
    if s >= 60:
        s -= 60
        m += 1
    if m >= 60:
        m -= 60
        d += 1
    d = d % 360
    s_str = f"{int(s):02d}" if abs(s - int(s)) < 1e-6 else f"{s:05.2f}"
    return f"{d}°{m:02d}'{s_str}''"

def azimuth_EN(dE, dN):
    ang = math.degrees(math.atan2(dE, dN))     # 0° = Norte, sentido horário
    return (ang + 360.0) % 360.0

def dist(dE, dN):
    return math.hypot(dE, dN)

def order_start_corner(coords, corner="SW"):
    """
    Reordena a lista fechada de vértices para o canto desejado:
    corner in {"SW","SE","NE","NW"}.
    """
    closed = coords[0] == coords[-1]
    pts = coords[:-1] if closed else coords[:]

    if corner == "SW":      key = lambda p: (p[1], p[0])             # menor N, depois menor E
    elif corner == "SE":    key = lambda p: (p[1], -p[0])            # menor N, depois maior E
    elif corner == "NE":    key = lambda p: (-p[1], -p[0])           # maior N, depois maior E
    elif corner == "NW":    key = lambda p: (-p[1], p[0])            # maior N, depois menor E
    else:                   key = lambda p: (p[1], p[0])

    idx = min(range(len(pts)), key=lambda i: key(pts[i]))
    pts_rot = pts[idx:] + pts[:idx]
    pts_rot.append(pts_rot[0])
    return pts_rot

def order_start_near_point(coords, refE, refN):
    """Reordena começando no vértice mais próximo de (refE, refN)."""
    closed = coords[0] == coords[-1]
    pts = coords[:-1] if closed else coords[:]
    idx = min(range(len(pts)), key=lambda i: (pts[i][0]-refE)**2 + (pts[i][1]-refN)**2)
    pts_rot = pts[idx:] + pts[:idx]
    pts_rot.append(pts_rot[0])
    return pts_rot

def ensure_orientation(coords, clockwise=True):
    """
    Garante orientação do anel (horária/anti-horária) usando LinearRing.is_ccw.
    coords: lista FECHADA [(E,N),...,(E,N)]
    """
    ring = LinearRing(coords)
    is_ccw = ring.is_ccw
    if clockwise and is_ccw:
        coords = list(reversed(coords))
    if (not clockwise) and (not is_ccw):
        coords = list(reversed(coords))
    # garantir fechado após reverso
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords

def fmt_num(v, casas=2):
    s = f"{v:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_coord(N, E, Z=None):
    if Z is None or (isinstance(Z, float) and (np.isnan(Z) or np.isinf(Z))):
        s = f"N={N:.4f}, E={E:.4f}"
    else:
        s = f"N={N:.4f}, E={E:.4f}, Z={Z:.2f}"
    return s.replace(".", ",")


# ================== Núcleo ==================
def sample_Z_from_raster(src: rasterio.DatasetReader, coords_EN):
    return [float(v[0]) if v[0] is not None else float("nan") for v in src.sample(coords_EN)]

def build_memorial(coordsENZ):
    """Gera memorial (Markdown) e tabela ENZ; coordsENZ deve estar FECHADO."""
    if coordsENZ[0][:2] != coordsENZ[-1][:2]:
        coordsENZ = coordsENZ + [coordsENZ[0]]

    rows, azis, parts = [], [], []
    n = len(coordsENZ) - 1

    for i in range(n):
        E1, N1, Z1 = coordsENZ[i]
        E2, N2, Z2 = coordsENZ[i+1]
        dE, dN = E2 - E1, N2 - N1
        D = dist(dE, dN)
        A = azimuth_EN(dE, dN)
        azis.append(A)

        rows.append({
            "Segmento": f"{i+1}–{i+2 if i+2<=n else 1}",
            "Distancia_m": round(D, 2),
            "Azimute": dms_str(A),
            "Ponto": f"P{i+1}",
            "E": round(E1, 4),
            "N": round(N1, 4),
            "Z": round(Z1, 2),
        })

        if i == 0:
            parts.append(
                f"**Ponto {i+1}** ({fmt_coord(N1,E1,Z1)}), deste ponto segue com distância "
                f"D={fmt_num(D,2)} m e azimute Az={dms_str(A)} até o **Ponto {i+2}** "
                f"({fmt_coord(N2,E2,Z2)});"
            )
        else:
            prev = azis[i-1]
            delta = ((A - prev + 540) % 360) - 180
            lado = "à direita" if delta < 0 else "à esquerda"
            parts.append(
                f"deflete {lado} e segue com D={fmt_num(D,2)} m e azimute Az={dms_str(A)} "
                f"até o **Ponto {i+2 if i+2<=n else 1}** ({fmt_coord(N2,E2,Z2)});"
            )

    E_last, N_last, Z_last = coordsENZ[-1]
    rows.append({"Segmento":"", "Distancia_m":None, "Azimute":"", "Ponto":f"P{n}",
                 "E":round(E_last,4), "N":round(N_last,4), "Z":round(Z_last,2)})

    df = pd.DataFrame(rows, columns=["Segmento","Distancia_m","Azimute","Ponto","E","N","Z"])
    memorial_md = " ".join(parts)
    return memorial_md, df

def process_all(shp_path, dem_path, out_dir, id_field=None, export_points=False,
                start_corner="SW", start_near=None, clockwise=True, log_cb=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise RuntimeError("Shapefile sem feições de polígono.")
    shp_crs = gdf.crs

    with rasterio.open(dem_path) as src:
        raster_crs = src.crs
        transformer = None
        if shp_crs != raster_crs:
            transformer = Transformer.from_crs(shp_crs, raster_crs, always_xy=True)
            log_cb(f"CRS diferente: reprojetando vértices de {shp_crs} → {raster_crs}")

        total_polys = 0
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if not isinstance(geom, (Polygon, MultiPolygon)):
                continue
            polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)

            for ip, poly in enumerate(polys, 1):
                total_polys += 1
                ident = str(row[id_field]) if (id_field and id_field in gdf.columns) else f"feat_{idx}_{ip}"

                coords = list(poly.exterior.coords)  # [(E,N),...,(E,N)] fechado
                # 1) Ponto inicial
                if start_near is not None:
                    coords = order_start_near_point(coords, start_near[0], start_near[1])
                else:
                    coords = order_start_corner(coords, start_corner)

                # 2) Sentido
                coords = ensure_orientation(coords, clockwise=clockwise)

                # 3) Amostrar Z (no CRS do raster)
                EN_src = [(float(x), float(y)) for x, y in coords]  # CRS do SHP (para saída)
                EN_raster = [transformer.transform(x, y) for x, y in EN_src] if transformer else EN_src
                Zs = sample_Z_from_raster(src, EN_raster)
                coordsENZ = [(EN_src[i][0], EN_src[i][1], Zs[i]) for i in range(len(EN_src))]

                # 4) memorial + tabela
                memorial_md, df = build_memorial(coordsENZ)

                # 5) salvar
                (out_dir / f"memorial_{ident}.md").write_text(memorial_md + "\n", encoding="utf-8")
                df.to_excel(out_dir / f"tabela_{ident}.xlsx", index=False, sheet_name="Coordenadas")

                # 6) SHP de pontos (opcional)
                if export_points:
                    pts = coordsENZ[:-1] if coordsENZ[0][:2] == coordsENZ[-1][:2] else coordsENZ
                    pid = list(range(1, len(pts)+1))
                    gdf_pts = gpd.GeoDataFrame({
                        "POLY_ID": [ident]*len(pts),
                        "PID": pid,
                        "P_NAME": [f"P{i}" for i in pid],
                        "E": [round(p[0], 4) for p in pts],
                        "N": [round(p[1], 4) for p in pts],
                        "Z": [None if (np.isnan(p[2]) or np.isinf(p[2])) else round(p[2], 2) for p in pts],
                    }, geometry=[Point(p[0], p[1]) for p in pts], crs=shp_crs)
                    shp_out = out_dir / f"pontos_{ident}.shp"
                    gdf_pts.to_file(shp_out, driver="ESRI Shapefile", encoding="utf-8")
                    log_cb(f"    ↳ SHP de pontos: {shp_out.name} ({len(gdf_pts)} vértices)")

                log_cb(f"[OK] {ident}: {len(coordsENZ)-1} lados → memorial_*.md, tabela_*.xlsx")

        log_cb(f"Concluído. {total_polys} polígonos processados.")
    return True


# ================== Interface (Tkinter) ==================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Memorial Planimétrico/Planialtimétrico (SHP + MDT)")
        self.geometry("880x650")

        self.shp_path = tk.StringVar()
        self.dem_path = tk.StringVar()
        self.out_dir  = tk.StringVar()
        self.id_field = tk.StringVar(value="(nenhum)")
        self.export_points = tk.BooleanVar(value=True)

        # Opções novas
        self.start_rule = tk.StringVar(value="SW")     # SW/SE/NE/NW
        self.clockwise  = tk.StringVar(value="CW")     # CW/CCW
        self.refE = tk.StringVar(value="")
        self.refN = tk.StringVar(value="")
        self.use_ref = tk.BooleanVar(value=False)

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        # SHP
        ttk.Label(frm, text="Shapefile (polígonos):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.shp_path, width=80).grid(row=1, column=0, columnspan=2, sticky="we", pady=2)
        ttk.Button(frm, text="Selecionar...", command=self.pick_shp).grid(row=1, column=2, padx=6)

        # MDT
        ttk.Label(frm, text="MDT (GeoTIFF):").grid(row=2, column=0, sticky="w", pady=(10,0))
        ttk.Entry(frm, textvariable=self.dem_path, width=80).grid(row=3, column=0, columnspan=2, sticky="we", pady=2)
        ttk.Button(frm, text="Selecionar...", command=self.pick_dem).grid(row=3, column=2, padx=6)

        # OUT
        ttk.Label(frm, text="Pasta de saída:").grid(row=4, column=0, sticky="w", pady=(10,0))
        ttk.Entry(frm, textvariable=self.out_dir, width=80).grid(row=5, column=0, columnspan=2, sticky="we", pady=2)
        ttk.Button(frm, text="Selecionar...", command=self.pick_outdir).grid(row=5, column=2, padx=6)

        # ID + checkbox pontos
        ttk.Label(frm, text="Campo ID (opcional):").grid(row=6, column=0, sticky="w", pady=(10,0))
        self.id_combo = ttk.Combobox(frm, textvariable=self.id_field, state="readonly", width=32, values=["(nenhum)"])
        self.id_combo.grid(row=7, column=0, sticky="w")
        self.id_combo.current(0)
        ttk.Checkbutton(frm, text="Exportar SHP de pontos (P1…Pn)", variable=self.export_points).grid(row=7, column=1, sticky="w")

        # ======== Opções de início e sentido ========
        box = ttk.LabelFrame(frm, text="Numeração e orientação", padding=8)
        box.grid(row=8, column=0, columnspan=3, sticky="we", pady=(12,6))

        ttk.Label(box, text="Regra do Ponto Inicial:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(box, textvariable=self.start_rule, state="readonly",
                     values=["SW","SE","NE","NW"], width=6).grid(row=0, column=1, sticky="w", padx=(6,12))

        ttk.Label(box, text="Sentido do percurso:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(box, textvariable=self.clockwise, state="readonly",
                     values=["CW (horário)","CCW (anti-horário)"], width=20).grid(row=0, column=3, sticky="w", padx=(6,12))
        # normalizar internamente depois

        ttk.Checkbutton(box, text="Forçar início pelo ponto mais próximo de (E,N):", variable=self.use_ref).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6,0))
        ttk.Entry(box, textvariable=self.refE, width=18).grid(row=1, column=2, sticky="w", padx=(6,2))
        ttk.Entry(box, textvariable=self.refN, width=18).grid(row=1, column=3, sticky="w")

        # Logs
        ttk.Label(frm, text="Logs:").grid(row=9, column=0, sticky="w", pady=(12,0))
        self.log = tk.Text(frm, height=14)
        self.log.grid(row=10, column=0, columnspan=3, sticky="nsew")
        frm.rowconfigure(10, weight=1)
        frm.columnconfigure(1, weight=1)

        # Executar
        ttk.Button(frm, text="Executar", command=self.run).grid(row=11, column=2, sticky="e", pady=(8,0))

    def log_print(self, msg):
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.update_idletasks()

    def pick_shp(self):
        p = filedialog.askopenfilename(title="Selecionar Shapefile",
                                       filetypes=[("Shapefile","*.shp"),("Todos","*.*")])
        if p:
            self.shp_path.set(p)
            try:
                gdf = gpd.read_file(p)
                cols = ["(nenhum)"] + [c for c in gdf.columns if c.lower() != "geometry"]
                self.id_combo["values"] = cols
                self.id_combo.current(0)
            except Exception as e:
                messagebox.showwarning("Aviso", f"Não foi possível ler os campos do SHP.\n{e}")

    def pick_dem(self):
        p = filedialog.askopenfilename(title="Selecionar MDT (GeoTIFF)",
                                       filetypes=[("GeoTIFF","*.tif *.tiff"),("Todos","*.*")])
        if p:
            self.dem_path.set(p)

    def pick_outdir(self):
        p = filedialog.askdirectory(title="Selecionar pasta de saída")
        if p:
            self.out_dir.set(p)

    def run(self):
        shp = self.shp_path.get().strip()
        dem = self.dem_path.get().strip()
        outd = self.out_dir.get().strip()
        idf = self.id_field.get()
        if idf == "(nenhum)":
            idf = None

        if not shp or not Path(shp).exists():
            messagebox.showerror("Erro", "Informe um Shapefile válido.")
            return
        if not dem or not Path(dem).exists():
            messagebox.showerror("Erro", "Informe um MDT GeoTIFF válido.")
            return
        if not outd:
            messagebox.showerror("Erro", "Informe a pasta de saída.")
            return

        # interpretar opções
        rule = self.start_rule.get()
        cw = True if self.clockwise.get().startswith("CW") else False
        start_near = None
        if self.use_ref.get():
            try:
                e = float(self.refE.get().replace(",", "."))
                n = float(self.refN.get().replace(",", "."))
                start_near = (e, n)
            except:
                messagebox.showerror("Erro", "Coordenadas E/N inválidas para o ponto de referência.")
                return

        try:
            self.log_print(f"Iniciando…\nSHP: {shp}\nMDT: {dem}\nSaída: {outd}\nID: {idf or '(nenhum)'}")
            self.log_print(f"Regra Ponto 1: {rule} | Sentido: {'Horário' if cw else 'Anti-horário'} | Forçar início por referência: {bool(start_near)}")
            process_all(shp, dem, outd,
                        id_field=idf,
                        export_points=True,
                        start_corner=rule,
                        start_near=start_near,
                        clockwise=cw,
                        log_cb=self.log_print)
            self.log_print("✓ Finalizado.")
            messagebox.showinfo("Concluído", "Processamento finalizado com sucesso.")
        except Exception as e:
            self.log_print(f"[ERRO] {e}")
            messagebox.showerror("Erro", str(e))


if __name__ == "__main__":
    app = App()
    app.mainloop()
