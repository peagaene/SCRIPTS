# perimetro.py
"""
Script standalone que lê um shapefile de polígono, numera os vértices
(formato P-001), monta tabela (Vértice, Azimute, Distância, E, N, Altitude)
e salva um DXF pronto para plotar.

Suporta leitura de MDT para preencher a coluna de altitude.
Requer: shapely, ezdxf e (fiona ou pyshp/shapefile) para ler o shapefile.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import ezdxf

try:
    from shapely.geometry import Point, Polygon, shape
except ImportError as exc:
    raise SystemExit("Dependência ausente: instale shapely (pip install shapely).") from exc

try:
    import rasterio
except ImportError:
    rasterio = None

try:  # fiona é mais robusto; usa se existir
    import fiona
except ImportError:  # pragma: no cover - ambiente sem fiona
    fiona = None

try:  # fallback: pyshp
    import shapefile  # type: ignore
except ImportError:  # pragma: no cover - ambiente sem pyshp
    shapefile = None

# ---------------- Parâmetros e simbologia mínima ----------------
STYLE_TEXTO = "Arial"
BLOCO_VERTICE = "VERTICE"
LAYER_VERTICE_PTO = "SIMBOLOGIA_VERTICES_PERIMETRO"
LAYER_VERTICE_TXT = "SIMBOLOGIA_VERTICES_PERIMETRO"
LAYER_TABELA = "TABELA_ALINHAMENTOS"


@dataclass
class Params:
    altura_texto_P: float = 2
    p_label_offset_m: float = 1.8
    p_label_offset_step: float = 0.5
    altura_texto_tabela: float = 2.0
    tabela_cell_w: float = 20.0
    tabela_cell_h: float = 6.0
    tabela_offset_x: float = 120.0
    tabela_offset_y: float = 0.0
    style_texto: str = STYLE_TEXTO


def _vertices_do_poligono(poly: Polygon) -> List[Tuple[float, float]]:
    """Extrai vértices (sem o último duplicado) e rotaciona para começar no ponto mais ao norte."""
    coords = list(poly.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    verts: list[tuple[float, float]] = []
    for c in coords:
        x, y = c[0], c[1]  # aceita (x,y) ou (x,y,z)
        verts.append((float(x), float(y)))
    if not verts:
        return verts
    start_idx = max(range(len(verts)), key=lambda i: (verts[i][1], -verts[i][0]))
    return verts[start_idx:] + verts[:start_idx]


def _dms_str(az_deg: float) -> str:
    d = int(az_deg)
    m_f = (az_deg - d) * 60.0
    m = int(m_f)
    s = int(round((m_f - m) * 60.0))
    if s == 60:
        s = 0
        m += 1
    if m == 60:
        m = 0
        d = (d + 1) % 360
    deg = "\u00B0"
    return f"{d:02d}{deg}{m:02d}'{s:02d}''"


def _dist(p1, p2) -> float:
    return float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def _azimute(p1, p2) -> float:
    dE = p2[0] - p1[0]
    dN = p2[1] - p1[1]
    ang = math.degrees(math.atan2(dE, dN))
    if ang < 0:
        ang += 360.0
    return float(ang)


def _bbox_max(vertices):
    xs = [p[0] for p in vertices]
    ys = [p[1] for p in vertices]
    return max(xs), max(ys)


def _add_text_center(ms, text, cx, cy, h, layer, style=STYLE_TEXTO):
    t = ms.add_text(text, dxfattribs={"height": h, "style": style, "layer": layer})
    t.dxf.insert = (cx, cy)
    t.dxf.align_point = (cx, cy)
    t.dxf.halign = 1
    t.dxf.valign = 2
    return t


def _draw_cell(ms, x, y, w, h, layer):
    ms.add_lwpolyline(
        [(x, y), (x + w, y), (x + w, y - h), (x, y - h), (x, y)],
        close=True,
        dxfattribs={"layer": layer},
    )


def _table_header(ms, x0, y0, cw, ch, txt_h, layer, style):
    # Segmento | Azimute | Distância | E | N | Altitude
    headers = ["Segmento", "Azimute", "Distancia (m)", "E", "N", "Altitude (m)"]
    x = x0
    for htxt in headers:
        _draw_cell(ms, x, y0, cw, ch, layer)
        _add_text_center(ms, htxt, x + cw / 2, y0 - ch / 2, txt_h, layer, style)
        x += cw


def _table_row(ms, row_idx, x0, y0, cw, ch, txt_h, layer, style, pid, az, dist, e, n, alt):
    y = y0 - (1 + row_idx) * ch
    alt_txt = "" if alt is None else f"{alt:.2f}".replace(".", ",")
    dist_txt = f"{dist:.2f}".replace(".", ",")
    e_txt = f"{e:.3f}".replace(".", ",")
    n_txt = f"{n:.3f}".replace(".", ",")
    cols = [pid, _dms_str(az), dist_txt, e_txt, n_txt, alt_txt]
    col_positions = [x0 + i * cw for i in range(6)]
    for x, val in zip(col_positions, cols):
        _draw_cell(ms, x, y, cw, ch, layer)
        _add_text_center(ms, val, x + cw / 2, y - ch / 2, txt_h, layer, style)


def processar_perimetros(
    ms,
    doc,
    params: Params,
    perimetros: list,
    perimetro_limite=None,
    altitudes: list[float | None] | None = None,
):
    """Usa o primeiro polígono recebido."""
    if not perimetros:
        return
    poly = perimetros[0]
    style_texto = getattr(params, "style_texto", STYLE_TEXTO)
    V = _vertices_do_poligono(poly)

    # 1) Inserir VERTICE e rótulos P-001
    ref_geom = perimetro_limite if perimetro_limite is not None else poly
    try:
        centroid = ref_geom.centroid if ref_geom is not None else poly.centroid
    except Exception:
        centroid = None
    try:
        boundary_geom = ref_geom.buffer(0) if ref_geom is not None else poly.buffer(0)
    except Exception:
        boundary_geom = None
    offset_m = float(getattr(params, "p_label_offset_m", 1.8))
    step_m = max(0.1, float(getattr(params, "p_label_offset_step", 0.5)))

    for i, (x, y) in enumerate(V, start=1):
        pid_label = f"P-{i:03d}"
        ms.add_blockref(BLOCO_VERTICE, (x, y), dxfattribs={"layer": LAYER_VERTICE_PTO})
        px, py = float(x), float(y)
        if centroid is not None:
            dir_x = px - float(centroid.x)
            dir_y = py - float(centroid.y)
            norm = float(math.hypot(dir_x, dir_y))
            if norm < 1e-6:
                dir_x, dir_y = 1.0, 0.0
            else:
                dir_x /= norm
                dir_y /= norm
            dist = offset_m
            candidate = Point(px + dir_x * dist, py + dir_y * dist)
            tries = 0
            while (
                boundary_geom is not None
                and (boundary_geom.contains(candidate) or boundary_geom.distance(candidate) < 0.05)
                and tries < 40
            ):
                dist += step_m
                candidate = Point(px + dir_x * dist, py + dir_y * dist)
                tries += 1
            px, py = candidate.x, candidate.y
        else:
            px += offset_m
            py += offset_m
        _add_text_center(ms, pid_label, px, py, params.altura_texto_P, LAYER_VERTICE_TXT, style_texto)

    # 2) Tabela (Vértice / Azimute / Distância / E / N / Altitude)
    max_x, max_y = _bbox_max(V)
    x0 = max_x + params.tabela_offset_x
    y0 = max_y + params.tabela_offset_y
    cw, ch, th = params.tabela_cell_w, params.tabela_cell_h, params.altura_texto_tabela

    _table_header(ms, x0, y0, cw, ch, th, LAYER_TABELA, style_texto)

    n = len(V)
    for i in range(n):
        a = V[i]
        b = V[(i + 1) % n]
        dist = _dist(a, b)
        az = _azimute(a, b)
        pid = f"P-{i+1:03d} - P-{(i+1) % n + 1:03d}"
        e, ncoord = a[0], a[1]
        alt_val = None
        if altitudes and i < len(altitudes):
            alt_val = altitudes[i]
        _table_row(ms, i, x0, y0, cw, ch, th, LAYER_TABELA, style_texto, pid, az, dist, e, ncoord, alt_val)

    try:
        setattr(params, "area_table_anchor", (x0, y0 - (1 + n) * ch - ch))
    except Exception:
        pass


# ---------------- Utilidades para rodar standalone ----------------
def _ensure_layer(doc, name: str):
    if name not in doc.layers:
        doc.layers.new(name=name)


def _ensure_text_style(doc, name: str):
    if name not in doc.styles:
        doc.styles.new(name, dxfattribs={"font": "arial.ttf"})


def _ensure_vertice_block(doc):
    """Cria um bloco simples de vértice (dois círculos concêntricos) se não existir."""
    if BLOCO_VERTICE in doc.blocks:
        return
    blk = doc.blocks.new(BLOCO_VERTICE)
    blk.add_circle((0, 0), 0.25)  # diâmetro 0.5
    blk.add_circle((0, 0), 0.50)  # diâmetro 1.0


def _criar_documento(params: Params):
    doc = ezdxf.new(setup=True)
    _ensure_text_style(doc, params.style_texto)
    for layer in (LAYER_VERTICE_PTO, LAYER_VERTICE_TXT, LAYER_TABELA):
        _ensure_layer(doc, layer)
    _ensure_vertice_block(doc)
    return doc, doc.modelspace()


def _sample_altitudes_mdt(path: Path, vertices: list[tuple[float, float]], nodata: float | None):
    if rasterio is None:
        raise SystemExit("Para ler MDT informe rasterio instalado (pip install rasterio).")
    vals: list[float | None] = []
    with rasterio.open(path) as src:
        nodata_val = src.nodata if nodata is None else nodata
        for (x, y) in vertices:
            sample = next(src.sample([(x, y)]))
            val = float(sample[0])
            if (nodata_val is not None and math.isclose(val, nodata_val)) or math.isnan(val):
                vals.append(None)
            else:
                vals.append(val)
    return vals


def _load_polygons_fiona(path: Path, layer_field: str | None, layer_value: str | None):
    if fiona is None:
        return []
    polys: list[Polygon] = []
    with fiona.open(path) as src:
        for feat in src:
            props = feat.get("properties") or {}
            if layer_field and layer_value and str(props.get(layer_field)) != str(layer_value):
                continue
            geom = shape(feat["geometry"])
            if geom.is_empty:
                continue
            if geom.geom_type == "Polygon":
                polys.append(Polygon(geom.exterior.coords))
            elif geom.geom_type == "MultiPolygon":
                biggest = max(geom.geoms, key=lambda g: g.area)
                polys.append(Polygon(biggest.exterior.coords))
    return polys


def _load_polygons_pyshp(path: Path, layer_field: str | None, layer_value: str | None):
    if shapefile is None:
        return []
    polys: list[Polygon] = []
    reader = shapefile.Reader(str(path))
    field_names = [f[0] for f in reader.fields[1:]]
    layer_idx = field_names.index(layer_field) if layer_field and layer_field in field_names else None
    for sr in reader.iterShapeRecords():
        if layer_idx is not None:
            val = sr.record[layer_idx]
            if str(val) != str(layer_value):
                continue
        geom = shape(sr.shape.__geo_interface__)
        if geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polys.append(Polygon(geom.exterior.coords))
        elif geom.geom_type == "MultiPolygon":
            biggest = max(geom.geoms, key=lambda g: g.area)
            polys.append(Polygon(biggest.exterior.coords))
    return polys


def carregar_poligonos(path: Path, layer_field: str | None = None, layer_value: str | None = None) -> list[Polygon]:
    """Lê polígonos do shapefile usando fiona ou pyshp."""
    if not path.exists():
        raise FileNotFoundError(f"Shapefile não encontrado: {path}")
    polys = _load_polygons_fiona(path, layer_field, layer_value)
    if not polys:
        polys = _load_polygons_pyshp(path, layer_field, layer_value)
    if not polys:
        raise RuntimeError("Nenhum polígono encontrado. Instale fiona ou pyshp, ou verifique filtros.")
    return polys


def parse_args():
    ap = argparse.ArgumentParser(description="Gera DXF de vértices/azimutes a partir de um shapefile de polígono.")
    ap.add_argument("shapefile", nargs="?", type=Path, help="Caminho do shapefile (extensão .shp).")
    ap.add_argument("-o", "--output", type=Path, default=Path("perimetro.dxf"), help="Caminho do DXF de saída.")
    ap.add_argument("--layer-field", default=None, help="Campo para filtrar (ex: Layer).")
    ap.add_argument("--layer-value", default=None, help="Valor do campo a manter.")
    ap.add_argument("--offset-label", type=float, default=None, help="Distância inicial para afastar o rótulo Pn (m).")
    ap.add_argument("--offset-step", type=float, default=None, help="Passo extra caso o afastamento colida (m).")
    ap.add_argument("--mdt", type=Path, default=None, help="Raster MDT para extrair altitude dos vértices.")
    ap.add_argument("--mdt-nodata", type=float, default=None, help="Valor de NODATA do MDT (opcional).")
    ap.add_argument("--gui", action="store_true", help="Abre interface para escolher arquivos/pastas.")
    return ap.parse_args()


def gerar_dxf(
    shapefile: Path,
    output: Path,
    layer_field: str | None,
    layer_value: str | None,
    offset_label: float | None,
    offset_step: float | None,
    mdt: Path | None,
    mdt_nodata: float | None,
):
    params = Params()
    if offset_label is not None:
        params.p_label_offset_m = float(offset_label)
    if offset_step is not None:
        params.p_label_offset_step = float(offset_step)

    polys = carregar_poligonos(shapefile, layer_field, layer_value)
    altitudes = None
    if mdt is not None:
        verts = _vertices_do_poligono(polys[0])
        altitudes = _sample_altitudes_mdt(mdt, verts, mdt_nodata)

    doc, ms = _criar_documento(params)
    processar_perimetros(ms, doc, params, polys, altitudes=altitudes)
    doc.saveas(output)
    return output


def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("Gerar DXF de Perímetro")
    root.geometry("580x280")

    def row(label, default=""):
        r = tk.Frame(root)
        tk.Label(r, text=label, width=20, anchor="w").pack(side="left")
        ent = tk.Entry(r, width=60)
        ent.insert(0, default)
        ent.pack(side="left", expand=True, fill="x")
        r.pack(fill="x", padx=8, pady=4)
        return ent

    ent_shp = row("Shapefile (.shp):")
    ent_mdt = row("MDT (opcional):")
    ent_out = row("DXF de saída:", "perimetro.dxf")

    def browse_file(entry, types, save=False):
        if save:
            path = filedialog.asksaveasfilename(defaultextension=types[0][1], filetypes=types)
        else:
            path = filedialog.askopenfilename(filetypes=types)
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    btn_frame = tk.Frame(root)
    tk.Button(btn_frame, text="Procurar SHP", command=lambda: browse_file(ent_shp, [("Shapefile", "*.shp")])).pack(
        side="left", padx=4
    )
    tk.Button(btn_frame, text="Procurar MDT", command=lambda: browse_file(ent_mdt, [("Raster", "*.tif;*.img;*.asc")])).pack(
        side="left", padx=4
    )
    tk.Button(
        btn_frame,
        text="Salvar DXF em...",
        command=lambda: browse_file(ent_out, [("DXF", "*.dxf")], save=True),
    ).pack(side="left", padx=4)
    btn_frame.pack(pady=4)

    status = tk.StringVar(value="")
    tk.Label(root, textvariable=status, fg="blue").pack(pady=2)

    def on_generate():
        shp = ent_shp.get().strip()
        out = ent_out.get().strip()
        mdt = ent_mdt.get().strip() or None
        if not shp:
            messagebox.showerror("Erro", "Informe o shapefile.")
            return
        if not out:
            messagebox.showerror("Erro", "Informe o caminho do DXF de saída.")
            return
        try:
            output_path = gerar_dxf(
                Path(shp),
                Path(out),
                None,
                None,
                None,
                None,
                Path(mdt) if mdt else None,
                None,
            )
            status.set(f"DXF salvo em: {output_path}")
            messagebox.showinfo("Sucesso", f"DXF gerado: {output_path}")
        except Exception as exc:  # pragma: no cover - GUI
            messagebox.showerror("Erro", str(exc))

    tk.Button(root, text="Gerar DXF", command=on_generate, width=20).pack(pady=8)
    root.mainloop()


def main():
    args = parse_args()
    # Abre GUI automaticamente se nenhum shapefile for informado ou se --gui for passado
    if args.gui or not args.shapefile:
        run_gui()
        return
    output_path = gerar_dxf(
        args.shapefile,
        args.output,
        args.layer_field,
        args.layer_value,
        args.offset_label,
        args.offset_step,
        args.mdt,
        args.mdt_nodata,
    )
    print(f"DXF salvo em: {output_path}")


if __name__ == "__main__":
    main()
