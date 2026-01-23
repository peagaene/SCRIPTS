# ORTO_LOAD_TK.py — GUI com Tkinter (sem PySimpleGUI)
import os, tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import arcpy

arcpy.env.parallelProcessingFactor = "50%"
arcpy.env.overwriteOutput = True

def pick_gdb():
    path = filedialog.askdirectory(title="Selecione a File GDB (.gdb)")
    if path and path.lower().endswith(".gdb"):
        gdb_var.set(path)
        # listar raster datasets
        ws_old = arcpy.env.workspace
        arcpy.env.workspace = path
        try:
            rasters = arcpy.ListRasters() or []
            rasters = [r for r in rasters
                       if arcpy.Describe(os.path.join(path, r)).dataType == "RasterDataset"]
        finally:
            arcpy.env.workspace = ws_old
        raster_list.delete(0, tk.END)
        for r in rasters:
            raster_list.insert(tk.END, r)
    elif path:
        messagebox.showerror("Erro", "Escolha uma pasta .gdb válida.")

def pick_tifs():
    files = filedialog.askopenfilenames(title="Selecione os GeoTIFFs",
                                        filetypes=[("GeoTIFF", "*.tif *.tiff")])
    if files:
        for f in files:
            if f not in tifs:
                tifs.append(f)
                tifs_list.insert(tk.END, f)

def del_sel_tifs():
    sel = list(tifs_list.curselection())
    for i in reversed(sel):
        tifs.pop(i)
        tifs_list.delete(i)

def run():
    gdb = gdb_var.get().strip()
    if not gdb or not os.path.isdir(gdb) or not gdb.lower().endswith(".gdb"):
        messagebox.showerror("Erro", "Selecione uma File GDB válida.")
        return
    if raster_list.curselection() == ():
        messagebox.showerror("Erro", "Selecione um Raster Dataset.")
        return
    if not tifs:
        messagebox.showerror("Erro", "Adicione pelo menos um TIFF.")
        return

    dataset = raster_list.get(raster_list.curselection()[0])
    target = os.path.join(gdb, dataset)
    if not arcpy.Exists(target):
        messagebox.showerror("Erro", f"Raster Dataset não encontrado:\n{target}")
        return

    # opcional: chunk
    try:
        chunk = max(1, int(chunk_var.get()))
    except Exception:
        chunk = 400

    # checa bandcount do destino
    try:
        bcount = arcpy.Describe(target).bandCount
    except Exception:
        bcount = None

    # filtra por extensão e bandas
    tifs_ok = []
    for p in tifs:
        if not p.lower().endswith((".tif", ".tiff")):
            continue
        if bcount is not None:
            try:
                if getattr(arcpy.Describe(p), "bandCount", None) not in (None, bcount):
                    continue
            except Exception:
                pass
        tifs_ok.append(p)

    if not tifs_ok:
        messagebox.showerror("Erro", "Nenhum TIFF válido para mosaicar.")
        return

    # executa Mosaic em blocos
    try:
        for i in range(0, len(tifs_ok), chunk):
            part = tifs_ok[i:i+chunk]
            arcpy.management.Mosaic(
                inputs=";".join(part),
                target=target,
                mosaic_type="LAST",
                colormap="FIRST",
                background_value="#",
                nodata_value="#",
                onebit_to_eightbit="NONE",
                mosaicking_tolerance=0.0,
            )
        # pirâmides e estatísticas
        arcpy.management.BuildPyramidsAndStatistics(
            in_raster_dataset=target,
            build_pyramids="BUILD_PYRAMIDS",
            calculate_statistics="CALCULATE_STATISTICS",
            skip_existing="SKIP_EXISTING",
            pyramid_level="-1",
            pyramid_resampling_technique="NEAREST"
        )
        messagebox.showinfo("OK", f"Concluído:\n{target}")
    except Exception as e:
        messagebox.showerror("Erro", str(e))

# GUI
root = tk.Tk()
root.title("Carregar ortofotos em Raster Dataset (GDB) — Tkinter")
root.geometry("900x600")

gdb_var = tk.StringVar()
chunk_var = tk.StringVar(value="400")
tifs = []

frm = tk.Frame(root); frm.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(frm, text="File GDB:").grid(row=0, column=0, sticky="w")
tk.Entry(frm, textvariable=gdb_var, width=80).grid(row=0, column=1, sticky="we")
tk.Button(frm, text="Procurar…", command=pick_gdb).grid(row=0, column=2, padx=5)

tk.Label(frm, text="Raster Dataset:").grid(row=1, column=0, sticky="w", pady=(8,0))
raster_list = tk.Listbox(frm, height=6, width=60, exportselection=False)
raster_list.grid(row=1, column=1, columnspan=2, sticky="we", pady=(8,0))

tk.Label(frm, text="TIFFs selecionados:").grid(row=2, column=0, sticky="w", pady=(12,0))
tifs_list = tk.Listbox(frm, height=14, width=100, selectmode=tk.EXTENDED)
tifs_list.grid(row=3, column=0, columnspan=3, sticky="nsew")
frm.grid_rowconfigure(3, weight=1)
frm.grid_columnconfigure(1, weight=1)

btns = tk.Frame(frm); btns.grid(row=4, column=0, columnspan=3, sticky="w", pady=6)
tk.Button(btns, text="Adicionar TIFFs…", command=pick_tifs).pack(side="left", padx=4)
tk.Button(btns, text="Remover selecionados", command=del_sel_tifs).pack(side="left", padx=4)

opts = tk.Frame(frm); opts.grid(row=5, column=0, columnspan=3, sticky="we", pady=6)
tk.Label(opts, text="Tamanho do bloco:").pack(side="left")
tk.Entry(opts, textvariable=chunk_var, width=6).pack(side="left", padx=6)
tk.Button(opts, text="Executar", command=run).pack(side="right")

root.mainloop()
