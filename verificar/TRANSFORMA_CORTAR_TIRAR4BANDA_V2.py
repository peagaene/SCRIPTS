# === PROCESSAMENTO — LÓGICA ORIGINAL + PATCHES + JP2 EM PASTA GLOBAL ===
# ➤ TIF primeiro (reprojeta se preciso, recorta, drop alpha/copia) → depois JP2
# ➤ JP2 escrito DIRETO na pasta informada na UI (sem subpastas)
# ➤ Only-JP2: ignora processamento e gera JP2 direto do original
# ➤ UI: pares com fila dinâmica, “Enfileirar este/Todos”, status; caminho global de JP2 habilitado pelo checkbox
# ➤ Performance: fecha datasets sempre, sindex + unary_union local, GDAL cache↑, OPJ threads↓, logs em lote, gc periódico

import os, shutil, threading, time, math, gc
from datetime import datetime, timedelta
from queue import Queue, Empty

import PySimpleGUI as sg
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.enums import Resampling
from shapely.geometry import box, mapping
from osgeo import gdal, osr

# === CONFIG GLOBAL ===
gdal.UseExceptions()
gdal.SetConfigOption("CHECK_DISK_FREE_SPACE", "FALSE")
gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
gdal.SetCacheMax(1024 * 1024 * 1024)        # ~1 GiB
os.environ.setdefault("GDAL_CACHEMAX", "1024")
os.environ.setdefault("OPJ_NUM_THREADS", str(max(1, math.floor((os.cpu_count() or 4) / 2))))

TARGET_EPSG = "EPSG:31983"
LOG_PATH = os.path.join(os.path.expanduser("~"), "log_processamento_completo.txt")
DEFAULT_SHP = r"\\192.168.2.29\d\2212_GOV_SAO_PAULO\BLOCO.shp"
JP2_QUALITY_PERCENT = 10

# === EVENTOS DA UI ===
EVT_STATUS   = "-STATUS-"
EVT_PROGRESS = "-PROGRESS-"
EVT_FINISHED = "-FINISHED-"
EVT_LOG      = "-LOG-"

# === Notificação (Windows) ===
def flash_taskbar(window, count=8):
    if os.name != "nt":
        return
    try:
        import ctypes
        class FLASHWINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("hwnd", ctypes.c_void_p),
                        ("dwFlags", ctypes.c_uint), ("uCount", ctypes.c_uint), ("dwTimeout", ctypes.c_uint)]
        user32 = ctypes.windll.user32
        hwnd = int(window.TKroot.winfo_id())
        info = FLASHWINFO(ctypes.sizeof(FLASHWINFO), hwnd, 0x00000006, count, 0)  # TRAY+TIMER
        user32.FlashWindowEx(ctypes.byref(info))
    except Exception:
        pass

# === HELPERS ===
def robust_remove(path, tries=12, delay=0.2):
    for i in range(tries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return True
        except (PermissionError, OSError):
            time.sleep(delay * (i + 1))
    return False

def salvar_log(linhas):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n=== Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for l in linhas: f.write(l + "\n")
        f.write(f"=== Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

def count_tifs(root_dir):
    c = 0
    for dp, _, fs in os.walk(root_dir):
        for f in fs:
            if f.lower().endswith(".tif"):
                c += 1
    return c

def get_epsg_code(img_path):
    try:
        ds = gdal.Open(img_path)
        if ds is None: return None
        proj = ds.GetProjection()
        ds = None
        if not proj: return None
        srs = osr.SpatialReference(); srs.ImportFromWkt(proj)
        if srs.IsProjected():  return srs.GetAuthorityCode("PROJCS")
        if srs.IsGeographic(): return srs.GetAuthorityCode("GEOGCS")
        return None
    except Exception:
        return None

def assign_epsg_inplace(img, epsg=31983):
    ds = gdal.Open(img, gdal.GA_Update)
    if ds is None: raise RuntimeError("Não abriu dataset para SetProjection.")
    srs = osr.SpatialReference(); srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt()); ds = None

def has_georef(path: str) -> bool:
    try:
        ds = gdal.Open(path)
        if ds is None: return False
        proj = ds.GetProjection()
        gt = ds.GetGeoTransform(can_return_null=True)
        ds = None
        if not proj or gt is None: return False
        return not (gt[0]==0 and gt[1]==1 and gt[2]==0 and gt[3]==0 and gt[4]==0 and gt[5]==1)
    except Exception:
        return False

def make_gdal_progress_cb(force_event, window, get_total, processados, prefix):
    last = [0.0]
    def cb(complete, message, user_data):
        if force_event.is_set(): return 0
        if complete - last[0] >= 0.05:
            last[0] = complete
            window.write_event_value(EVT_PROGRESS, (processados, get_total(), f"{prefix} {int(complete*100)}%"))
        return 1
    return cb

def gdal_warp_to_31983(src_path, dst_path, force_event, window, get_total, processados):
    cb = make_gdal_progress_cb(force_event, window, get_total, processados, "Reprojetando")
    opts = gdal.WarpOptions(dstSRS=TARGET_EPSG, multithread=True, dstAlpha=True,
                            creationOptions=["TILED=YES","BIGTIFF=IF_SAFER"], callback=cb)
    ds = gdal.Warp(dst_path, src_path, options=opts)
    if ds is None: raise RuntimeError("gdal.Warp falhou.")
    ds = None

def gdal_translate_drop_alpha_or_copy(src_path, dst_path, force_event, window, get_total, processados):
    """Se >=4 bandas, pega 1-2-3; se 1-3 bandas, só copia (GTiff tiled, COMPRESS=NONE)."""
    cb = make_gdal_progress_cb(force_event, window, get_total, processados, "Gerando TIF")
    ds0 = gdal.Open(src_path)
    if ds0 is None: raise RuntimeError("Falha ao abrir para Translate.")
    bands = ds0.RasterCount; ds0 = None
    bandList = [1,2,3] if bands >= 4 else None
    opts = gdal.TranslateOptions(format="GTiff",
                                 creationOptions=["TILED=YES","INTERLEAVE=PIXEL","COMPRESS=NONE"],
                                 bandList=bandList,
                                 callback=cb)
    ds = gdal.Translate(dst_path, src_path, options=opts)
    if ds is None: raise RuntimeError("gdal.Translate falhou (TIF).")
    ds = None

def gdal_translate_to_jp2(src_path, jp2_path, force_event=None, window=None, get_total=None, processados=0):
    cb = make_gdal_progress_cb(force_event, window, get_total, processados, "Gerando JP2")
    co = [f"QUALITY={JP2_QUALITY_PERCENT}"]
    if has_georef(src_path):
        co += ["GMLJP2=YES","GeoJP2=YES"]
    opts = gdal.TranslateOptions(format="JP2OpenJPEG", creationOptions=co, callback=cb)
    ds = gdal.Translate(jp2_path, src_path, options=opts)
    if ds is None: raise RuntimeError("gdal.Translate falhou (JP2).")
    ds = None

def downscale_transform(transform, in_w, in_h, max_dim=2000):
    if max(in_w, in_h) <= max_dim:
        return transform, in_h, in_w
    scale = max(in_w, in_h) / float(max_dim)
    out_w = int(round(in_w / scale)); out_h = int(round(in_h / scale))
    new_transform = transform * Affine.scale(in_w/out_w, in_h/out_h)
    return new_transform, out_h, out_w

def is_already_clipped(src, geoms, nodata_pref=255):
    """True se todo o exterior do shape já está NODATA (ou alpha=0)."""
    t = src.transform; h, w = src.height, src.width
    new_t, oh, ow = downscale_transform(t, w, h, max_dim=2000)
    inside = geometry_mask(geoms, transform=new_t, out_shape=(oh, ow), invert=True, all_touched=False)
    outside = ~inside
    if src.count == 4:
        alpha = src.read(4, out_shape=(oh, ow), resampling=Resampling.nearest)
        return (alpha[outside] == 0).all()
    nod = src.nodata if src.nodata is not None else nodata_pref
    if nod is None: return False
    band1 = src.read(1, out_shape=(oh, ow), resampling=Resampling.nearest)
    return (band1[outside] == nod).all()

def compute_base_out(img_path, in_root, out_root):
    """Retorna a pasta .../IMAGENS_TRANSFORMADAS (base de saída do TIF)."""
    if out_root:
        rel_parent = os.path.relpath(os.path.dirname(img_path), in_root)
        parent = os.path.join(out_root, rel_parent)
    else:
        parent = os.path.dirname(img_path)
    base_out = os.path.join(parent, "IMAGENS_TRANSFORMADAS")
    return base_out

# === WORKER (consome a fila de pares) ===
def worker_process_queue(pairs_q: Queue, shapefile, jp2_base_dir, only_jp2, gerar_jp2_flag, substituir_original,
                         window, stop_event: threading.Event, force_event: threading.Event,
                         shared_total):
    try:
        # ---- LOG em lote ----
        LOG_FLUSH_EVERY = 20
        log_buffer = []
        def log(msg):
            log_buffer.append(msg)
            if len(log_buffer) >= LOG_FLUSH_EVERY:
                window.write_event_value(EVT_LOG, "\n".join(log_buffer))
                log_buffer.clear()

        # ---- Shapefile / sindex ----
        if not only_jp2:
            window.write_event_value(EVT_STATUS, "Lendo shapefile…")
            gdf = gpd.read_file(shapefile).to_crs(epsg=31983)
            geom_unificada = gdf.geometry.union_all()
            try:
                sindex = gdf.sindex
            except Exception:
                sindex = None
        else:
            gdf = None; geom_unificada = None; sindex = None

        processados = 0
        start = time.time()
        def get_total(): return shared_total["value"]

        idle_rounds = 0
        while not stop_event.is_set():
            try:
                in_root, out_root, pair_idx = pairs_q.get(timeout=1.0)
            except Empty:
                idle_rounds += 1
                if idle_rounds >= 5:
                    break
                continue
            idle_rounds = 0

            for dp, _, fs in os.walk(in_root):
                for f in fs:
                    if stop_event.is_set() or force_event.is_set(): break
                    if not f.lower().endswith(".tif"): continue

                    img = os.path.join(dp, f)
                    nome = os.path.basename(img)
                    base_out = compute_base_out(img, in_root, out_root)
                    os.makedirs(base_out, exist_ok=True)

                    saida_final = os.path.join(base_out, nome)
                    base_nome   = os.path.splitext(nome)[0]
                    cortada_tmp = os.path.join(base_out, f"recorte_{base_nome}.tif")
                    reproj_tmp  = os.path.join(base_out, f"temp_{base_nome}.tif")
                    # JP2 direto na pasta global informada
                    saida_jp2   = os.path.join(jp2_base_dir, f"{base_nome}.jp2") if jp2_base_dir else None

                    # === ONLY-JP2 ===
                    if only_jp2:
                        try:
                            if saida_jp2 is None:
                                log(f"❌ Sem pasta JP2 definida; pulando {nome}")
                            elif os.path.exists(saida_jp2):
                                log(f"🟡 JP2 já existe: {os.path.basename(saida_jp2)}")
                            else:
                                window.write_event_value(EVT_PROGRESS, (processados, get_total(), f"Gerando JP2: {nome}"))
                                gdal_translate_to_jp2(img, saida_jp2, force_event, window, get_total, processados)
                                log(f"✅ JP2 gerado: {nome}")
                        except Exception as e:
                            log(f"❌ Falha JP2 {nome}: {e}")
                        finally:
                            processados += 1
                            elapsed = max(1, time.time() - start)
                            total = get_total()
                            eta = int(elapsed * (total - processados) / max(1, processados))
                            window.write_event_value(EVT_PROGRESS, (processados, total, f"ETA ~ {str(timedelta(seconds=eta))}"))
                            if processados % 20 == 0: gc.collect()
                        continue

                    # === PIPELINE COMPLETO ===
                    window.write_event_value(EVT_PROGRESS, (processados, get_total(), f"Avaliando: {nome}"))

                    # 1) EPSG / reprojeção
                    epsg = get_epsg_code(img)
                    need_reproj = False
                    src_for_clip = img

                    if epsg == "31983":
                        need_reproj = False
                    elif not epsg:
                        try: assign_epsg_inplace(img, 31983)
                        except Exception as e: log(f"⚠️ SetProjection falhou ({nome}): {e}")
                        need_reproj = False
                    else:
                        need_reproj = True

                    if need_reproj:
                        try:
                            window.write_event_value(EVT_PROGRESS, (processados, get_total(), f"Reprojetando: {nome}"))
                            gdal_warp_to_31983(img, reproj_tmp, force_event, window, get_total, processados)
                            src_for_clip = reproj_tmp
                        except Exception as e:
                            log(f"❌ Erro reprojeção {nome}: {e}")
                            if os.path.exists(reproj_tmp): robust_remove(reproj_tmp)
                            processados += 1
                            window.write_event_value(EVT_PROGRESS, (processados, get_total(), ""))
                            if processados % 20 == 0: gc.collect()
                            continue

                    # 2) Recorte
                    try:
                        with rasterio.open(src_for_clip) as src:
                            geom_raster = box(*src.bounds)
                            if not geom_raster.intersects(geom_unificada):
                                window[f"-PAIRSTAT-{pair_idx}-"].update("Sem interseção", text_color="orange")
                                log(f"⏩ Sem interseção: {nome}")
                            else:
                                if sindex is not None:
                                    cand_idx = list(sindex.intersection(geom_raster.bounds))
                                    intersecao = gdf.iloc[cand_idx]
                                    intersecao = intersecao[intersecao.geometry.intersects(geom_raster)]
                                else:
                                    intersecao = gdf[gdf.geometry.intersects(geom_raster)]

                                if intersecao.empty:
                                    window[f"-PAIRSTAT-{pair_idx}-"].update("Interseção vazia", text_color="orange")
                                    log(f"⏩ Interseção vazia: {nome}")
                                else:
                                    need_alpha_drop = (src.count >= 4)
                                    geom_clip = intersecao.unary_union
                                    shapes = [mapping(geom_clip)]
                                    already_clip = is_already_clipped(src, shapes, nodata_pref=255)
                                    need_crop = not already_clip

                                    need_any = need_reproj or need_crop or need_alpha_drop or gerar_jp2_flag or substituir_original
                                    if not need_any:
                                        window[f"-PAIRSTAT-{pair_idx}-"].update("OK (pulado)", text_color="grey20")
                                        log(f"🟡 Pulado (já OK): {nome}")
                                    else:
                                        write_target = src_for_clip
                                        if need_crop:
                                            window.write_event_value(EVT_PROGRESS, (processados, get_total(), f"Recortando: {nome}"))
                                            img_cortada, out_transform = mask(
                                                src, shapes, crop=True, nodata=255
                                            )
                                            meta = src.meta.copy()
                                            meta.update({
                                                "driver":"GTiff",
                                                "height": img_cortada.shape[1],
                                                "width":  img_cortada.shape[2],
                                                "transform": out_transform,
                                                "nodata": 255,
                                                "dtype": src.dtypes[0]
                                            })
                                            os.makedirs(base_out, exist_ok=True)
                                            with rasterio.open(cortada_tmp, "w", **meta) as dest:
                                                dest.write(img_cortada)
                                            del img_cortada; time.sleep(0.03)
                                            write_target = cortada_tmp

                                        # 3) TIF primeiro
                                        tif_final_created = False
                                        try:
                                            if write_target == cortada_tmp:
                                                gdal_translate_drop_alpha_or_copy(write_target, saida_final, force_event, window, get_total, processados)
                                                tif_final_created = True
                                                robust_remove(cortada_tmp)
                                            elif write_target == src_for_clip:
                                                with rasterio.open(src_for_clip) as _src_tmp:
                                                    if _src_tmp.count >= 4:
                                                        gdal_translate_drop_alpha_or_copy(write_target, saida_final, force_event, window, get_total, processados)
                                                        tif_final_created = True
                                            if substituir_original and (tif_final_created or write_target != img):
                                                src_for_replace = saida_final if tif_final_created else write_target
                                                time.sleep(0.03)
                                                robust_remove(img)
                                                shutil.move(src_for_replace, img)
                                                log(f"   ↳ Original substituído: {nome}")
                                                jp2_input = img
                                            else:
                                                if tif_final_created:
                                                    log(f"✅ TIF: {saida_final}")
                                                jp2_input = saida_final if tif_final_created else write_target

                                        except Exception as e:
                                            log(f"❌ Erro TIF {nome}: {e}")
                                            jp2_input = write_target

                                        # 4) JP2 depois (na pasta global)
                                        try:
                                            if gerar_jp2_flag and jp2_base_dir:
                                                os.makedirs(jp2_base_dir, exist_ok=True)
                                                if not os.path.exists(saida_jp2):
                                                    window.write_event_value(EVT_PROGRESS, (processados, get_total(), f"Gerando JP2: {nome}"))
                                                    gdal_translate_to_jp2(jp2_input, saida_jp2, force_event, window, get_total, processados)
                                                    log(f"   ↳ JP2: {os.path.basename(saida_jp2)}")
                                                else:
                                                    log(f"   ↳ JP2 já existe: {os.path.basename(saida_jp2)}")
                                        except Exception as e:
                                            log(f"❌ Falha JP2 {nome}: {e}")

                    except Exception as e:
                        log(f"❌ Erro geral: {nome} — {e}")

                    # limpar reproj tmp
                    if src_for_clip != img and os.path.exists(src_for_clip):
                        robust_remove(src_for_clip)

                    processados += 1
                    elapsed = max(1, time.time() - start)
                    total = get_total()
                    eta = int(elapsed * (total - processados) / max(1, processados))
                    window.write_event_value(EVT_PROGRESS, (processados, total, f"ETA ~ {str(timedelta(seconds=eta))}"))
                    if processados % 20 == 0:
                        gc.collect()

        # flush final dos logs
        try:
            if log_buffer:
                window.write_event_value(EVT_LOG, "\n".join(log_buffer))
                log_buffer.clear()
        except Exception:
            pass

        window.write_event_value(EVT_FINISHED, f"Concluído. Processados {processados}/{shared_total['value']}. Log: {LOG_PATH}")

    except Exception as e:
        window.write_event_value(EVT_FINISHED, f"Falha inesperada: {e}")

# === INTERFACE ===
sg.theme("SystemDefaultForReal")

status_row = [
    sg.Text("Status:", font=("Segoe UI", 10, "bold")),
    sg.Text("Parado", key="-STATUS-TXT-", text_color="grey20", font=("Segoe UI", 10)),
    sg.Push(),
    sg.Text("Arquivos:", font=("Segoe UI", 10)),
    sg.Text("0/0", key="-COUNT-", font=("Segoe UI", 10, "bold")),
    sg.Text("  |  Tempo:", font=("Segoe UI", 10)),
    sg.Text("00:00:00", key="-ELAPSED-", font=("Segoe UI", 10, "bold")),
]

def make_pair_block(idx: int):
    return [sg.pin(
        sg.Frame(f"Par {idx+1}", [
            [sg.Text("Entrada", size=(12,1)),
             sg.Input(key=f"-IN-{idx}-", size=(60,1)),
             sg.FolderBrowse(key=f"-BIN-{idx}-")],
            [sg.Text("Saída (base TIF)", size=(12,1)),
             sg.Input(key=f"-OUT-{idx}-", size=(60,1)),
             sg.FolderBrowse(key=f"-BOUT-{idx}-")],
            [sg.Text("Status:", size=(12,1)),
             sg.Text("aguardando…", key=f"-PAIRSTAT-{idx}-", size=(40,1), text_color="grey20"),
             sg.Push(),
             sg.Button("Enfileirar este", key=f"-ENQ-{idx}-")]
        ], key=f"-PAIR-{idx}-", relief=sg.RELIEF_SUNKEN, pad=(0,6), expand_x=True)
    )]

pairs_col = sg.Column([make_pair_block(0)], key="-PAIRS-COL-", pad=(0,0),
                      scrollable=True, vertical_scroll_only=True,
                      size=(950,320), expand_x=True, expand_y=True)

layout = [
    [sg.Frame("Pares (Entrada → Saída base TIF)", [
        [pairs_col],
        [sg.Button("Adicionar par", key="-ADDPAIR-"),
         sg.Button("Remover último", key="-REMPAIR-", disabled=True),
         sg.Push(),
         sg.Button("Enfileirar todos", key="-ENQ-ALL-")]
    ], expand_x=True, expand_y=True)],
    [sg.Text("Shapefile de corte (.shp):")],
    [sg.Input(default_text=DEFAULT_SHP, key="-SHP-", size=(80,1)),
     sg.FileBrowse(file_types=(("Shapefile", "*.shp"),), key="-BROWSHP-", initial_folder=os.path.dirname(DEFAULT_SHP))],
    [sg.Checkbox("Gerar JP2 (≈1:10)", default=True, key="-JP2-")],
    [sg.Text("Pasta de saída do JP2:"), sg.Input(key="-JP2DIR-", size=(80,1), disabled=False),
     sg.FolderBrowse(key="-BROWSE-JP2-", disabled=False)],
    [sg.Checkbox("Substituir original após concluir (sem backup)", default=False, key="-REPLACE-")],
    status_row,
    [sg.ProgressBar(100, orientation='h', size=(80, 20), key='-PROG-')],
    [sg.Button("Iniciar", key="-START-", size=(12,1)),
     sg.Button("Cancelar", key="-CANCEL-", size=(12,1), disabled=True),
     sg.Button("Forçar parada", key="-FORCE-", size=(14,1), disabled=True),
     sg.Push(), sg.Button("Sair")],
    [sg.Output(size=(140, 22), key='-OUT-')]
]
window = sg.Window("Processamento de Imagens — GDAL (TIF→JP2, otimizado)", layout,
                   finalize=True, resizable=True)

# estado dos pares
pair_indices = [0]
next_pair_id = 1
hidden_stack = []

# execução
pairs_q     = Queue()
stop_event  = threading.Event()
force_event = threading.Event()
worker_th   = None
start_time  = None

# progresso
shared_total = {"value": 0}
processed_seen = 0

def add_pair_row(idx):
    window.extend_layout(window["-PAIRS-COL-"], [make_pair_block(idx)])
    window.refresh()
    try:
        canvas = window["-PAIRS-COL-"].Widget.canvas
        canvas.yview_moveto(1.0)
    except Exception:
        pass

def enqueue_pairs(idx_list, values, only_jp2):
    added = 0
    for idx in idx_list:
        in_dir  = (values.get(f"-IN-{idx}-")  or "").strip()
        out_dir = (values.get(f"-OUT-{idx}-") or "").strip()
        if not in_dir or not os.path.isdir(in_dir):
            window[f"-PAIRSTAT-{idx}-"].update("Entrada inválida", text_color="red")
            continue
        if out_dir and not os.path.isdir(out_dir):
            try: os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                window[f"-PAIRSTAT-{idx}-"].update(f"Saída inválida: {e}", text_color="red")
                continue
        qtd = count_tifs(in_dir)
        if qtd == 0:
            window[f"-PAIRSTAT-{idx}-"].update("Sem .tif", text_color="orange")
            continue
        shared_total["value"] += qtd
        window["-PROG-"].update(current_count=processed_seen, max=max(shared_total["value"],1))
        window["-COUNT-"].update(f"{processed_seen}/{shared_total['value']}")
        window["-OUT-"].print(f"➕ Par {idx+1} enfileirado ({qtd} imagens).")
        window[f"-PAIRSTAT-{idx}-"].update("na fila…", text_color="green")
        pairs_q.put((in_dir, out_dir if out_dir else None, idx))
        added += 1
    return added

def start_worker(jp2_base_dir, only_jp2, gerar_jp2_flag, substituir_original, shp):
    global worker_th, start_time
    start_time = time.time()
    worker_th = threading.Thread(
        target=worker_process_queue,
        args=(pairs_q, shp, jp2_base_dir, only_jp2, gerar_jp2_flag, substituir_original,
              window, stop_event, force_event, shared_total),
        daemon=True
    )
    worker_th.start()

# === LOOP ===
while True:
    event, values = window.read(timeout=200)
    if event == sg.WIN_CLOSED or event == "Sair":
        if worker_th and worker_th.is_alive():
            stop_event.set(); force_event.set()
            worker_th.join(timeout=2)
        break

    # habilita/desabilita campo JP2 conforme checkbox
    if event == "-JP2-" or event is None:
        jp2_on = values.get("-JP2-", True)
        window["-JP2DIR-"].update(disabled=not jp2_on)
        window["-BROWSE-JP2-"].update(disabled=not jp2_on)

    if start_time and worker_th and worker_th.is_alive():
        elapsed = timedelta(seconds=int(time.time() - start_time))
        window["-ELAPSED-"].update(str(elapsed))

    if event == "-ADDPAIR-":
        if hidden_stack:
            idx = hidden_stack.pop()
            window[f"-PAIR-{idx}-"].update(visible=True)
            pair_indices.append(idx)
        else:
            idx = next_pair_id
            add_pair_row(idx)
            pair_indices.append(idx)
            next_pair_id += 1
        window["-REMPAIR-"].update(disabled=(len(pair_indices)==1))

    elif event == "-REMPAIR-":
        if worker_th and worker_th.is_alive():
            sg.popup("Remover par desabilitado enquanto processa."); continue
        if len(pair_indices) > 1:
            idx = pair_indices.pop()
            window[f"-PAIR-{idx}-"].update(visible=False)
            hidden_stack.append(idx)
            for k in (f"-IN-{idx}-", f"-OUT-{idx}-"):
                if k in values: window[k].update("")
            window["-REMPAIR-"].update(disabled=(len(pair_indices)==1))
            window.refresh()

    if isinstance(event, str) and event.startswith("-ENQ-") and event.endswith("-"):
        try: idx = int(event.split("-")[2])
        except Exception: idx = None
        if idx is not None:
            only_jp2 = (values["-JP2-"] and not values["-REPLACE-"])
            if not worker_th or not worker_th.is_alive():
                sg.popup("Clique em Iniciar antes de enfileirar."); continue
            _ = enqueue_pairs([idx], values, only_jp2)

    if event == "-ENQ-ALL-":
        if not worker_th or not worker_th.is_alive():
            sg.popup("Clique em Iniciar antes de enfileirar."); continue
        only_jp2 = (values["-JP2-"] and not values["-REPLACE-"])
        added = enqueue_pairs(pair_indices, values, only_jp2)
        if added == 0:
            window["-OUT-"].print("Nada novo para enfileirar.")

    if event == "-START-":
        gerar_jp2_flag      = values["-JP2-"]
        substituir_original = values["-REPLACE-"]
        only_jp2 = (gerar_jp2_flag and not substituir_original)  # mesma regra de "only jp2"

        jp2_base_dir = (values.get("-JP2DIR-") or "").strip()
        if gerar_jp2_flag:
            if not jp2_base_dir:
                sg.popup_error("Informe a pasta de saída do JP2.")
                continue
            try:
                os.makedirs(jp2_base_dir, exist_ok=True)
            except Exception as e:
                sg.popup_error(f"Não foi possível criar a pasta de JP2:\n{e}")
                continue

        shp = values["-SHP-"]
        if not only_jp2:
            if not shp or not os.path.isfile(shp):
                sg.popup_error("Selecione um shapefile válido."); continue

        stop_event.clear(); force_event.clear()
        shared_total["value"] = 0; processed_seen = 0
        while not pairs_q.empty():
            try: pairs_q.get_nowait()
            except Exception: break

        _ = enqueue_pairs(pair_indices, values, only_jp2)
        if shared_total["value"] == 0:
            sg.popup_error("Nenhuma imagem encontrada nos pares informados."); continue

        window["-OUT-"].print("▶️ Iniciando…")
        window["-STATUS-TXT-"].update("Rodando", text_color="green")
        window["-PROG-"].update(current_count=0, max=max(shared_total["value"],1))
        window["-COUNT-"].update(f"0/{shared_total['value']}")
        window["-CANCEL-"].update(disabled=False)
        window["-FORCE-"].update(disabled=False)
        window["-REMPAIR-"].update(disabled=True)

        start_worker(jp2_base_dir, only_jp2, gerar_jp2_flag, substituir_original, shp)

    if event == "-CANCEL-":
        if worker_th and worker_th.is_alive():
            stop_event.set()
            window["-STATUS-TXT-"].update("Cancelando…", text_color="orange")
            window["-OUT-"].print("⛔ Cancelar solicitado (encerra após a etapa atual)…")

    if event == "-FORCE-":
        if worker_th and worker_th.is_alive():
            force_event.set()
            window["-STATUS-TXT-"].update("Forçando parada…", text_color="orange")
            window["-OUT-"].print("⛔ Força STOP solicitado (aborta operação atual)…")

    if event == EVT_PROGRESS:
        processed_seen, total, etapa = values[event]
        window["-PROG-"].update(current_count=processed_seen, max=max(total,1))
        window["-COUNT-"].update(f"{processed_seen}/{total}")
        if etapa:
            window["-STATUS-TXT-"].update(f"Rodando — {etapa}", text_color="green")

    elif event == EVT_STATUS:
        window["-STATUS-TXT-"].update(values[event], text_color="orange")

    elif event == EVT_LOG:
        window["-OUT-"].print(values[event])

    elif event == EVT_FINISHED:
        window["-OUT-"].print(values[event])
        flash_taskbar(window, count=8)
        window["-STATUS-TXT-"].update("Parado", text_color="grey20")
        window["-CANCEL-"].update(disabled=True)
        window["-FORCE-"].update(disabled=True)
        window["-REMPAIR-"].update(disabled=(len(pair_indices)==1))
        start_time = None

window.close()
