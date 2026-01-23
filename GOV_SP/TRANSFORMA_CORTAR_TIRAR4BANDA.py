# === PROCESSAMENTO â€” LÃ“GICA ORIGINAL + PATCHES + JP2 EM PASTA GLOBAL ===
# âž¤ TIF primeiro (reprojeta se preciso, recorta, drop alpha/copia) â†’ depois JP2
# âž¤ JP2 escrito DIRETO na pasta informada na UI (sem subpastas)
# âž¤ Only-JP2: ignora processamento e gera JP2 direto do original
# âž¤ UI: pares com fila dinÃ¢mica, â€œEnfileirar este/Todosâ€, status; caminho global de JP2 habilitado pelo checkbox
# âž¤ Performance: fecha datasets sempre, sindex + unary_union local, GDAL cacheâ†‘, OPJ threadsâ†“, logs em lote, gc periÃ³dico

import os, shutil, threading, time, math, gc
from datetime import datetime, timedelta
from queue import Queue, Empty

import numpy as np
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

# === NotificaÃ§Ã£o (Windows) ===
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
        f.write(f"\n=== InÃ­cio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
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
    if ds is None: raise RuntimeError("NÃ£o abriu dataset para SetProjection.")
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
    """Se >=4 bandas, pega 1-2-3; se 1-3 bandas, sÃ³ copia (GTiff tiled, COMPRESS=NONE)."""
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
    """Retorna True quando a área fora do shape já está vazia (nodata/alpha)."""
    t = src.transform
    h, w = src.height, src.width
    new_t, oh, ow = downscale_transform(t, w, h, max_dim=2000)
    inside = geometry_mask(geoms, transform=new_t, out_shape=(oh, ow), invert=True, all_touched=False)
    outside = ~inside
    if not outside.any():
        return False
    if src.count == 4:
        alpha = src.read(4, out_shape=(oh, ow), resampling=Resampling.nearest)
        outside_alpha = alpha[outside]
        if outside_alpha.size == 0:
            return False
        return (outside_alpha == 0).all()
    band1 = src.read(1, out_shape=(oh, ow), resampling=Resampling.nearest)
    outside_vals = band1[outside]
    if outside_vals.size == 0:
        return False
    if np.issubdtype(band1.dtype, np.floating) and np.isnan(outside_vals).all():
        return True
    nod = src.nodata if src.nodata is not None else nodata_pref
    if np.issubdtype(band1.dtype, np.floating):
        valid = outside_vals[~np.isnan(outside_vals)]
        if valid.size == 0:
            return True
        if nod is not None and np.isfinite(nod) and np.allclose(valid, nod, atol=1e-3):
            return True
        if np.nanstd(valid) <= 1e-3:
            return True
    else:
        if nod is not None and (outside_vals == nod).all():
            return True
        if np.unique(outside_vals).size == 1:
            return True
    return False

def compute_base_out(img_path, in_root, out_root):
    """Retorna a pasta .../IMAGENS_TRANSFORMADAS (base de saÃ­da do TIF)."""
    if out_root:
        rel_parent = os.path.relpath(os.path.dirname(img_path), in_root)
        parent = os.path.join(out_root, rel_parent)
    else:
        parent = os.path.dirname(img_path)
    base_out = os.path.join(parent, "IMAGENS_TRANSFORMADAS")
    return base_out

# === WORKER (consome a fila de pares) ===
def worker_process_queue(pairs_q: Queue,
                         window, stop_event: threading.Event, force_event: threading.Event,
                         shared_total):
    try:
        # ---- LOG em lote ----

        LOG_FLUSH_LINES = 12
        LOG_FLUSH_INTERVAL = 0.75
        log_buffer = []
        last_flush = [time.time()]

        def flush(force=False):
            if not log_buffer:
                return
            elapsed = time.time() - last_flush[0]
            if force or len(log_buffer) >= LOG_FLUSH_LINES or elapsed >= LOG_FLUSH_INTERVAL:
                window.write_event_value(EVT_LOG, "\n".join(log_buffer))
                log_buffer.clear()
                last_flush[0] = time.time()

        def log(msg, *, immediate=False):
            log_buffer.append(msg)
            # Se é uma mensagem de conclusão de arquivo, forçar flush imediato
            if any(keyword in msg.lower() for keyword in ['gerado', 'concluído', 'finalizado', 'tif:', 'jp2:', 'substituído']):
                flush(force=True)
            else:
                flush(force=immediate)

        # ---- Shapefile / sindex ----
        if False:
            window.write_event_value(EVT_STATUS, "Lendo shapefileâ€¦")
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
                in_root, out_root, pair_jp2_dir, pair_idx, opts = pairs_q.get(timeout=1.0)
            except Empty:
                idle_rounds += 1
                if idle_rounds >= 5:
                    break
                continue
            idle_rounds = 0

            # Processar todos os arquivos TIF do par atual
            for dp, _, fs in os.walk(in_root):
                if stop_event.is_set() or force_event.is_set(): 
                    break
                    
                for f in fs:
                    if stop_event.is_set() or force_event.is_set(): 
                        break
                    if not f.lower().endswith(".tif"): 
                        continue

                    img = os.path.join(dp, f)
                    nome = os.path.basename(img)
                    base_out = compute_base_out(img, in_root, out_root)
                    os.makedirs(base_out, exist_ok=True)

                    saida_final = os.path.join(base_out, nome)
                    base_nome   = os.path.splitext(nome)[0]
                    cortada_tmp = os.path.join(base_out, f"recorte_{base_nome}.tif")
                    reproj_tmp  = os.path.join(base_out, f"temp_{base_nome}.tif")
                    # Opções por item da fila
                    do_reproj = bool(opts.get("do_reproj", True))
                    do_crop = bool(opts.get("do_crop", True))
                    drop_alpha_allowed = bool(opts.get("drop4", True))
                    gerar_jp2_flag = bool(opts.get("do_jp2", True))
                    substituir_original = bool(opts.get("replace", False))
                    shp_path = opts.get("shp")
                    jp2_base_dir = (pair_jp2_dir or (opts.get("global_jp2_dir") or None))
                    dest_jp2_dir = jp2_base_dir
                    saida_jp2   = os.path.join(dest_jp2_dir, f"{base_nome}.jp2") if dest_jp2_dir else None
                    only_jp2 = (gerar_jp2_flag and not substituir_original and not (do_reproj or do_crop or drop_alpha_allowed))

                    # === ONLY-JP2 ===
                    if only_jp2:
                        try:
                            if saida_jp2 is None:
                                log(f"Sem pasta JP2 definida; pulando {nome}")
                            elif os.path.exists(saida_jp2):
                                log(f"JP2 já existe: {os.path.basename(saida_jp2)}")
                            else:
                                os.makedirs(dest_jp2_dir, exist_ok=True)
                                window.write_event_value(EVT_PROGRESS, (processados, get_total(), f"Gerando JP2: {nome}"))
                                gdal_translate_to_jp2(img, saida_jp2, force_event, window, get_total, processados)
                                log(f"JP2 gerado: {nome}")
                        except Exception as e:
                            log(f"Falha JP2 {nome}: {e}")
                        finally:
                            processados += 1
                            elapsed = max(1, time.time() - start)
                            total = get_total()
                            eta = int(elapsed * (total - processados) / max(1, processados))
                            window.write_event_value(EVT_PROGRESS, (processados, total, f"ETA ~ {str(timedelta(seconds=eta))}"))
                            
                            # Forçar atualização imediata do log após cada arquivo
                            flush(force=True)
                            
                            if processados % 20 == 0: 
                                gc.collect()
                        continue

                    # === PIPELINE COMPLETO ===
                    window.write_event_value(EVT_PROGRESS, (processados, get_total(), f"Avaliando: {nome}"))

                    # 1) EPSG / reprojeÃ§Ã£o
                    epsg = get_epsg_code(img)
                    need_reproj = False
                    src_for_clip = img

                    if epsg == "31983":
                        need_reproj = False
                    elif not epsg:
                        try: assign_epsg_inplace(img, 31983)
                        except Exception as e: log(f"SetProjection falhou ({nome}): {e}")
                        need_reproj = False
                    else:
                        need_reproj = True

                    if need_reproj:
                        try:
                            window.write_event_value(EVT_PROGRESS, (processados, get_total(), f"Reprojetando: {nome}"))
                            gdal_warp_to_31983(img, reproj_tmp, force_event, window, get_total, processados)
                            src_for_clip = reproj_tmp
                        except Exception as e:
                            log(f"Erro reprojeção {nome}: {e}")
                            if os.path.exists(reproj_tmp): robust_remove(reproj_tmp)
                            processados += 1
                            window.write_event_value(EVT_PROGRESS, (processados, get_total(), ""))
                            if processados % 20 == 0: gc.collect()
                            continue

                    # 2) Recorte
                    try:
                        with rasterio.open(src_for_clip) as src:
                            geom_raster = box(*src.bounds)
                            # preparar shapefile se for cortar
                            if do_crop:
                                try:
                                    gdf = gpd.read_file(shp_path).to_crs(epsg=31983)
                                    geom_unificada = gdf.geometry.union_all()
                                    try:
                                        sindex = gdf.sindex
                                    except Exception:
                                        sindex = None
                                except Exception as e:
                                    gdf = None; geom_unificada = None; sindex = None
                            if not do_crop:
                                # pular recorte nesta imagem
                                shapes = None
                                already_clip = True
                                need_crop = False
                                intersecao = None
                                # segue para avaliação de need_any
                            elif (geom_unificada is None or gdf is None):
                                # se recorte solicitado mas shape inválido
                                window[f"-PAIRSTAT-{pair_idx}-"].update("Erro shapefile", text_color="red")
                                log(f"Erro shapefile: {nome}")
                                shapes = None
                                already_clip = True
                                need_crop = False
                                intersecao = None
                            if do_crop and not geom_raster.intersects(geom_unificada):
                                window[f"-PAIRSTAT-{pair_idx}-"].update("Sem intersecção", text_color="orange")
                                log(f"Sem intersecção: {nome}")
                            else:
                                if do_crop and sindex is not None:
                                    cand_idx = list(sindex.intersection(geom_raster.bounds))
                                    intersecao = gdf.iloc[cand_idx]
                                    intersecao = intersecao[intersecao.geometry.intersects(geom_raster)]
                                elif do_crop:
                                    intersecao = gdf[gdf.geometry.intersects(geom_raster)]
                                else:
                                    intersecao = None

                                if do_crop and (intersecao is None or intersecao.empty):
                                    window[f"-PAIRSTAT-{pair_idx}-"].update("Intersecção vazia", text_color="orange")
                                    log(f"Intersecção vazia: {nome}")
                                else:
                                    need_alpha_drop = (drop_alpha_allowed and (src.count >= 4))
                                    if do_crop:
                                        geom_clip = intersecao.unary_union
                                        shapes = [mapping(geom_clip)]
                                        already_clip = is_already_clipped(src, shapes, nodata_pref=255)
                                        need_crop = not already_clip
                                    else:
                                        shapes = None
                                        already_clip = True
                                        need_crop = False

                                    need_any = need_reproj or need_crop or need_alpha_drop or gerar_jp2_flag or substituir_original
                                    if not need_any:
                                        window[f"-PAIRSTAT-{pair_idx}-"].update("OK (pulado)", text_color="grey20")
                                        log(f"Pulado (já OK): {nome}")
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
                                                    if drop_alpha_allowed and (_src_tmp.count >= 4):
                                                        gdal_translate_drop_alpha_or_copy(write_target, saida_final, force_event, window, get_total, processados)
                                                        tif_final_created = True
                                            if substituir_original and (tif_final_created or write_target != img):
                                                src_for_replace = saida_final if tif_final_created else write_target
                                                time.sleep(0.03)
                                                robust_remove(img)
                                                shutil.move(src_for_replace, img)
                                                log(f"   Original substituído: {nome}")
                                                jp2_input = img
                                            else:
                                                if tif_final_created:
                                                    log(f"TIF: {saida_final}")
                                                jp2_input = saida_final if tif_final_created else write_target

                                        except Exception as e:
                                            log(f"Erro TIF {nome}: {e}")
                                            jp2_input = write_target

                                        # 4) JP2 depois (na pasta global)
                                        try:
                                            if gerar_jp2_flag and jp2_base_dir:
                                                os.makedirs(jp2_base_dir, exist_ok=True)
                                                if not os.path.exists(saida_jp2):
                                                    window.write_event_value(EVT_PROGRESS, (processados, get_total(), f"Gerando JP2: {nome}"))
                                                    gdal_translate_to_jp2(jp2_input, saida_jp2, force_event, window, get_total, processados)
                                                    log(f"   JP2: {os.path.basename(saida_jp2)}")
                                                else:
                                                    log(f"   JP2 já existe: {os.path.basename(saida_jp2)}")
                                        except Exception as e:
                                            log(f"Falha JP2 {nome}: {e}")

                    except Exception as e:
                        log(f"Erro geral: {nome}” {e}")

                    # limpar reproj tmp
                    if src_for_clip != img and os.path.exists(src_for_clip):
                        robust_remove(src_for_clip)

                    processados += 1
                    elapsed = max(1, time.time() - start)
                    total = get_total()
                    eta = int(elapsed * (total - processados) / max(1, processados))
                    window.write_event_value(EVT_PROGRESS, (processados, total, f"ETA ~ {str(timedelta(seconds=eta))}"))
                    
                    # Forçar atualização imediata do log após cada arquivo
                    flush(force=True)
                    
                    if processados % 20 == 0:
                        gc.collect()

        # flush final dos logs
        try:
            flush(force=True)
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
    frame_layout = [
        [sg.Text("Entrada", size=(14, 1)),
         sg.Input(key=f"-IN-{idx}-", expand_x=True),
         sg.FolderBrowse(key=f"-BIN-{idx}-", target=f"-IN-{idx}-")],
        [sg.Text("Saída (base TIF)", size=(14, 1)),
         sg.Input(key=f"-OUT-{idx}-", expand_x=True),
         sg.FolderBrowse(key=f"-BOUT-{idx}-", target=f"-OUT-{idx}-")],
        [sg.Text("Saída JP2 (opcional)", size=(14, 1)),
         sg.Input(key=f"-JP2PAIR-{idx}-", expand_x=True),
         sg.FolderBrowse(key=f"-BJP2-{idx}-", target=f"-JP2PAIR-{idx}-")],
        [sg.Text("Status:", size=(14, 1)),
         sg.Text("aguardando…", key=f"-PAIRSTAT-{idx}-", size=(50, 1), text_color="grey20"),
         sg.Push(),
         sg.Button("Enfileirar este", key=f"-ENQ-{idx}-", pad=((8, 0), (0, 0)))]
    ]
    return [sg.pin(sg.Frame(
        f"Par {idx+1}",
        frame_layout,
        key=f"-PAIR-{idx}-",
        relief=sg.RELIEF_SUNKEN,
        pad=(0, 6),
        expand_x=True
    ))]

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
    [
        sg.Frame(
            "Operações",
            [[
                sg.Checkbox("Reprojetar para EPSG:31983", default=True, key="-OP-REPROJ-"),
                sg.Checkbox("Cortar pelo shapefile", default=True, key="-OP-CROP-"),
                sg.Checkbox("Remover 4ª banda (RGB)", default=True, key="-OP-DROP4-")
            ]],
            relief=sg.RELIEF_GROOVE,
            expand_x=True
        )
    ],
    [sg.Checkbox("Gerar JP2 (≈1:10)", default=True, key="-JP2-")],
    [sg.Text("Pasta padrão do JP2 (opcional):"), sg.Input(key="-JP2DIR-", size=(80,1), disabled=False),
     sg.FolderBrowse(key="-BROWSE-JP2-", disabled=False)],
    [sg.Text("Par sem caminho JP2 usa este valor; deixe vazio para exigir caminho por par.", text_color="grey35", font=("Segoe UI", 9))],
    [sg.Checkbox("Substituir original após concluir (sem backup)", default=False, key="-REPLACE-")],
    status_row,
    [sg.ProgressBar(100, orientation='h', size=(80, 20), key='-PROG-')],
    [sg.Button("Iniciar", key="-START-", size=(12,1)),
     sg.Button("Cancelar", key="-CANCEL-", size=(12,1), disabled=True),
     sg.Button("Forçar parada", key="-FORCE-", size=(14,1), disabled=True),
     sg.Push(), sg.Button("Sair")],
    [sg.Output(size=(140, 22), key='-OUT-')]
]
window = sg.Window("Processamento de Imagens” GDAL (TIF→JP2, otimizado)", layout,
                   finalize=True, resizable=True)

# estado dos pares
pair_indices = [0]
next_pair_id = 1
hidden_stack = []

# execuÃ§Ã£o
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

def enqueue_pairs(idx_list, values, default_jp2_dir):
    added = 0
    for idx in idx_list:
        in_dir = (values.get(f"-IN-{idx}-") or "").strip()
        out_dir = (values.get(f"-OUT-{idx}-") or "").strip()
        jp2_dir_override = (values.get(f"-JP2PAIR-{idx}-") or "").strip()
        # opções por item (capturadas no momento do enfileiramento)
        do_reproj = bool(values.get("-OP-REPROJ-", True))
        do_crop = bool(values.get("-OP-CROP-", True))
        drop4 = bool(values.get("-OP-DROP4-", True))
        do_jp2 = bool(values.get("-JP2-", True))
        replace_original = bool(values.get("-REPLACE-", False))
        shp_path = (values.get("-SHP-") or "").strip()

        if not in_dir or not os.path.isdir(in_dir):
            window[f"-PAIRSTAT-{idx}-"].update("Entrada inválida", text_color="red")
            continue

        if out_dir and not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                window[f"-PAIRSTAT-{idx}-"].update(f"Saída inválida: {e}", text_color="red")
                continue

        if do_jp2 and jp2_dir_override:
            try:
                os.makedirs(jp2_dir_override, exist_ok=True)
            except Exception as e:
                window[f"-PAIRSTAT-{idx}-"].update(f"Saída JP2 inválida: {e}", text_color="red")
                continue

        qtd = count_tifs(in_dir)
        if qtd == 0:
            window[f"-PAIRSTAT-{idx}-"].update("Sem .tif", text_color="orange")
            continue

        shared_total["value"] += qtd
        window["-PROG-"].update(current_count=processed_seen, max=max(shared_total["value"], 1))
        window["-COUNT-"].update(f"{processed_seen}/{shared_total['value']}")

        if do_jp2:
            if jp2_dir_override:
                jp2_hint = jp2_dir_override
            elif default_jp2_dir:
                jp2_hint = default_jp2_dir.strip()
            else:
                jp2_hint = None
            if jp2_hint:
                status_msg = f"na fila — JP2: {jp2_hint}"
                status_color = "green"
            else:
                status_msg = "na fila — JP2 indefinido"
                status_color = "orange"
        else:
            status_msg = "na fila"
            status_color = "green"

        window["-OUT-"].print(f"⚙️ Par {idx+1} enfileirado ({qtd} imagens).")
        window[f"-PAIRSTAT-{idx}-"].update(status_msg, text_color=status_color)
        # Validar shapefile se o item requer corte
        if do_crop:
            if not shp_path or not os.path.isfile(shp_path):
                window[f"-PAIRSTAT-{idx}-"].update("Shapefile inválido para corte", text_color="red")
                continue
        item_opts = {
            "do_reproj": do_reproj,
            "do_crop": do_crop,
            "drop4": drop4,
            "do_jp2": do_jp2,
            "replace": replace_original,
            "global_jp2_dir": (default_jp2_dir or "").strip() or None,
            "shp": shp_path or None
        }
        pairs_q.put((in_dir, out_dir if out_dir else None, jp2_dir_override if jp2_dir_override else None, idx, item_opts))
        added += 1
    return added

def start_worker():
    global worker_th, start_time
    start_time = time.time()
    worker_th = threading.Thread(
        target=worker_process_queue,
        args=(pairs_q, window, stop_event, force_event, shared_total),
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
            if not worker_th or not worker_th.is_alive():
                sg.popup("Clique em Iniciar antes de enfileirar."); continue
            default_jp2_dir = (values.get("-JP2DIR-") or "").strip()
            _ = enqueue_pairs([idx], values, default_jp2_dir)

    if event == "-ENQ-ALL-":
        if not worker_th or not worker_th.is_alive():
            sg.popup("Clique em Iniciar antes de enfileirar."); continue
        default_jp2_dir = (values.get("-JP2DIR-") or "").strip()
        added = enqueue_pairs(pair_indices, values, default_jp2_dir)
        if added == 0:
            window["-OUT-"].print("Nada novo para enfileirar.")

    if event == "-START-":
        gerar_jp2_flag      = values["-JP2-"]
        substituir_original = values["-REPLACE-"]
        only_jp2 = (gerar_jp2_flag and not substituir_original)  # mesma regra de "only jp2"

        jp2_base_dir = (values.get("-JP2DIR-") or "").strip()
        if gerar_jp2_flag:
            if jp2_base_dir:
                try:
                    os.makedirs(jp2_base_dir, exist_ok=True)
                except Exception as e:
                    sg.popup_error(f"Não foi possível criar a pasta de JP2:\n{e}")
                    continue
            else:
                overrides = any((values.get(f"-JP2PAIR-{idx}-") or "").strip() for idx in pair_indices)
                if not overrides:
                    sg.popup_error("Informe a pasta de saída do JP2 (global ou por par).")
                    continue

        shp = values["-SHP-"]
        if values.get("-OP-CROP-", True):
            if not shp or not os.path.isfile(shp):
                sg.popup_error("Selecione um shapefile vÃ¡lido."); continue

        stop_event.clear(); force_event.clear()
        shared_total["value"] = 0; processed_seen = 0
        while not pairs_q.empty():
            try: pairs_q.get_nowait()
            except Exception: break

        _ = enqueue_pairs(pair_indices, values, jp2_base_dir)
        if shared_total["value"] == 0:
            sg.popup_error("Nenhuma imagem encontrada nos pares informados."); continue

        window["-OUT-"].print("Iniciando…")
        window["-STATUS-TXT-"].update("Rodando", text_color="green")
        window["-PROG-"].update(current_count=0, max=max(shared_total["value"],1))
        window["-COUNT-"].update(f"0/{shared_total['value']}")
        window["-CANCEL-"].update(disabled=False)
        window["-FORCE-"].update(disabled=False)
        window["-REMPAIR-"].update(disabled=True)

        start_worker()

    if event == "-CANCEL-":
        if worker_th and worker_th.is_alive():
            stop_event.set()
            window["-STATUS-TXT-"].update("Cancelando…", text_color="orange")
            window["-OUT-"].print("Cancelar solicitado (encerra após a etapa atual)…")

    if event == "-FORCE-":
        if worker_th and worker_th.is_alive():
            force_event.set()
            window["-STATUS-TXT-"].update("Forçando parada…", text_color="orange")
            window["-OUT-"].print("Força STOP solicitado (aborta operação atual)…")

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
