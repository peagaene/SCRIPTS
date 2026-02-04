# === PROCESSAMENTO - TIF/TXT COM CORTE POR MI_3 ===
# -> TIF: reprojeta (se precisar), recorta e organiza a saída base
# -> TXT: recorta pontos usando o mesmo polígono identificado por MI_3
# -> UI mantém pares na fila com status e opção de substituir o original
# -> Performance: fecha datasets sempre, usa sindex/unary_union, GDAL cache, logs em lote e GC periódico

import os, re, shutil, threading, time, math, gc, traceback
from datetime import datetime, timedelta
from queue import Queue, Empty
from typing import Optional

import numpy as np
import PySimpleGUI as sg
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.enums import Resampling
from shapely.geometry import box, mapping, Point
from shapely.ops import unary_union
from osgeo import gdal, osr

# === CONFIG GLOBAL ===
gdal.UseExceptions()
gdal.SetConfigOption("CHECK_DISK_FREE_SPACE", "FALSE")
gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
gdal.SetCacheMax(1024 * 1024 * 1024)        # ~1 GiB
os.environ.setdefault("GDAL_CACHEMAX", "1024")
os.environ.setdefault("OPJ_NUM_THREADS", str(max(1, math.floor((os.cpu_count() or 4) / 2))))

TARGET_EPSG = "EPSG:31982"
LOG_PATH = os.path.join(os.path.expanduser("~"), "log_processamento_completo.txt")
DEFAULT_SHP = r"\\192.168.2.28\i\80225_PROJETO_IAT_PARANA\5 Processamento Laser\Articulacao_Auxiliar_5000+50.shp"
SUPPORTED_EXTENSIONS = (".tif", ".txt")

# === EVENTOS DA UI ===
EVT_STATUS   = "-STATUS-"
EVT_PROGRESS = "-PROGRESS-"
EVT_FINISHED = "-FINISHED-"
EVT_LOG      = "-LOG-"
EVT_JOBSTAT  = "-JOBSTAT-"

def _report_error(context: str, exc: Exception):
    """Mostra no console e retorna uma string com contexto de erro."""
    tb = traceback.format_exc()
    msg = f"[ERRO] {context}: {exc}"
    print(msg)
    if tb:
        print(tb)
    return msg

def _sanitize_color(value: str, default="grey20"):
    """Garante uma cor válida para o Tkinter (nome básico ou #RGB/#RRGGBB)."""
    if not value:
        return default
    val = str(value).strip()
    if val.startswith('#'):
        hex_part = val[1:]
        if len(hex_part) in (3, 6) and all(ch in '0123456789abcdefABCDEF' for ch in hex_part):
            return f'#{hex_part}'
        return default
    known = {'grey20','gray20','black','white','red','orange','green','blue','yellow'}
    return val if val.lower() in known else default

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

def count_supported_files(root_dir):
    total = 0
    try:
        for dp, _, fs in os.walk(root_dir):
            for f in fs:
                name = f.lower()
                if name.endswith(".tif") or name.endswith(".txt"):
                    total += 1
    except OSError:
        return 0
    return total

MI3_PATTERN = re.compile(r"(\d{3,}_[A-Z0-9]+_[A-Z0-9]+_[A-Z0-9]+_[A-Z0-9]+)", re.IGNORECASE)

def extract_mi3_from_name(path: str) -> Optional[str]:
    """Extrai o trecho 2863_3_SO_F_II (ou equivalente) do nome do arquivo."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem_norm = re.sub(r"[-]+", "_", stem)  # trata nomes com hífen como separador
    match = MI3_PATTERN.search(stem_norm)
    if match:
        return match.group(1).upper()
    parts = stem_norm.split("_")
    tokens_upper = [p.upper() for p in parts]
    start = None
    for marker in ("MDS", "MDT", "MDE"):
        if marker in tokens_upper:
            start = tokens_upper.index(marker) + 1
            break
    if start is None:
        start = max(0, len(parts) - 5)
    segment = parts[start:start+5]
    if len(segment) < 5:
        return None
    return "_".join(segment).upper()

def normalize_mi3(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.strip().upper().replace("_", "-")

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

def assign_epsg_inplace(img, epsg=31982):
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

def gdal_warp_to_31982(src_path, dst_path, force_event, window, get_total, processados):
    cb = make_gdal_progress_cb(force_event, window, get_total, processados, "Reprojetando")
    opts = gdal.WarpOptions(dstSRS=TARGET_EPSG, multithread=True, dstAlpha=True,
                            creationOptions=["TILED=YES","BIGTIFF=IF_SAFER"], callback=cb)
    ds = gdal.Warp(dst_path, src_path, options=opts)
    if ds is None: raise RuntimeError("gdal.Warp falhou.")
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

def compute_base_out(img_path):
    """Retorna a pasta .../IMAGENS_TRANSFORMADAS ao lado da origem."""
    parent = os.path.dirname(img_path)
    return os.path.join(parent, "IMAGENS_TRANSFORMADAS")

def prepare_shp_bundle(shp_path: str):
    gdf = gpd.read_file(shp_path)
    if gdf.crs:
        gdf = gdf.to_crs(epsg=31982)
    else:
        gdf = gdf.set_crs(epsg=31982)
    try:
        sindex = gdf.sindex
    except Exception:
        sindex = None
    attr_map = {}
    if "MI_3" in gdf.columns:
        for _, row in gdf.iterrows():
            key = normalize_mi3(str(row["MI_3"]))
            geom = row.geometry
            if not key or geom is None or geom.is_empty:
                continue
            attr_map.setdefault(key, []).append(geom)
    geom_unificada = gdf.geometry.union_all() if len(gdf) else None
    return {
        "gdf": gdf,
        "sindex": sindex,
        "geom_unificada": geom_unificada,
        "attr_map": attr_map
    }

def clip_txt_points(src_txt: str, dst_txt: str, clip_geom, encoding="utf-8"):
    total, kept = 0, 0
    with open(src_txt, "r", encoding=encoding, errors="ignore") as src, \
         open(dst_txt, "w", encoding="utf-8") as dst:
        for line in src:
            stripped = line.strip()
            if not stripped:
                dst.write(line)
                continue
            tokens = re.split(r"[;\s,]+", stripped)
            coords = []
            for token in tokens:
                try:
                    coords.append(float(token.replace(",", ".")))
                except ValueError:
                    continue
                if len(coords) == 2:
                    break
            if len(coords) < 2:
                dst.write(line)
                continue
            total += 1
            pt = Point(coords[0], coords[1])
            if clip_geom.covers(pt):
                kept += 1
                dst.write(line)
    return total, kept

def get_clip_geometry(shp_bundle, filename: str, geom_raster=None):
    if not shp_bundle:
        return None, None
    raw_code = extract_mi3_from_name(filename)
    mi3_norm = normalize_mi3(raw_code)
    label = mi3_norm or raw_code
    if not mi3_norm:
        return None, label
    geoms = shp_bundle["attr_map"].get(mi3_norm)
    if not geoms:
        return None, label
    geom_clip = unary_union(geoms)
    if geom_clip is None or geom_clip.is_empty:
        return None, label
    if geom_raster is not None and not geom_raster.intersects(geom_clip):
        return None, label
    return geom_clip, mi3_norm

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
            if any(keyword in msg.lower() for keyword in ['gerado', 'concluído', 'finalizado', 'tif:', 'txt:', 'substituído']):
                flush(force=True)
            else:
                flush(force=immediate)
        def job_status(job_id, text, color="grey20"):
            try:
                window.write_event_value(EVT_JOBSTAT, (job_id, text, color))
            except Exception:
                pass
        # ---- cache simples para shapefiles reutilizados ----
        shp_cache = {}


        processados = 0
        start = time.time()
        def get_total(): return shared_total["value"]

        idle_rounds = 0
        while not stop_event.is_set():
            try:
                job_id, in_root, opts = pairs_q.get(timeout=1.0)
            except Empty:
                idle_rounds += 1
                if idle_rounds >= 5:
                    break
                continue
            idle_rounds = 0
            job_status(job_id, "Processando", "orange")

            do_reproj = bool(opts.get("do_reproj", True))
            do_crop = bool(opts.get("do_crop", True))
            substituir_original = bool(opts.get("replace", False))
            shp_path = opts.get("shp")
            shp_bundle = None
            if do_crop and shp_path:
                shp_bundle = shp_cache.get(shp_path)
                if shp_bundle is None:
                    try:
                        window.write_event_value(EVT_STATUS, f"Lendo shapefile ({os.path.basename(shp_path)})")
                    except Exception:
                        pass
                    try:
                        shp_bundle = prepare_shp_bundle(shp_path)
                        shp_cache[shp_path] = shp_bundle
                    except Exception as e:
                        job_status(job_id, "Erro shapefile", "red")
                        log(f"Erro ao carregar shapefile {os.path.basename(shp_path)}: {e}")
                        shp_bundle = None
                        do_crop = False
            elif do_crop:
                job_status(job_id, "Shapefile inválido", "red")
                do_crop = False

            # Processar todos os arquivos TIF do par atual
            for dp, _, fs in os.walk(in_root):
                if stop_event.is_set() or force_event.is_set(): 
                    break
                    
                for f in fs:
                    if stop_event.is_set() or force_event.is_set(): 
                        break

                    if f.lower().endswith(".txt"):
                        txt_path = os.path.join(dp, f)
                        nome_txt = os.path.basename(txt_path)
                        window.write_event_value(EVT_PROGRESS, (processados, get_total(), f"TXT: {nome_txt}"))
                        if not do_crop or not shp_bundle:
                            log(f"TXT sem shapefile configurado: {nome_txt}")
                        else:
                            clip_geom, clip_label = get_clip_geometry(shp_bundle, nome_txt, None)
                            if clip_geom is None:
                                if clip_label:
                                    log(f"TXT MI_3 não encontrado ({clip_label}): {nome_txt}")
                                else:
                                    log(f"TXT sem MI_3 no nome: {nome_txt}")
                            else:
                                base_out_txt = compute_base_out(txt_path)
                                os.makedirs(base_out_txt, exist_ok=True)
                                tmp_txt = os.path.join(base_out_txt, f"_tmp_{nome_txt}")
                                try:
                                    total_pts, kept_pts = clip_txt_points(txt_path, tmp_txt, clip_geom)
                                    destino_txt = txt_path if substituir_original else os.path.join(base_out_txt, nome_txt)
                                    if substituir_original:
                                        robust_remove(txt_path)
                                        shutil.move(tmp_txt, txt_path)
                                        log(f"TXT substituído: {nome_txt} ({kept_pts}/{total_pts} pontos)")
                                    else:
                                        if os.path.exists(destino_txt):
                                            robust_remove(destino_txt)
                                        shutil.move(tmp_txt, destino_txt)
                                        log(f"TXT: {destino_txt} ({kept_pts}/{total_pts} pontos)")
                                except Exception as e:
                                    log(f"Erro TXT {nome_txt}: {e}")
                                    if os.path.exists(tmp_txt):
                                        robust_remove(tmp_txt)
                        processados += 1
                        elapsed = max(1, time.time() - start)
                        total = get_total()
                        eta = int(elapsed * (total - processados) / max(1, processados))
                        window.write_event_value(EVT_PROGRESS, (processados, total, f"ETA ~ {str(timedelta(seconds=eta))}"))
                        flush(force=True)
                        if processados % 20 == 0:
                            gc.collect()
                        continue

                    if not f.lower().endswith(".tif"): 
                        continue

                    img = os.path.join(dp, f)
                    nome = os.path.basename(img)
                    base_out = compute_base_out(img)
                    os.makedirs(base_out, exist_ok=True)

                    saida_final = os.path.join(base_out, nome)
                    base_nome   = os.path.splitext(nome)[0]
                    cortada_tmp = os.path.join(base_out, f"recorte_{base_nome}.tif")
                    reproj_tmp  = os.path.join(base_out, f"temp_{base_nome}.tif")
                    # === PIPELINE COMPLETO ===
                    window.write_event_value(EVT_PROGRESS, (processados, get_total(), f"Avaliando: {nome}"))

                    # 1) EPSG / reprojeÃ§Ã£o
                    epsg = get_epsg_code(img)
                    need_reproj = False
                    src_for_clip = img

                    if epsg == "31982":
                        need_reproj = False
                    elif not epsg:
                        try: assign_epsg_inplace(img, 31982)
                        except Exception as e: log(f"SetProjection falhou ({nome}): {e}")
                        need_reproj = False
                    else:
                        need_reproj = True

                    if need_reproj:
                        try:
                            window.write_event_value(EVT_PROGRESS, (processados, get_total(), f"Reprojetando: {nome}"))
                            gdal_warp_to_31982(img, reproj_tmp, force_event, window, get_total, processados)
                            src_for_clip = reproj_tmp
                        except Exception as e:
                            log(f"Erro reprojeção {nome}: {e}")
                            if os.path.exists(reproj_tmp): robust_remove(reproj_tmp)
                            processados += 1
                            window.write_event_value(EVT_PROGRESS, (processados, get_total(), ""))
                            if processados % 20 == 0: gc.collect()
                            continue

                    write_target = src_for_clip
                    try:
                        with rasterio.open(src_for_clip) as src:
                            geom_raster = box(*src.bounds)
                            shapes = None
                            need_crop = False
                            if do_crop and shp_bundle:
                                clip_geom, clip_label = get_clip_geometry(shp_bundle, nome, geom_raster)
                                if clip_geom is None:
                                    if clip_label:
                                        log(f"MI_3 não encontrado ({clip_label}): {nome}")
                                    else:
                                        log(f"MI_3 não identificado no nome: {nome}")
                                else:
                                    shapes = [mapping(clip_geom)]
                                    already_clip = is_already_clipped(src, shapes, nodata_pref=255)
                                    need_crop = not already_clip
                            else:
                                shapes = None
                                need_crop = False

                            need_any = need_reproj or need_crop or substituir_original
                            if not need_any:
                                job_status(job_id, "OK (pulado)", "grey20")
                                log(f"Pulado (já OK): {nome}")
                            else:
                                if need_crop and shapes:
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
                                    with rasterio.open(cortada_tmp, "w", **meta) as dest:
                                        dest.write(img_cortada)
                                    del img_cortada; time.sleep(0.03)
                                    write_target = cortada_tmp

                                try:
                                    if substituir_original:
                                        if write_target != img:
                                            robust_remove(img)
                                            shutil.move(write_target, img)
                                        log(f"   Original substituído: {nome}")
                                    else:
                                        if write_target in (cortada_tmp, reproj_tmp):
                                            shutil.move(write_target, saida_final)
                                        elif write_target == img:
                                            shutil.copy2(img, saida_final)
                                        else:
                                            shutil.copy2(write_target, saida_final)
                                        log(f"TIF: {saida_final}")
                                except Exception as e:
                                    log(f"Erro TIF {nome}: {e}")
                    except Exception as e:
                        log(f"Erro geral: {nome} {e}")

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

            if stop_event.is_set() or force_event.is_set():
                job_status(job_id, "Interrompido", "orange")
                break
            else:
                job_status(job_id, "Concluído", "grey20")

        # flush final dos logs
        try:
            flush(force=True)
        except Exception:
            pass

        window.write_event_value(EVT_FINISHED, f"Concluído. Processados {processados}/{shared_total['value']}. Log: {LOG_PATH}")

    except Exception as e:
        _report_error("Falha inesperada no worker", e)
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

queue_table = sg.Table(
    values=[],
    headings=["#", "Pasta", "Status"],
    auto_size_columns=False,
    col_widths=[4, 60, 20],
    justification="left",
    num_rows=7,
    key="-QUEUE-TBL-",
    expand_x=True,
    expand_y=True,
    enable_events=True,
    select_mode=sg.TABLE_SELECT_MODE_EXTENDED,
    alternating_row_color="#f4f4f4"
)

input_row = [
    sg.Text("Pasta com TIF/TXT:", size=(18, 1)),
    sg.Input(key="-IN-PATH-", expand_x=True),
    sg.FolderBrowse(key="-BROWSE-IN-", target="-IN-PATH-"),
    sg.Button("Adicionar a fila", key="-ADD-JOB-", size=(16, 1))
]

queue_controls = [
    sg.Button("Remover selecionados", key="-REMOVE-JOB-", size=(20, 1)),
    sg.Button("Limpar fila", key="-CLEAR-JOB-", size=(12, 1)),
    sg.Push(),
    sg.Text("Saida: os originais serao sobrescritos", text_color="grey30")
]

layout = [
    [input_row],
    [sg.Frame("Fila de processamento", [
        [queue_table],
        queue_controls
    ], expand_x=True, expand_y=True)],
    [sg.Text("Shapefile de corte (.shp):")],
    [sg.Input(default_text=DEFAULT_SHP, key="-SHP-", size=(80,1)),
     sg.FileBrowse(file_types=(("Shapefile", "*.shp"),), key="-BROWSHP-", initial_folder=os.path.dirname(DEFAULT_SHP))],
    [
        sg.Frame(
            "Operacoes",
            [[
                sg.Checkbox("Reprojetar para EPSG:31982", default=True, key="-OP-REPROJ-"),
                sg.Checkbox("Cortar pelo shapefile", default=True, key="-OP-CROP-")
            ]],
            relief=sg.RELIEF_GROOVE,
            expand_x=True
        )
    ],
    status_row,
    [sg.ProgressBar(100, orientation='h', size=(80, 20), key='-PROG-')],
    [sg.Button("Iniciar", key="-START-", size=(12,1)),
     sg.Button("Cancelar", key="-CANCEL-", size=(12,1), disabled=True),
     sg.Button("Forcar parada", key="-FORCE-", size=(14,1), disabled=True),
     sg.Push(), sg.Button("Sair")],
    [sg.Output(size=(140, 22), key='-OUT-')]
]
window = sg.Window("Processamento de Imagens GDAL (TIF/TXT)", layout,
                   finalize=True, resizable=True)

# estado dos jobs
jobs = {}
job_order = []
next_job_id = 1

# execucao
pairs_q     = Queue()
stop_event  = threading.Event()
force_event = threading.Event()
worker_th   = None
start_time  = None

# progresso
shared_total = {"value": 0}
processed_seen = 0

def refresh_queue_table():
    bg_default = _sanitize_color(sg.theme_background_color(), "#ffffff")
    data = [[i + 1, jobs[jid]["path"], jobs[jid]["status"]] for i, jid in enumerate(job_order)]
    row_colors = []
    for i, jid in enumerate(job_order):
        color = _sanitize_color(jobs[jid].get("color"))
        row_colors.append((i, color, bg_default))
    try:
        window["-QUEUE-TBL-"].update(values=data, row_colors=row_colors)
    except Exception as e:
        _report_error("Falha ao colorir tabela", e)
        window["-QUEUE-TBL-"].update(values=data, row_colors=[])

def set_job_status(job_id, text, color="grey20"):
    job = jobs.get(job_id)
    if not job:
        return
    job["status"] = text
    job["color"] = _sanitize_color(color)
    refresh_queue_table()

def add_job_entry(path_str):
    global next_job_id
    try:
        path = (path_str or "").strip()
        if not path or not os.path.isdir(path):
            sg.popup_error("Selecione uma pasta valida.")
            return None
        qtd = count_supported_files(path)
        if qtd == 0:
            sg.popup_error("Nenhum TIF/TXT foi encontrado nessa pasta.")
            return None
        job_id = next_job_id
        next_job_id += 1
        jobs[job_id] = {
            "path": path,
            "status": "Aguardando",
            "color": "grey20",
            "count": qtd,
            "enqueued": False
        }
        job_order.append(job_id)
        refresh_queue_table()
        window["-OUT-"].print(f"Pasta adicionada ({qtd} arquivos): {path}")
        window["-IN-PATH-"].update("")
        return job_id
    except Exception as e:
        msg = _report_error("Falha ao adicionar pasta", e)
        sg.popup_error(msg)
        return None

def remove_jobs(row_indices):
    if not row_indices:
        return
    for idx in sorted(set(row_indices), reverse=True):
        if 0 <= idx < len(job_order):
            job_id = job_order.pop(idx)
            jobs.pop(job_id, None)
    refresh_queue_table()

def clear_all_jobs():
    jobs.clear()
    job_order.clear()
    refresh_queue_table()

def push_job_to_worker(job_id, values):
    job = jobs.get(job_id)
    if not job:
        return False
    try:
        in_dir = job["path"]
        if not os.path.isdir(in_dir):
            set_job_status(job_id, "Pasta nao encontrada", "red")
            return False
        do_reproj = bool(values.get("-OP-REPROJ-", True))
        do_crop = bool(values.get("-OP-CROP-", True))
        shp_path = (values.get("-SHP-") or "").strip()
        if do_crop and (not shp_path or not os.path.isfile(shp_path)):
            set_job_status(job_id, "Shapefile invalido", "red")
            return False
        qtd = count_supported_files(in_dir)
        if qtd == 0:
            set_job_status(job_id, "Sem arquivos", "orange")
            return False
        job["count"] = qtd
        job["enqueued"] = True
        set_job_status(job_id, "Na fila", "green")
        opts = {
            "do_reproj": do_reproj,
            "do_crop": do_crop,
            "replace": True,
            "shp": shp_path or None
        }
        pairs_q.put((job_id, in_dir, opts))
        shared_total["value"] += qtd
        window["-PROG-"].update(current_count=processed_seen, max=max(shared_total["value"], 1))
        window["-COUNT-"].update(f"{processed_seen}/{shared_total['value']}")
        window["-OUT-"].print(f">> Pasta enfileirada ({qtd} arquivos): {in_dir}")
        return True
    except Exception as e:
        msg = _report_error("Falha ao enfileirar pasta", e)
        set_job_status(job_id, f"Erro: {e}", "red")
        sg.popup_error(msg)
        return False

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
    try:
        event, values = window.read(timeout=200)
    except Exception as e:
        try:
            msg = _report_error("Erro na interface", e)
            sg.popup_error(msg)
        finally:
            break
    if event == sg.WIN_CLOSED or event == "Sair":
        if worker_th and worker_th.is_alive():
            stop_event.set(); force_event.set()
            worker_th.join(timeout=2)
        break


    if start_time and worker_th and worker_th.is_alive():
        elapsed = timedelta(seconds=int(time.time() - start_time))
        window["-ELAPSED-"].update(str(elapsed))

    if event == "-ADD-JOB-":
        job_id = add_job_entry(values.get("-IN-PATH-"))
        if job_id and worker_th and worker_th.is_alive():
            push_job_to_worker(job_id, values)

    elif event == "-REMOVE-JOB-":
        if worker_th and worker_th.is_alive():
            sg.popup("Remoção desabilitada durante o processamento.")
        else:
            remove_jobs(values.get("-QUEUE-TBL-", []))

    elif event == "-CLEAR-JOB-":
        if worker_th and worker_th.is_alive():
            sg.popup("Remoção desabilitada durante o processamento.")
        else:
            clear_all_jobs()

    if event == "-START-":
        if worker_th and worker_th.is_alive():
            sg.popup("Processamento já está em andamento.")
            continue
        if not job_order:
            sg.popup_error("Adicione ao menos uma pasta antes de iniciar.")
            continue
        if values.get("-OP-CROP-", True):
            shp = (values.get("-SHP-") or "").strip()
            if not shp or not os.path.isfile(shp):
                sg.popup_error("Selecione um shapefile válido.")
                continue

        stop_event.clear(); force_event.clear()
        shared_total["value"] = 0; processed_seen = 0
        while not pairs_q.empty():
            try:
                pairs_q.get_nowait()
            except Exception:
                break

        enqueued = 0
        for job_id in job_order:
            jobs[job_id]["enqueued"] = False
            if push_job_to_worker(job_id, values):
                enqueued += 1

        if enqueued == 0:
            sg.popup_error("Nenhuma pasta válida para processar.")
            continue

        window["-OUT-"].print("Iniciando...")
        window["-STATUS-TXT-"].update("Rodando", text_color="green")
        window["-PROG-"].update(current_count=0, max=max(shared_total["value"],1))
        window["-COUNT-"].update(f"0/{shared_total['value']}")
        window["-CANCEL-"].update(disabled=False)
        window["-FORCE-"].update(disabled=False)

        start_worker()

    if event == "-CANCEL-":
        if worker_th and worker_th.is_alive():
            stop_event.set()
            window["-STATUS-TXT-"].update("Cancelando...", text_color="orange")
            window["-OUT-"].print("Cancelamento solicitado (encerra após a etapa atual).")

    if event == "-FORCE-":
        if worker_th and worker_th.is_alive():
            force_event.set()
            window["-STATUS-TXT-"].update("Forçando parada...", text_color="orange")
            window["-OUT-"].print("Parada imediata solicitada (aborta a operação atual).")

    if event == EVT_PROGRESS:
        processed_seen, total, etapa = values[event]
        window["-PROG-"].update(current_count=processed_seen, max=max(total,1))
        window["-COUNT-"].update(f"{processed_seen}/{total}")
        if etapa:
            window["-STATUS-TXT-"].update(f"Rodando - {etapa}", text_color="green")

    elif event == EVT_STATUS:
        window["-STATUS-TXT-"].update(values[event], text_color="orange")

    elif event == EVT_LOG:
        window["-OUT-"].print(values[event])

    elif event == EVT_JOBSTAT:
        try:
            job_id, text, color = values[event]
        except Exception:
            continue
        set_job_status(job_id, text, color)

    elif event == EVT_FINISHED:
        window["-OUT-"].print(values[event])
        flash_taskbar(window, count=8)
        window["-STATUS-TXT-"].update("Parado", text_color="grey20")
        window["-CANCEL-"].update(disabled=True)
        window["-FORCE-"].update(disabled=True)
        start_time = None

window.close()
