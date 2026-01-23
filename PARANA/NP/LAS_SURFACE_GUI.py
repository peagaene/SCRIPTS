import os
import sys
import json
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable

import numpy as np

# GUI
try:
    import PySimpleGUI as sg

    _GUI_LIB = "psg"
except Exception:
    import tkinter as tk  # type: ignore
    from tkinter import filedialog, messagebox

    _GUI_LIB = "tk"

# PDAL
try:
    import pdal
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PDAL não encontrado. Instale com: pip install pdal") from exc

# Optional SciPy for triangulated interpolation
_HAS_SCIPY = True
try:
    from scipy.spatial import Delaunay, cKDTree  # type: ignore
    from scipy.interpolate import LinearNDInterpolator  # type: ignore
except Exception:
    _HAS_SCIPY = False

# Optional scikit-learn (fallback KDTree)
_HAS_SKLEARN = True
try:
    from sklearn.neighbors import KDTree  # type: ignore
except Exception:
    _HAS_SKLEARN = False

# Optional GDAL for raster export
_HAS_GDAL = True
try:
    from osgeo import gdal, osr
except Exception:
    _HAS_GDAL = False


@dataclass
class SurfaceJob:
    lote_label: str
    bloco_label: str
    mi3: str
    npc_path: str


@dataclass
class SurfaceParams:
    base_folder: str
    target_epsg: int = 31982
    mds_cell_size: Tuple[float, float] = (1.0, 1.0)
    mdt_cell_size: Tuple[float, float] = (0.25, 0.25)
    workers: int = min(4, os.cpu_count() or 1)
    overwrite: bool = False
    generate_geotiff: bool = True
    generate_ecw: bool = True


def emit_log(
    message: str,
    *,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    phase: Optional[str] = None,
    color: Optional[str] = None,
    worker_id: Optional[int] = None,
    level: str = "info",
) -> None:
    payload: Dict[str, Any] = {
        "type": "log",
        "message": str(message),
        "level": level,
    }
    if phase:
        payload["phase"] = phase
    if color:
        payload["color"] = color
    if worker_id is not None:
        payload["worker_id"] = worker_id
    if progress_cb:
        try:
            progress_cb(payload)
            return
        except Exception:
            pass
    print(str(message))


def _resolve_cell_size(value: Tuple[float, float]) -> Tuple[float, float]:
    try:
        cx = float(value[0])
        cy = float(value[1])
    except Exception as exc:
        raise ValueError(f"Tamanho de célula inválido: {value}") from exc
    if cx <= 0 or cy <= 0:
        raise ValueError(f"Tamanho de célula deve ser positivo: {value}")
    return (cx, cy)


def generate_surface_products(
    points_arr: Any,
    cell_size_x: float,
    cell_size_y: float,
    epsg: int,
    out_txt_path: str,
    out_las_path: str,
    out_tif_path: Optional[str],
    out_ecw_path: Optional[str],
    log: Optional[Callable[[str, str], None]] = None,
) -> List[str]:
    generated: List[str] = []

    def emit(msg: str, level: str = "info") -> None:
        if log:
            log(msg, level)

    if points_arr.size == 0:
        return generated
    if cell_size_x <= 0 or cell_size_y <= 0:
        emit("Aviso: tamanho de célula inválido.", "warning")
        return generated
    xs = points_arr["X"].astype(float)
    ys = points_arr["Y"].astype(float)
    zs = points_arr["Z"].astype(float)
    if xs.size == 0:
        return generated
    minx, maxx = float(np.min(xs)), float(np.max(xs))
    miny, maxy = float(np.min(ys)), float(np.max(ys))
    if maxx - minx < 1e-6 or maxy - miny < 1e-6:
        emit("Aviso: extensão mínima insuficiente para gerar grade.", "warning")
        return generated
    gx = np.arange(minx, maxx + cell_size_x * 0.5, cell_size_x)
    gy = np.arange(miny, maxy + cell_size_y * 0.5, cell_size_y)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid = np.full(grid_x.shape, np.nan, dtype=np.float32)
    interpolated = False

    if _HAS_SCIPY:
        try:
            tri = Delaunay(np.c_[xs, ys])
            interpolator = LinearNDInterpolator(tri, zs)
            interp_vals = interpolator(grid_x, grid_y)
            if interp_vals is not None:
                grid = np.asarray(interp_vals, dtype=np.float32)
                interpolated = True
        except Exception:
            try:
                tree = cKDTree(np.c_[xs, ys])
                _, idx = tree.query(np.c_[grid_x.ravel(), grid_y.ravel()], k=1)
                grid = zs[idx].reshape(grid_x.shape).astype(np.float32)
                interpolated = True
            except Exception:
                interpolated = False

    if not interpolated and _HAS_SKLEARN:
        try:
            tree = KDTree(np.c_[xs, ys])
            _, ind = tree.query(np.c_[grid_x.ravel(), grid_y.ravel()], k=1)
            grid = zs[ind[:, 0]].reshape(grid_x.shape).astype(np.float32)
            interpolated = True
        except Exception:
            interpolated = False

    if not interpolated:
        emit("Aviso: não foi possível interpolar grade (SciPy/sklearn indisponível).", "warning")
        return generated

    valid_mask = np.isfinite(grid)
    if not valid_mask.any():
        emit("Aviso: grade interpolada vazia.", "warning")
        return generated

    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_las_path), exist_ok=True)
    np.savetxt(
        out_txt_path,
        np.column_stack((grid_x[valid_mask], grid_y[valid_mask], grid[valid_mask])),
        fmt="%.3f %.3f %.3f",
        header="X Y Z",
        comments="",
    )
    generated.append(out_txt_path)

    pipe_json = {
        "pipeline": [
            {"type": "readers.text", "filename": out_txt_path, "header": "X Y Z"},
            {
                "type": "writers.las",
                "filename": out_las_path,
                "scale_x": 0.01,
                "scale_y": 0.01,
                "scale_z": 0.01,
                "offset_x": "auto",
                "offset_y": "auto",
                "offset_z": "auto",
            },
        ]
    }
    pipe = pdal.Pipeline(json.dumps(pipe_json))
    pipe.execute()
    generated.append(out_las_path)

    if (out_tif_path or out_ecw_path) and _HAS_GDAL:
        try:
            if out_tif_path:
                os.makedirs(os.path.dirname(out_tif_path), exist_ok=True)
            if out_ecw_path:
                os.makedirs(os.path.dirname(out_ecw_path), exist_ok=True)
            data = np.where(valid_mask, grid, -9999.0).astype(np.float32)
            data = np.flipud(data)
            origin_x = float(gx[0] - cell_size_x * 0.5)
            origin_y = float(gy[-1] + cell_size_y * 0.5)
            if out_tif_path:
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.Create(
                    out_tif_path,
                    int(grid_x.shape[1]),
                    int(grid_x.shape[0]),
                    1,
                    gdal.GDT_Float32,
                    options=["TILED=YES", "COMPRESS=DEFLATE"],
                )
                if ds is not None:
                    ds.SetGeoTransform((origin_x, cell_size_x, 0.0, origin_y, 0.0, -cell_size_y))
                    if epsg and epsg > 0:
                        try:
                            srs = osr.SpatialReference()
                            srs.ImportFromEPSG(int(epsg))
                            ds.SetProjection(srs.ExportToWkt())
                        except Exception:
                            pass
                    band = ds.GetRasterBand(1)
                    band.WriteArray(np.where(valid_mask, grid, -9999.0))
                    band.SetNoDataValue(-9999.0)
                    band.FlushCache()
                    ds.FlushCache()
                    ds = None
                    generated.append(out_tif_path)
                    if out_ecw_path:
                        try:
                            gdal.Translate(out_ecw_path, out_tif_path, format="ECW")
                            generated.append(out_ecw_path)
                        except Exception:
                            emit("Aviso: falha ao gerar ECW (driver/codec indisponível).", "warning")
                else:
                    emit("Aviso: driver GTiff indisponível para gerar raster.", "warning")
            elif out_ecw_path:
                emit("Aviso: ECW requer GeoTIFF intermediário. Habilite GeoTIFF.", "warning")
        except Exception as exc:
            emit(f"Aviso: erro ao exportar raster: {exc}", "warning")
    elif (out_tif_path or out_ecw_path) and not _HAS_GDAL:
        emit("Aviso: GDAL não disponível para exportar GeoTIFF/ECW.", "warning")

    return generated


def discover_jobs(base_folder: str, log_cb: Optional[Callable[[str], None]] = None) -> List[SurfaceJob]:
    jobs: List[SurfaceJob] = []
    npc_root = os.path.join(base_folder, "5_NUVEM_PONTOS")
    if not os.path.isdir(npc_root):
        if log_cb:
            log_cb(f"Aviso: diretório {npc_root} não encontrado.",)
        return jobs
    for lote_name in sorted(os.listdir(npc_root)):
        lote_path = os.path.join(npc_root, lote_name)
        if not os.path.isdir(lote_path):
            continue
        for bloco_name in sorted(os.listdir(lote_path)):
            bloco_path = os.path.join(lote_path, bloco_name)
            npc_dir = os.path.join(bloco_path, "2_NPc_COMPLETO", "2_1_LAS")
            if not os.path.isdir(npc_dir):
                continue
            for fname in sorted(os.listdir(npc_dir)):
                if not fname.lower().endswith(".laz"):
                    continue
                stem = fname[:-4]
                if "_NPc_T_" in stem:
                    continue  # usamos apenas NPc completo
                parts = stem.split("_")
                if len(parts) < 6 or parts[0] != "ES" or not parts[1].startswith("L") or parts[3] != "NPc":
                    continue
                lote_label = parts[1][1:]
                bloco_label = parts[2]
                mi3 = parts[4]
                jobs.append(
                    SurfaceJob(
                        lote_label=lote_label,
                        bloco_label=bloco_label,
                        mi3=mi3,
                        npc_path=os.path.join(npc_dir, fname),
                    )
                )
    if log_cb:
        log_cb(f"Encontrados {len(jobs)} arquivos NPc para gerar superfícies.")
    return jobs


def process_surface_job(
    job: SurfaceJob,
    params: SurfaceParams,
    subprogress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    worker_id: Optional[int] = None,
) -> Dict[str, Any]:
    start = time.time()
    result: Dict[str, Any] = {
        "status": "ok",
        "MI_3": job.mi3,
        "lote": job.lote_label,
        "bloco": job.bloco_label,
        "generated_files": [],
        "mensagem": "",
    }
    lote_dir = f"LOTE_{job.lote_label}"
    bloco_dir = f"BLOCO_{job.bloco_label}"

    def log_phase(msg: str, *, phase: str, level: str = "info") -> None:
        emit_log(msg, progress_cb=subprogress_cb, phase=phase, worker_id=worker_id, level=level)

    if cancel_event and cancel_event.is_set():
        result["status"] = "cancelado"
        result["mensagem"] = "Cancelado pelo usuário"
        return result

    mds_cx, mds_cy = _resolve_cell_size(params.mds_cell_size)
    mdt_cx, mdt_cy = _resolve_cell_size(params.mdt_cell_size)

    npc_path = job.npc_path
    mds_base = os.path.join(params.base_folder, "6_MDS", lote_dir, bloco_dir)
    mdt_base = os.path.join(params.base_folder, "7_MDT", lote_dir, bloco_dir)

    base_name_mds = f"ES_L{job.lote_label}_{job.bloco_label}_MDS_{job.mi3}_R0"
    base_name_mdt = f"ES_L{job.lote_label}_{job.bloco_label}_MDT_{job.mi3}_R0"

    def build_paths(base_dir: str, basename: str) -> Tuple[str, str, Optional[str], Optional[str]]:
        txt_path = os.path.join(base_dir, "2_ASCII", f"{basename}.txt")
        las_path = os.path.join(base_dir, "1_LAS", f"{basename}.las")
        tif_path = os.path.join(base_dir, "4_GeoTIFF", f"{basename}.tif") if params.generate_geotiff else None
        ecw_path = os.path.join(base_dir, "3_ECW", f"{basename}.ecw") if params.generate_ecw else None
        return txt_path, las_path, tif_path, ecw_path

    mds_txt, mds_las, mds_tif, mds_ecw = build_paths(mds_base, base_name_mds)
    mdt_txt, mdt_las, mdt_tif, mdt_ecw = build_paths(mdt_base, base_name_mdt)

    if not params.overwrite:
        if os.path.exists(mds_las) and os.path.exists(mdt_las):
            result["status"] = "pulado_existente"
            result["mensagem"] = "MDS/MDT já gerados"
            return result

    try:
        if subprogress_cb:
            subprogress_cb({"type": "substep", "phase": "MDS", "status": "start", "mi3": job.mi3, "worker_id": worker_id})

        mds_pipe = pdal.Pipeline(json.dumps({
            "pipeline": [
                {"type": "readers.las", "filename": npc_path},
                {"type": "filters.range", "limits": "Classification![7:7]"},
            ]
        }))
        mds_pipe.execute()
        mds_arrs = mds_pipe.arrays
        if mds_arrs:
            generated_mds = generate_surface_products(
                mds_arrs[0],
                mds_cx,
                mds_cy,
                params.target_epsg,
                mds_txt,
                mds_las,
                mds_tif,
                mds_ecw,
                log=lambda msg, level="info": log_phase(msg, phase="MDS", level=level),
            )
            result["generated_files"].extend(generated_mds)
        else:
            log_phase("Aviso: nenhum ponto disponível para MDS.", phase="MDS", level="warning")
        if subprogress_cb:
            subprogress_cb({"type": "substep", "phase": "MDS", "status": "done", "mi3": job.mi3, "worker_id": worker_id})

        if subprogress_cb:
            subprogress_cb({"type": "substep", "phase": "MDT", "status": "start", "mi3": job.mi3, "worker_id": worker_id})

        mdt_pipe = pdal.Pipeline(json.dumps({
            "pipeline": [
                {"type": "readers.las", "filename": npc_path},
                {"type": "filters.range", "limits": "Classification[2:2]"},
            ]
        }))
        mdt_pipe.execute()
        mdt_arrs = mdt_pipe.arrays
        if mdt_arrs:
            generated_mdt = generate_surface_products(
                mdt_arrs[0],
                mdt_cx,
                mdt_cy,
                params.target_epsg,
                mdt_txt,
                mdt_las,
                mdt_tif,
                mdt_ecw,
                log=lambda msg, level="info": log_phase(msg, phase="MDT", level=level),
            )
            result["generated_files"].extend(generated_mdt)
        else:
            log_phase("Aviso: nenhum ponto disponível para MDT.", phase="MDT", level="warning")
        if subprogress_cb:
            subprogress_cb({"type": "substep", "phase": "MDT", "status": "done", "mi3": job.mi3, "worker_id": worker_id})

        result["saida_filename"] = "; ".join(result["generated_files"])
    except Exception as exc:
        result["status"] = "erro"
        result["mensagem"] = str(exc)
    finally:
        result["duracao_s"] = round(time.time() - start, 3)
    return result


def run_surface_processing(
    params: SurfaceParams,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> List[Dict[str, Any]]:
    jobs = discover_jobs(params.base_folder, log_cb=lambda msg: emit_log(msg, progress_cb=progress_cb))
    results: List[Dict[str, Any]] = []
    total_jobs = len(jobs)

    if progress_cb:
        progress_cb({"type": "block_start", "block_idx": 1, "total": total_jobs})

    def subprogress(payload: Dict[str, Any]) -> None:
        if progress_cb:
            progress_cb(payload)

    if params.workers and params.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        slot_lock = threading.Lock()
        slots = list(range(1, (params.workers or 1) + 1))

        def acquire_slot() -> int:
            with slot_lock:
                return slots.pop(0) if slots else 0

        def release_slot(slot: int) -> None:
            if slot:
                with slot_lock:
                    slots.append(slot)

        with ThreadPoolExecutor(max_workers=params.workers) as ex:
            futs = []
            for job in jobs:
                if cancel_event and cancel_event.is_set():
                    break

                def submit_job(j: SurfaceJob) -> Any:
                    wid = acquire_slot()

                    def task() -> Dict[str, Any]:
                        try:
                            return process_surface_job(j, params, subprogress_cb=subprogress, cancel_event=cancel_event, worker_id=wid)
                        finally:
                            release_slot(wid)

                    return ex.submit(task)

                futs.append(submit_job(job))

            completed = 0
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {"status": "erro", "mensagem": str(exc), "generated_files": []}
                results.append(res)
                completed += 1
                if progress_cb:
                    progress_cb({"type": "block_progress", "block_idx": 1, "completed": completed, "total": total_jobs})
    else:
        for idx, job in enumerate(jobs, 1):
            if cancel_event and cancel_event.is_set():
                results.append({"status": "cancelado", "mensagem": "Cancelado pelo usuário", "generated_files": []})
                continue
            res = process_surface_job(job, params, subprogress_cb=subprogress, cancel_event=cancel_event, worker_id=1)
            results.append(res)
            if progress_cb:
                progress_cb({"type": "block_progress", "block_idx": 1, "completed": idx, "total": total_jobs})

    if progress_cb:
        progress_cb({"type": "block_done", "block_idx": 1, "total": total_jobs})
    return results


def build_gui_and_run() -> None:
    if _GUI_LIB == "psg":
        sg.theme("SystemDefault")
        tab_layout = [
            [sg.Text("Pasta base (contendo 5_NUVEM_PONTOS)"), sg.Input(key="base_folder"), sg.FolderBrowse("Browse")],
            [sg.Text("EPSG alvo"), sg.Input(key="epsg", size=(8, 1), default_text="31982"),
             sg.Text("MDS passo X/Y (m)"), sg.Input(key="mds_x", size=(6, 1), default_text="1.0"),
             sg.Input(key="mds_y", size=(6, 1), default_text="1.0")],
            [sg.Text("MDT passo X/Y (m)"), sg.Input(key="mdt_x", size=(6, 1), default_text="0.25"),
             sg.Input(key="mdt_y", size=(6, 1), default_text="0.25")],
            [sg.Checkbox("Gerar GeoTIFF", default=True, key="gen_tif"),
             sg.Checkbox("Gerar ECW", default=True, key="gen_ecw"),
             sg.Checkbox("Sobrescrever existentes", default=False, key="overwrite")],
            [sg.Text("Workers"), sg.Spin(values=[1, 2, 3, 4, 5, 6, 7, 8], initial_value=min(4, os.cpu_count() or 1), key="workers")],
            [sg.Button("Gerar", key="EXEC", size=(15, 2)), sg.Button("Cancelar", key="CANCEL", size=(15, 2)), sg.Button("Fechar", size=(15, 2))],
            [sg.ProgressBar(max_value=100, orientation='h', size=(50, 20), key='PROG'), sg.Text("", key="PROG_TXT")],
            [sg.Multiline(size=(100, 18), key="log", autoscroll=True, reroute_stdout=True, reroute_stderr=True)],
        ]
        layout = [[sg.TabGroup([[sg.Tab('Gerar MDS/MDT', tab_layout)]])]]
        win = sg.Window("Geração de MDS/MDT a partir de NPc", layout, finalize=True, resizable=True)

        current_total = 0
        current_done = 0
        cancel_evt: Optional[threading.Event] = None
        conv_total = conv_done = 0  # dummy placeholders for reuse

        phase_colors = {
            "MDS": "firebrick",
            "MDT": "royalblue",
        }

        def print_phase_log(prefix: str, phase: str, suffix: str) -> None:
            color = phase_colors.get(phase)
            try:
                win['log'].print(prefix, end='')
                win['log'].print(phase, text_color=color, end='')
                win['log'].print(suffix)
            except Exception:
                try:
                    win['log'].print(f"{prefix}{phase}{suffix}")
                except Exception:
                    pass

        def run_in_thread(params_local: SurfaceParams, cancel_ev: Optional[threading.Event]) -> None:
            try:
                def progress_event(payload: Dict[str, Any]) -> None:
                    win.write_event_value('-PROGRESS-', payload)
                run_surface_processing(params_local, progress_cb=progress_event, cancel_event=cancel_ev)
                win.write_event_value('-DONE-', None)
            except Exception as exc:
                win.write_event_value('-ERROR-', str(exc))

        while True:
            ev, values = win.read(timeout=150)
            if ev in (sg.WINDOW_CLOSED, "Fechar"):
                break
            if ev == "EXEC":
                try:
                    base_folder = str(values.get("base_folder") or "").strip()
                    if not base_folder:
                        sg.popup_error("Informe a pasta base.")
                        continue
                    try:
                        mds_cell = (float(values.get("mds_x") or 1.0), float(values.get("mds_y") or 1.0))
                        mdt_cell = (float(values.get("mdt_x") or 0.25), float(values.get("mdt_y") or 0.25))
                        mds_cell = _resolve_cell_size(mds_cell)
                        mdt_cell = _resolve_cell_size(mdt_cell)
                    except Exception as exc:
                        sg.popup_error(f"Tamanhos de célula inválidos: {exc}")
                        continue
                    params = SurfaceParams(
                        base_folder=base_folder,
                        target_epsg=int(values.get("epsg") or 31982),
                        mds_cell_size=mds_cell,
                        mdt_cell_size=mdt_cell,
                        workers=int(values.get("workers") or 1),
                        overwrite=bool(values.get("overwrite")),
                        generate_geotiff=bool(values.get("gen_tif")),
                        generate_ecw=bool(values.get("gen_ecw")),
                    )
                    if params.workers < 1:
                        params.workers = 1

                    win['EXEC'].update(disabled=True)
                    for key in ("base_folder", "epsg", "mds_x", "mds_y", "mdt_x", "mdt_y", "gen_tif", "gen_ecw", "overwrite", "workers"):
                        try:
                            win[key].update(disabled=True)
                        except Exception:
                            pass
                    win['CANCEL'].update(disabled=False)
                    win['log'].update("")
                    win['PROG'].update_bar(0)
                    win['PROG_TXT'].update("Iniciando...")
                    cancel_evt = threading.Event()
                    threading.Thread(target=run_in_thread, args=(params, cancel_evt), daemon=True).start()
                except Exception as exc:
                    sg.popup_error(f"Erro: {exc}")
            elif ev == "CANCEL":
                if cancel_evt is not None:
                    cancel_evt.set()
                    win['CANCEL'].update(disabled=True)
                    win['PROG_TXT'].update("Cancelando... aguardando término")
            elif ev == '-PROGRESS-':
                payload = values.get('-PROGRESS-') or {}
                if payload.get('type') == 'block_start':
                    current_total = int(payload.get('total') or 0)
                    current_done = 0
                elif payload.get('type') == 'block_progress':
                    current_done = int(payload.get('completed') or 0)
                    current_total = int(payload.get('total') or current_total)
                elif payload.get('type') == 'block_done':
                    current_done = current_total
                elif payload.get('type') == 'substep':
                    phase = payload.get('phase')
                    status = payload.get('status')
                    mi3 = payload.get('mi3')
                    wid = payload.get('worker_id')
                    if status == 'start':
                        print_phase_log(f"Worker {wid}: iniciando ", phase, f" de MI_3={mi3}")
                    elif status == 'done':
                        print_phase_log(f"Worker {wid}: finalizou ", phase, f" de MI_3={mi3}")
                elif payload.get('type') == 'log':
                    message = payload.get('message', '')
                    phase = payload.get('phase')
                    color = payload.get('color') or phase_colors.get(phase)
                    level = payload.get('level', 'info')
                    worker = payload.get('worker_id')
                    if not color:
                        if level == 'warning':
                            color = 'goldenrod'
                        elif level == 'error':
                            color = 'red'
                    prefix = f"[W{worker}] " if worker else ""
                    try:
                        if color:
                            win['log'].print(prefix + message, text_color=color)
                        else:
                            win['log'].print(prefix + message)
                    except Exception:
                        pass
                total = max(1, current_total)
                pct = int((current_done / total) * 100)
                win['PROG'].update_bar(pct)
                win['PROG_TXT'].update(f"{current_done}/{current_total} ({pct}%)")
            elif ev == '-DONE-':
                win['EXEC'].update(disabled=False)
                win['CANCEL'].update(disabled=True)
                for key in ("base_folder", "epsg", "mds_x", "mds_y", "mdt_x", "mdt_y", "gen_tif", "gen_ecw", "overwrite", "workers"):
                    try:
                        win[key].update(disabled=False)
                    except Exception:
                        pass
                sg.popup("Processo concluído", keep_on_top=True)
            elif ev == '-ERROR-':
                win['EXEC'].update(disabled=False)
                win['CANCEL'].update(disabled=True)
                for key in ("base_folder", "epsg", "mds_x", "mds_y", "mdt_x", "mdt_y", "gen_tif", "gen_ecw", "overwrite", "workers"):
                    try:
                        win[key].update(disabled=False)
                    except Exception:
                        pass
                sg.popup_error(f"Erro: {values.get('-ERROR-')}")

        win.close()
    else:
        # Fallback simplificado com Tkinter
        root = tk.Tk()
        root.withdraw()
        base_dir = filedialog.askdirectory(title="Selecione a pasta base (contendo 5_NUVEM_PONTOS)")
        if not base_dir:
            messagebox.showinfo("Info", "Nenhuma pasta selecionada.")
            return
        params = SurfaceParams(base_folder=base_dir)
        _run_headless(params)
        messagebox.showinfo("Info", "Processo concluído.")
        root.destroy()


def _run_headless(params: SurfaceParams) -> None:
    t0 = time.time()
    results = run_surface_processing(params)
    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "pulado_existente")
    errors = [r for r in results if r.get("status") == "erro"]
    print(f"Concluído em {round(time.time() - t0, 2)} s")
    print(f"Gerados: {ok} | Pulados: {skipped} | Erros: {len(errors)}")
    if errors:
        for err in errors:
            print(f"  - MI_3={err.get('MI_3')}: {err.get('mensagem')}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--nogui":
        import argparse

        def parse_cell(value: str, default: Tuple[float, float]) -> Tuple[float, float]:
            parts = [p.strip() for p in value.split(",") if p.strip() != ""]
            try:
                if not parts:
                    return default
                if len(parts) == 1:
                    parts *= 2
                if len(parts) != 2:
                    raise ValueError("informe valores no formato dx,dy")
                cells = (float(parts[0]), float(parts[1]))
                if cells[0] <= 0 or cells[1] <= 0:
                    raise ValueError("valores devem ser positivos")
                return cells
            except Exception as exc:
                raise SystemExit(f"Tamanho de célula inválido '{value}': {exc}") from exc

        ap = argparse.ArgumentParser(description="Gerar MDS/MDT a partir das nuvens NPc.")
        ap.add_argument("--base", dest="base_folder", required=True, help="Pasta base contendo 5_NUVEM_PONTOS")
        ap.add_argument("--epsg", type=int, default=31982, help="EPSG alvo para os rasters")
        ap.add_argument("--mds-cell", default="1.0,1.0", help="Passo da malha MDS (dx,dy)")
        ap.add_argument("--mdt-cell", default="0.25,0.25", help="Passo da malha MDT (dx,dy)")
        ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
        ap.add_argument("--overwrite", action="store_true", help="Sobrescrever arquivos existentes")
        ap.add_argument("--no-geotiff", action="store_true", help="Não exportar GeoTIFF")
        ap.add_argument("--no-ecw", action="store_true", help="Não exportar ECW")
        args = ap.parse_args()
        params = SurfaceParams(
            base_folder=args.base_folder,
            target_epsg=args.epsg,
            mds_cell_size=parse_cell(args.mds_cell, (1.0, 1.0)),
            mdt_cell_size=parse_cell(args.mdt_cell, (0.25, 0.25)),
            workers=max(1, args.workers),
            overwrite=args.overwrite,
            generate_geotiff=not args.no_geotiff,
            generate_ecw=not args.no_ecw,
        )
        _run_headless(params)
    else:
        build_gui_and_run()
