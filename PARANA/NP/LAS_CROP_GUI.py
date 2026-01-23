import os
import sys
import json
import time
import math
import subprocess
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable
import threading
import numpy as np
# GUI
try:
    import PySimpleGUI as sg
    _GUI_LIB = "psg"
except Exception:
    import tkinter as tk  # type: ignore
    from tkinter import filedialog, messagebox
    _GUI_LIB = "tk"
# Geo stack
import geopandas as gpd
from shapely.geometry import Polygon
from shapely import wkt as shapely_wkt
# PDAL
try:
    import pdal
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PDAL não encontrado. Instale com: pip install pdal") from exc
# Fast bounds from LAZ tiles
_HAS_LASPY = True
try:
    import laspy
except Exception:
    _HAS_LASPY = False
try:
    from rtree import index as rtree_index
    _HAS_RTREE = True
except Exception:
    _HAS_RTREE = False
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
# Caminho fixo do shapefile de articulação (pode ser editado aqui se necessário)
DEFAULT_ARTICULATION_SHP = r"\\192.168.2.28\i\80225_PROJETO_IAT_PARANA\5 Processamento Laser\Articulacao_Auxiliar_5000+50.shp"
# Caminho do LASzip para conversão LAS -> LAZ
LASZIP_PATH = r'D:\LAStools\bin\laszip.exe'
@dataclass
class BlockConfig:
    """Configuration for a single block processing."""
    lote: str
    bloco_text: str
    output_folder_default: str  # Pasta base (NP)
    output_folder_classified: str  # Pasta base (NPc completo)
    output_folder_terrain: str  # Pasta base (NPc_T terreno)
@dataclass
class AppParams:
    input_folder: str
    blocks: List[BlockConfig]  # Lista de blocos para processar
    articulation_shp: str = DEFAULT_ARTICULATION_SHP
    target_epsg: int = 31982
    buffer_m: float = 0.0
    overwrite: bool = False
    workers: int = min(4, os.cpu_count() or 1)
    convert_to_laz: bool = False  # Se True, converte LAS para LAZ após processamento
    generate_surfaces: bool = True
    mds_cell_size: Tuple[float, float] = (1.0, 1.0)
    mdt_cell_size: Tuple[float, float] = (0.25, 0.25)
@dataclass
class TileInfo:
    path: str
    bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy


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



def read_tile_bounds(tile_path: str) -> Optional[Tuple[float, float, float, float]]:
    """Return bounds from LAZ/LAS header without loading points.
    Prefer laspy for speed; fallback to PDAL info.
    """
    try:
        if _HAS_LASPY:
            with laspy.open(tile_path) as f:  # type: ignore
                hdr = f.header
                return (float(hdr.mins[0]), float(hdr.mins[1]), float(hdr.maxs[0]), float(hdr.maxs[1]))
        # Fallback to PDAL info
        pipeline_json = {
            "pipeline": [
                {"type": "readers.las", "filename": tile_path},
                {"type": "filters.info"}
            ]
        }
        pipe = pdal.Pipeline(json.dumps(pipeline_json))
        pipe.execute()
        meta = json.loads(pipe.metadata)
        bounds = meta["metadata"]["filters.info"]["bbox"]
        return (bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"])  # type: ignore
    except Exception:
        return None
def scan_tiles(input_folder: str) -> List[TileInfo]:
    tiles: List[TileInfo] = []
    for root, _dirs, files in os.walk(input_folder):
        for name in files:
            if name.lower().endswith((".laz", ".las")):
                path = os.path.join(root, name)
                b = read_tile_bounds(path)
                if b is not None:
                    tiles.append(TileInfo(path=path, bounds=b))
    return tiles
class TileIndex:
    def __init__(self, tiles: List[TileInfo]):
        self.tiles = tiles
        if _HAS_RTREE and tiles:
            p = rtree_index.Property()
            p.interleaved = True
            self.idx = rtree_index.Index(properties=p)
            for i, t in enumerate(tiles):
                self.idx.insert(i, t.bounds)
        else:
            self.idx = None
    def query(self, bbox: Tuple[float, float, float, float]) -> List[TileInfo]:
        if self.idx is None:
            # linear scan
            minx, miny, maxx, maxy = bbox
            out: List[TileInfo] = []
            for t in self.tiles:
                tminx, tminy, tmaxx, tmaxy = t.bounds
                if not (tmaxx < minx or tmaxy < miny or tminx > maxx or tminy > maxy):
                    out.append(t)
            return out
        ids = list(self.idx.intersection(bbox))
        return [self.tiles[i] for i in ids]


def _resolve_cell_size(cell: Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(cell, (list, tuple)) and len(cell) == 2:
        try:
            return (float(cell[0]), float(cell[1]))
        except Exception:
            pass
    try:
        v = float(cell)  # type: ignore[arg-type]
        return (v, v)
    except Exception:
        return (1.0, 1.0)


def generate_surface_products(
    points_arr: Any,
    cell_size_x: float,
    cell_size_y: float,
    epsg: int,
    out_txt_path: str,
    out_las_path: str,
    out_tif_path: Optional[str] = None,
    out_ecw_path: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
    phase: Optional[str] = None,
) -> List[str]:
    generated: List[str] = []
    def emit(msg: str, level: str = "info") -> None:
        if log:
            log(msg, level)

    if points_arr.size == 0:
        return generated
    if cell_size_x <= 0 or cell_size_y <= 0:
        emit("Aviso: Tamanho de célula inválido para geração de superfícies.", "warning")
        return generated
    xs = points_arr["X"].astype(float)
    ys = points_arr["Y"].astype(float)
    zs = points_arr["Z"].astype(float)
    if xs.size == 0:
        return generated
    minx, maxx = float(np.min(xs)), float(np.max(xs))
    miny, maxy = float(np.min(ys)), float(np.max(ys))
    if maxx - minx < 1e-6 or maxy - miny < 1e-6:
        emit("Aviso: Extensão mínima insuficiente para gerar grade.", "warning")
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
            dist, ind = tree.query(np.c_[grid_x.ravel(), grid_y.ravel()], k=1)
            grid = zs[ind[:, 0]].reshape(grid_x.shape).astype(np.float32)
            interpolated = True
        except Exception:
            interpolated = False
    if not interpolated:
        emit("Aviso: Não foi possível interpolar grade (SciPy/sklearn indisponível).", "warning")
        return generated
    valid_mask = np.isfinite(grid)
    if not valid_mask.any():
        emit("Aviso: Grade interpolada vazia.", "warning")
        return generated
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    np.savetxt(
        out_txt_path,
        np.column_stack((grid_x[valid_mask], grid_y[valid_mask], grid[valid_mask])),
        fmt="%.3f %.3f %.3f",
        newline="\n",
    )
    generated.append(out_txt_path)
    pipe_json = {
        "pipeline": [
            {
                "type": "readers.text",
                "filename": out_txt_path,
                "skip": 0,
                "dimensions": "X=0:double,Y=1:double,Z=2:double",
            },
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
    os.makedirs(os.path.dirname(out_las_path), exist_ok=True)
    pipe = pdal.Pipeline(json.dumps(pipe_json))
    pipe.execute()
    generated.append(out_las_path)
    if out_tif_path or out_ecw_path:
        if not _HAS_GDAL:
            emit("Aviso: GDAL não disponível para exportar GeoTIFF/ECW.", "warning")
        else:
            try:
                os.makedirs(os.path.dirname(out_tif_path or ""), exist_ok=True)
                data = np.where(valid_mask, grid, -9999.0).astype(np.float32)
                data = np.flipud(data)
                origin_x = float(gx[0] - cell_size_x * 0.5)
                origin_y = float(gy[-1] + cell_size_y * 0.5)
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
                    ds.SetGeoTransform(
                        (origin_x, cell_size_x, 0.0, origin_y, 0.0, -cell_size_y)
                    )
                    if epsg and epsg > 0:
                        try:
                            srs = osr.SpatialReference()
                            srs.ImportFromEPSG(int(epsg))
                            ds.SetProjection(srs.ExportToWkt())
                        except Exception:
                            pass
                    band = ds.GetRasterBand(1)
                    band.WriteArray(data)
                    band.SetNoDataValue(-9999.0)
                    band.FlushCache()
                    ds.FlushCache()
                    ds = None
                    generated.append(out_tif_path)
                    if out_ecw_path:
                        try:
                            os.makedirs(os.path.dirname(out_ecw_path), exist_ok=True)
                            gdal.Translate(out_ecw_path, out_tif_path, format="ECW")
                            generated.append(out_ecw_path)
                        except Exception:
                            emit("Aviso: Falha ao gerar ECW (driver/codec indisponível).", "warning")
                else:
                    emit("Aviso: Driver GTiff indisponível para gerar raster.", "warning")
            except Exception as exc:
                emit(f"Aviso: erro ao exportar raster: {exc}", "warning")
    return generated
def ensure_crs(gdf: gpd.GeoDataFrame, target_epsg: int) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Shapefile sem CRS definido.")
    if gdf.crs.to_epsg() != target_epsg:
        gdf = gdf.to_crs(epsg=target_epsg)
    return gdf
def norm_lote(v):
    """Normalize LOTE value: '09' -> '9', '9.0' -> '9'"""
    s = str(v).strip()
    try:
        return str(int(float(s)))  # '09' -> '9', '9.0' -> '9'
    except:
        return s  # se vier algo não numérico
def norm_bloco(v):
    """Normalize BLOCOS value: convert to uppercase"""
    return str(v).strip().upper()
def convert_las_to_laz(las_file: str) -> bool:
    """Convert LAS file to LAZ using LASzip. Returns True if successful."""
    if not os.path.exists(LASZIP_PATH):
        print(f"⚠️ LASzip não encontrado em: {LASZIP_PATH}")
        return False
    
    laz_file = las_file.replace('.las', '.laz')
    if os.path.exists(laz_file):
        print(f"⏭️ LAZ já existe: {os.path.basename(laz_file)}")
        return True
    
    try:
        command = [LASZIP_PATH, '-i', las_file, '-o', laz_file]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            print(f"✅ Convertido: {os.path.basename(las_file)} -> {os.path.basename(laz_file)}")
            # Remove the original LAS file
            os.remove(las_file)
            print(f"🗑️ Removido: {os.path.basename(las_file)}")
            return True
        else:
            print(f"❌ Erro na conversão: {os.path.basename(las_file)}")
            print(f"   Erro: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        print(f"❌ Erro na conversão: {os.path.basename(las_file)} - {str(e)}")
        return False
def get_block_from_articulation(row) -> Tuple[Optional[str], Optional[str]]:
    """Return (lote, bloco) from articulation shapefile columns."""
    lote = row.get("LOTE")
    bloco = row.get("BLOCOS")
    lote_str = str(lote) if lote is not None and str(lote).strip() != "" else None
    bloco_str = str(bloco) if bloco is not None and str(bloco).strip() != "" else None
    return lote_str, bloco_str
def build_pdal_crop_pipeline(
    input_files: List[str],
    crop_polygon: Polygon,
    out_file: str,
    change_to_class: Optional[int] = None,
    classification_filter: Optional[str] = None,
) -> str:
    readers = [{"type": "readers.las", "filename": f} for f in input_files]
    filters = [
        {"type": "filters.merge"},
        {"type": "filters.crop", "polygon": crop_polygon.wkt}
    ]
    if change_to_class is not None:
        filters.append({"type": "filters.assign", "assignment": f"Classification[:]={change_to_class}"})
    if classification_filter:
        filters.append({"type": "filters.range", "limits": classification_filter})
    
    # Optimized writer settings (writes LAZ if filename ends with .laz and PDAL has LAZ support)
    writer = {
        "type": "writers.las", 
        "filename": out_file,
        "scale_x": 0.01,       # Reduce precision for speed
        "scale_y": 0.01,
        "scale_z": 0.01,
        "offset_x": "auto",
        "offset_y": "auto", 
        "offset_z": "auto"
    }
    
    pipeline = readers + filters + [writer]
    return json.dumps({"pipeline": pipeline})
def _run_pdal_cli_with_cancel(pipeline_json: str, outputs: List[str], cancel_event: Optional[threading.Event]) -> None:
    """Run PDAL pipeline as a subprocess with cooperative cancel.
    Writes pipeline JSON to a temp file and runs `pdal pipeline <file>`.
    If `cancel_event` is set during execution, tries to terminate and cleans `outputs`.
    Raises RuntimeError on cancel or non-zero return.
    """
    import tempfile
    tmp_json = None
    try:
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as f:
            f.write(pipeline_json)
            tmp_json = f.name
        proc = subprocess.Popen(["pdal", "pipeline", tmp_json], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if cancel_event is not None and cancel_event.is_set():
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.kill()
                except Exception:
                    pass
                for op in outputs:
                    try:
                        if op and os.path.exists(op):
                            os.remove(op)
                    except Exception:
                        pass
                raise RuntimeError("cancelado")
            time.sleep(0.1)
        if rc != 0:
            out, err = proc.communicate()
            raise RuntimeError(err.strip() or f"PDAL CLI retornou {rc}")
    finally:
        if tmp_json and os.path.exists(tmp_json):
            try:
                os.remove(tmp_json)
            except Exception:
                pass
def check_existing_files(output_folder: str, filename_base: str) -> bool:
    """Check if LAS or LAZ files already exist for this articulation."""
    las_file = os.path.join(output_folder, f"{filename_base}.las")
    laz_file = os.path.join(output_folder, f"{filename_base}.laz")
    
    return os.path.exists(las_file) or os.path.exists(laz_file)


def resolve_output_paths(block_config: BlockConfig, lote_value: str, bloco_value: str) -> Dict[str, str]:
    """Build output directories for NP, NPc completo e NPc_T."""
    lote_norm = str(norm_lote(lote_value) or "").strip()
    bloco_norm = str(norm_bloco(bloco_value) or "").strip()
    if not lote_norm:
        lote_norm = "DESCONHECIDO"
    if not bloco_norm:
        bloco_norm = "DESCONHECIDO"

    lote_dir = f"LOTE_{lote_norm.zfill(2) if lote_norm.isdigit() else lote_norm}"
    bloco_dir = f"BLOCO_{bloco_norm}"

    base_np = block_config.output_folder_default
    base_npc = block_config.output_folder_classified
    base_npt = block_config.output_folder_terrain

    np_dir = os.path.join(base_np, lote_dir, bloco_dir, "1_NP", "1_1_LAS")
    npc_dir = os.path.join(base_npc, lote_dir, bloco_dir, "2_NPc_COMPLETO", "2_1_LAS")
    npc_terrain_dir = os.path.join(base_npt, lote_dir, bloco_dir, "3_NPc_TERRENO", "3_1_LAS")

    return {
        "np_dir": np_dir,
        "npc_dir": npc_dir,
        "npc_terrain_dir": npc_terrain_dir,
        "base_np": base_np,
        "lote_dir": lote_dir,
        "bloco_dir": bloco_dir,
    }


def _count_points(path: str) -> int:
    """Return the number of points in a LAS/LAZ file."""
    if not path or not os.path.exists(path):
        return 0
    if _HAS_LASPY:
        try:
            with laspy.open(path) as las:  # type: ignore
                return int(las.header.point_count)
        except Exception:
            pass
    try:
        info_pipe = pdal.Pipeline(json.dumps({
            "pipeline": [
                {"type": "readers.las", "filename": path},
                {"type": "filters.info"}
            ]
        }))
        info_pipe.execute()
        meta = json.loads(info_pipe.metadata)
        return int(meta["metadata"]["filters.info"]["count"])
    except Exception:
        return 0


def process_one_feature(
    feat: Dict[str, Any],
    subprogress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    log_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    worker_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Worker function for parallel processing.
    Expects dict with keys: params (AppParams as dict), block_config (BlockConfig as dict), poly_wkt, mi3, lote_from_shp, bloco_from_shp, tiles (list[TileInfo]).
    """
    start = time.time()
    params_dict = feat["params"]
    block_dict = feat["block_config"]
    params = AppParams(**params_dict)
    block_config = BlockConfig(**block_dict)
    poly = shapely_wkt.loads(feat["poly_wkt"])  # rehydrate geometry in worker
    mi3 = feat["mi3"]
    lote_from_shp = feat.get("lote_from_shp")
    bloco_from_shp = feat.get("bloco_from_shp")
    tiles: List[TileInfo] = [TileInfo(**t) for t in feat["tiles"]]
    # Select tiles by bbox with small buffer for better coverage
    minx, miny, maxx, maxy = poly.bounds
    buffer = 50.0  # 50m buffer for better tile selection
    bbox = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)

    selected_files: List[str] = []
    for t in tiles:
        tminx, tminy, tmaxx, tmaxy = t.bounds
        if not (tmaxx < bbox[0] or tmaxy < bbox[1] or tminx > bbox[2] or tminy > bbox[3]):
            selected_files.append(t.path)
    completed_steps = 0
    result: Dict[str, Any] = {
        "saida_filename": None,
        "MI_3": mi3,
        "lote": block_config.lote,
        "bloco_digitado": block_config.bloco_text,
        "lote_do_shapefile": lote_from_shp,
        "bloco_do_shapefile": bloco_from_shp,
        "n_tiles": len(selected_files),
        "area_intersecao_m2": feat.get("area_inter", 0.0),
        "epsg_usado": params.target_epsg,
        "data_hora": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duracao_s": 0.0,
        "status": "ok",
        "mensagem": "",
        "generated_files": [],
        "n_pts_npc_terreno": 0,
        "worker_id": worker_id,
        "completed_steps": 0,
    }
    if not selected_files:
        result["status"] = "sem_tiles"
        result["mensagem"] = "Nenhum tile intersecta o polígono"
        result["completed_steps"] = completed_steps
        return result

    def log_message(msg: str, *, phase: Optional[str] = None, color: Optional[str] = None, level: str = "info", worker: Optional[int] = None) -> None:
        emit_log(msg, progress_cb=log_cb, phase=phase, color=color, worker_id=worker if worker is not None else worker_id, level=level)

    lote_to_use = lote_from_shp if lote_from_shp else block_config.lote
    bloco_to_use = bloco_from_shp if bloco_from_shp else block_config.bloco_text
    paths = resolve_output_paths(block_config, lote_to_use, bloco_to_use)
    np_dir = paths["np_dir"]
    npc_dir = paths["npc_dir"]
    npc_terrain_dir = paths["npc_terrain_dir"]

    out_name_default = f"ES_L{str(lote_to_use).zfill(2)}_{bloco_to_use}_NP_{mi3}_R0"
    out_name_classified = f"ES_L{str(lote_to_use).zfill(2)}_{bloco_to_use}_NPC_{mi3}_R0"
    out_name_terrain = f"ES_L{str(lote_to_use).zfill(2)}_{bloco_to_use}_NPC_T_{mi3}_R0"

    out_path_default = os.path.join(np_dir, f"{out_name_default}.laz")
    out_path_classified = os.path.join(npc_dir, f"{out_name_classified}.laz")
    out_path_terrain = os.path.join(npc_terrain_dir, f"{out_name_terrain}.laz")

    if not params.overwrite:
        default_exists = check_existing_files(np_dir, out_name_default)
        classified_exists = check_existing_files(npc_dir, out_name_classified)
        terrain_exists = check_existing_files(npc_terrain_dir, out_name_terrain)
        if default_exists and classified_exists and terrain_exists:
            result["status"] = "pulado_existente"
            result["saida_filename"] = f"{out_name_default}.las/.laz; {out_name_classified}.las/.laz; {out_name_terrain}.las/.laz"
            result["mensagem"] = "Arquivos já existem nas pastas de entrega"
            log_message(f"⚠️ MI_3={mi3}: arquivos já existem - pulando", worker=worker_id)
            result["completed_steps"] = completed_steps
            return result
        elif default_exists or classified_exists or terrain_exists:
            faltantes: List[str] = []
            if not default_exists:
                faltantes.append("NP")
            if not classified_exists:
                faltantes.append("NPc")
            if not terrain_exists:
                faltantes.append("NPc_T")
            log_message(f"⚠️ MI_3={mi3}: faltando {', '.join(faltantes)} - processando...", worker=worker_id)

    try:
        os.makedirs(np_dir, exist_ok=True)
        os.makedirs(npc_dir, exist_ok=True)
        os.makedirs(npc_terrain_dir, exist_ok=True)

        if subprogress_cb:
            subprogress_cb({"type": "substep", "phase": "NP", "status": "start", "mi3": mi3, "worker_id": worker_id})
        pipe_json_default = build_pdal_crop_pipeline(selected_files, poly, out_path_default, change_to_class=1)
        _run_pdal_cli_with_cancel(pipe_json_default, [out_path_default], cancel_event)
        default_points = _count_points(out_path_default)
        if subprogress_cb:
            subprogress_cb({"type": "substep", "phase": "NP", "status": "done", "mi3": mi3, "worker_id": worker_id})
        completed_steps += 1

        if subprogress_cb:
            subprogress_cb({"type": "substep", "phase": "NPC", "status": "start", "mi3": mi3, "worker_id": worker_id})
        pipe_json_classified = build_pdal_crop_pipeline(selected_files, poly, out_path_classified)
        _run_pdal_cli_with_cancel(pipe_json_classified, [out_path_classified], cancel_event)
        classified_points = _count_points(out_path_classified)
        if subprogress_cb:
            subprogress_cb({"type": "substep", "phase": "NPC", "status": "done", "mi3": mi3, "worker_id": worker_id})
        completed_steps += 1

        if subprogress_cb:
            subprogress_cb({"type": "substep", "phase": "NPC_T", "status": "start", "mi3": mi3, "worker_id": worker_id})
        pipe_json_terrain = build_pdal_crop_pipeline(
            selected_files,
            poly,
            out_path_terrain,
            classification_filter="Classification[2:2]",
        )
        _run_pdal_cli_with_cancel(pipe_json_terrain, [out_path_terrain], cancel_event)
        terrain_points = _count_points(out_path_terrain)
        if subprogress_cb:
            subprogress_cb({"type": "substep", "phase": "NPC_T", "status": "done", "mi3": mi3, "worker_id": worker_id})
        completed_steps += 1

        result["n_pts_np"] = default_points
        result["n_pts_npc"] = classified_points
        result["n_pts_npc_terreno"] = terrain_points
        if classified_points == 0:
            result["status"] = "sem_pontos"
            result["mensagem"] = f"Nenhum ponto encontrado para MI_3={mi3}."
            for fp in (out_path_default, out_path_classified, out_path_terrain):
                try:
                    if fp and os.path.exists(fp):
                        os.remove(fp)
                except Exception:
                    pass
            result["completed_steps"] = completed_steps
            return result

        missing_outputs = [fp for fp in (out_path_default, out_path_classified, out_path_terrain) if not os.path.isfile(fp)]
        if missing_outputs:
            result["status"] = "erro"
            result["mensagem"] = "PDAL nao gerou os arquivos esperados: " + ", ".join(os.path.basename(m) for m in missing_outputs)
            result["completed_steps"] = completed_steps
            return result

        result["saida_filename"] = "; ".join([out_path_default, out_path_classified, out_path_terrain])
        result["generated_files"].extend([out_path_default, out_path_classified, out_path_terrain])
        log_message(
            f"✅ MI_3={mi3} nuvens geradas (NP={default_points}, NPc={classified_points}, NPc_T={terrain_points})",
            worker=worker_id,
        )

        if params.generate_surfaces:
            root_out = paths["base_np"]
            lote_dir = paths["lote_dir"]
            bloco_dir = paths["bloco_dir"]
            mds_cx, mds_cy = _resolve_cell_size(params.mds_cell_size)
            mdt_cx, mdt_cy = _resolve_cell_size(params.mdt_cell_size)

            def _log_phase(message: str, *, phase: str, level: str = "info") -> None:
                log_message(message, phase=phase, level=level, worker=worker_id)

            if subprogress_cb:
                subprogress_cb({"type": "substep", "phase": "MDS", "status": "start", "mi3": mi3, "worker_id": worker_id})
            try:
                mds_pipe = pdal.Pipeline(json.dumps({
                    "pipeline": [
                        {"type": "readers.las", "filename": out_path_classified},
                        {"type": "filters.range", "limits": "Classification![7:7]"}
                    ]
                }))
                mds_pipe.execute()
                mds_arrs = mds_pipe.arrays
                if mds_arrs:
                    base_name_mds = f"ES_L{str(lote_to_use).zfill(2)}_{bloco_to_use}_MDS_{mi3}_R0"
                    mds_txt = os.path.join(root_out, "6_MDS", lote_dir, bloco_dir, "2_ASCII", f"{base_name_mds}.txt")
                    mds_las = os.path.join(root_out, "6_MDS", lote_dir, bloco_dir, "1_LAS", f"{base_name_mds}.las")
                    mds_tif = os.path.join(root_out, "4_GeoTIFF", lote_dir, bloco_dir, f"{base_name_mds}.tif")
                    mds_ecw = os.path.join(root_out, "3_ECW", lote_dir, bloco_dir, f"{base_name_mds}.ecw")
                    generated_mds = generate_surface_products(
                        mds_arrs[0],
                        mds_cx,
                        mds_cy,
                        params.target_epsg,
                        mds_txt,
                        mds_las,
                        out_tif_path=mds_tif,
                        out_ecw_path=mds_ecw,
                        log=lambda msg, level="info": _log_phase(msg, phase="MDS", level=level),
                    )
                    if generated_mds:
                        result["generated_files"].extend(generated_mds)
                        _log_phase(f"✅ MI_3={mi3} MDS gerado ({mds_cx} x {mds_cy} m)", phase="MDS")
                else:
                    _log_phase("Aviso: Nenhum ponto disponível para MDS.", phase="MDS", level="warning")
            except Exception as exc:
                _log_phase(f"Aviso: falha ao gerar MDS ({exc})", phase="MDS", level="warning")
            finally:
                if subprogress_cb:
                    subprogress_cb({"type": "substep", "phase": "MDS", "status": "done", "mi3": mi3, "worker_id": worker_id})
                completed_steps += 1

            if subprogress_cb:
                subprogress_cb({"type": "substep", "phase": "MDT", "status": "start", "mi3": mi3, "worker_id": worker_id})
            try:
                mdt_pipe = pdal.Pipeline(json.dumps({
                    "pipeline": [
                        {"type": "readers.las", "filename": out_path_classified},
                        {"type": "filters.range", "limits": "Classification[2:2]"}
                    ]
                }))
                mdt_pipe.execute()
                mdt_arrs = mdt_pipe.arrays
                if mdt_arrs:
                    base_name_mdt = f"ES_L{str(lote_to_use).zfill(2)}_{bloco_to_use}_MDT_{mi3}_R0"
                    mdt_txt = os.path.join(root_out, "7_MDT", lote_dir, bloco_dir, "2_ASCII", f"{base_name_mdt}.txt")
                    mdt_las = os.path.join(root_out, "7_MDT", lote_dir, bloco_dir, "1_LAS", f"{base_name_mdt}.las")
                    mdt_tif = os.path.join(root_out, "4_GeoTIFF", lote_dir, bloco_dir, f"{base_name_mdt}.tif")
                    mdt_ecw = os.path.join(root_out, "3_ECW", lote_dir, bloco_dir, f"{base_name_mdt}.ecw")
                    generated_mdt = generate_surface_products(
                        mdt_arrs[0],
                        mdt_cx,
                        mdt_cy,
                        params.target_epsg,
                        mdt_txt,
                        mdt_las,
                        out_tif_path=mdt_tif,
                        out_ecw_path=mdt_ecw,
                        log=lambda msg, level="info": _log_phase(msg, phase="MDT", level=level),
                    )
                    if generated_mdt:
                        result["generated_files"].extend(generated_mdt)
                        _log_phase(f"✅ MI_3={mi3} MDT gerado ({mdt_cx} x {mdt_cy} m)", phase="MDT")
                else:
                    _log_phase("Aviso: Nenhum ponto disponível para MDT.", phase="MDT", level="warning")
            except Exception as exc:
                _log_phase(f"Aviso: falha ao gerar MDT ({exc})", phase="MDT", level="warning")
            finally:
                if subprogress_cb:
                    subprogress_cb({"type": "substep", "phase": "MDT", "status": "done", "mi3": mi3, "worker_id": worker_id})
                completed_steps += 1

    except Exception as e:
        result["status"] = "erro"
        result["mensagem"] = str(e)
    finally:
        result["duracao_s"] = round(time.time() - start, 3)
        result["completed_steps"] = completed_steps
    return result
def run_processing(
    params: AppParams,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    all_results: List[Dict[str, Any]] = []

    def log_message(msg: str, *, phase: Optional[str] = None, color: Optional[str] = None, level: str = "info", worker: Optional[int] = None) -> None:
        emit_log(msg, progress_cb=progress_cb, phase=phase, color=color, worker_id=worker, level=level)

    art = gpd.read_file(params.articulation_shp)
    art = ensure_crs(art, params.target_epsg)
    if "MI_3" not in art.columns:
        raise ValueError("Shapefile de articulação não contém coluna 'MI_3'.")
    if "LOTE" not in art.columns:
        raise ValueError("Shapefile de articulação não contém coluna 'LOTE'.")
    if "BLOCOS" not in art.columns:
        raise ValueError("Shapefile de articulação não contém coluna 'BLOCOS'.")

    log_message("=== COMBINAÇÕES LOTE/BLOCOS DISPONÍVEIS NO SHAPEFILE ===")
    combos = (art[["LOTE", "BLOCOS"]]
              .assign(LOTE=lambda d: d["LOTE"].map(norm_lote),
                      BLOCOS=lambda d: d["BLOCOS"].map(norm_bloco))
              .value_counts()
              .sort_index())
    combos = combos.iloc[0:0]
    for (lote, bloco), count in combos.items():
        log_message(f"LOTE={lote}, BLOCOS={bloco}: {count} polígonos")

    log_message(f"=== PROCESSANDO {len(params.blocks)} BLOCO(S) ===")
    for i, block in enumerate(params.blocks, 1):
        log_message(f"Bloco {i}: LOTE={block.lote}, BLOCOS={block.bloco_text}")
        log_message(f"  - Base NP: {block.output_folder_default}")
        log_message(f"  - Base NPc completo: {block.output_folder_classified}")
        log_message(f"  - Base NPc terreno: {block.output_folder_terrain}")

    if params.buffer_m and params.buffer_m != 0:
        art["geometry"] = art.geometry.buffer(params.buffer_m)

    tiles = scan_tiles(params.input_folder)
    tindex = TileIndex(tiles)

    for block_idx, block in enumerate(params.blocks, 1):
        log_message("")
        log_message(f"PROCESSANDO BLOCO {block_idx}/{len(params.blocks)}: LOTE={block.lote}, BLOCOS={block.bloco_text}")

        jobs: List[Dict[str, Any]] = []
        skipped_existing_count = 0
        total_match = 0
        intersect_count = 0

        log_message("Verificando arquivos existentes nas pastas de entrega...")

        for _, row in art.iterrows():
            mi3 = row.get("MI_3")
            if mi3 is None or (isinstance(mi3, float) and math.isnan(mi3)) or str(mi3).strip() == "":
                warnings.append("Polígono sem MI_3 válido. Pulado.")
                continue

            lote_from_shp, bloco_from_shp = get_block_from_articulation(row)

            if lote_from_shp and bloco_from_shp:
                lote_shape_norm = norm_lote(lote_from_shp)
                lote_input_norm = norm_lote(block.lote)
                bloco_shape_norm = norm_bloco(bloco_from_shp)
                bloco_input_norm = norm_bloco(block.bloco_text)

                if lote_shape_norm != lote_input_norm or bloco_shape_norm != bloco_input_norm:
                    warnings.append(f"Polígono MI_3={mi3} tem LOTE={lote_from_shp}, BLOCOS={bloco_from_shp} - não corresponde ao digitado ({block.lote}, {block.bloco_text}). Pulado.")
                    continue

            total_match += 1
            poly: Polygon = row.geometry
            minx, miny, maxx, maxy = poly.bounds
            buffer = 100.0
            buffered_bbox = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
            candidate_tiles = tindex.query(buffered_bbox)
            if candidate_tiles:
                intersect_count += 1

            lote_to_use = lote_from_shp if lote_from_shp else block.lote
            bloco_to_use = bloco_from_shp if bloco_from_shp else block.bloco_text
            paths = resolve_output_paths(block, lote_to_use, bloco_to_use)
            out_name_default = f"ES_L{str(lote_to_use).zfill(2)}_{bloco_to_use}_NP_{mi3}_R0"
            out_name_classified = f"ES_L{str(lote_to_use).zfill(2)}_{bloco_to_use}_NPC_{mi3}_R0"
            out_name_terrain = f"ES_L{str(lote_to_use).zfill(2)}_{bloco_to_use}_NPC_T_{mi3}_R0"

            if not params.overwrite:
                default_exists = check_existing_files(paths["np_dir"], out_name_default)
                classified_exists = check_existing_files(paths["npc_dir"], out_name_classified)
                terrain_exists = check_existing_files(paths["npc_terrain_dir"], out_name_terrain)

                if default_exists and classified_exists and terrain_exists:
                    skipped_existing_count += 1
                    log_message(f"⏭️ MI_3={mi3}: Arquivos já existem - pulando")
                    continue
                elif default_exists or classified_exists or terrain_exists:
                    faltantes: List[str] = []
                    if not default_exists:
                        faltantes.append("NP")
                    if not classified_exists:
                        faltantes.append("NPc")
                    if not terrain_exists:
                        faltantes.append("NPc_T")
                    log_message(f"⚠️ MI_3={mi3}: faltando {', '.join(faltantes)} - processando...")

            if candidate_tiles:
                jobs.append({
                    "params": params.__dict__,
                    "block_config": block.__dict__,
                    "poly_wkt": poly.wkt,
                    "mi3": str(mi3),
                    "lote_from_shp": lote_from_shp,
                    "bloco_from_shp": bloco_from_shp,
                    "area_inter": 0.0,
                    "tiles": [t.__dict__ for t in candidate_tiles]
                })

        results: List[Dict[str, Any]] = []
        total_jobs = len(jobs)
        log_message(f"Articulações que intersectam tiles para LOTE={block.lote}, BLOCOS={block.bloco_text}: {intersect_count} (de {total_match})")
        steps_per_job = 3 + (2 if params.generate_surfaces else 0)
        total_steps_expected = total_jobs * steps_per_job
        steps_completed_global = 0

        if progress_cb:
            progress_cb({
                "type": "block_start",
                "block_idx": block_idx,
                "total": total_jobs,
                "total_jobs": total_jobs,
                "steps_per_job": steps_per_job,
                "total_steps": total_steps_expected,
            })

        if skipped_existing_count > 0:
            log_message(f"⏭️ {skipped_existing_count} articulações puladas (arquivos já existem)")

        log_message(f"Processando {total_jobs} polígonos com {params.workers} workers...")

        if params.workers and params.workers > 1 and total_jobs > 0:
            with ThreadPoolExecutor(max_workers=params.workers) as ex:
                slot_lock = threading.Lock()
                slots = list(range(1, (params.workers or 1) + 1))

                def acquire_slot() -> int:
                    with slot_lock:
                        return slots.pop(0) if slots else 0

                def release_slot(s: int) -> None:
                    if s:
                        with slot_lock:
                            slots.append(s)

                def do_job(job: Dict[str, Any]):
                    wid = acquire_slot()
                    mi3_job = job.get("mi3")
                    try:
                        if cancel_event and cancel_event.is_set():
                            return {"status": "cancelado", "saida_filename": None, "generated_files": [], "worker_id": wid, "MI_3": mi3_job}
                        sub_cb = progress_cb if progress_cb else None
                        log_cb_local = progress_cb if progress_cb else None
                        return process_one_feature(job, subprogress_cb=sub_cb, log_cb=log_cb_local, cancel_event=cancel_event, worker_id=wid)
                    finally:
                        release_slot(wid)

                futs = [ex.submit(do_job, job) for job in jobs]
                completed = 0
                for fut in as_completed(futs):
                    try:
                        res = fut.result()
                    except Exception as exc:
                        res = {"status": "erro", "mensagem": str(exc), "saida_filename": None, "generated_files": [], "worker_id": None}
                    results.append(res)
                    completed += 1
                    step_inc = int(res.get("completed_steps") or 0)
                    steps_completed_global = min(total_steps_expected, steps_completed_global + step_inc)
                    if progress_cb:
                        progress_cb({
                            "type": "block_progress",
                            "block_idx": block_idx,
                            "completed": completed,
                            "total": total_jobs,
                            "steps_completed": steps_completed_global,
                            "total_steps": total_steps_expected,
                        })
                    status = res.get("status") if isinstance(res, dict) else None
                    wid = res.get("worker_id") if isinstance(res, dict) else None
                    if status == "sem_pontos":
                        log_message(f"MI_3={res.get('MI_3')} sem pontos no recorte (nenhum arquivo gerado)", worker=wid, level="warning")
                    elif status == "erro":
                        log_message(f"MI_3={res.get('MI_3')} erro: {res.get('mensagem')}", worker=wid, level="error")
                    elif status == "pulado_existente":
                        log_message(f"MI_3={res.get('MI_3')} pulado (arquivos já existem)", worker=wid)
                    elif status == "cancelado":
                        log_message(f"MI_3={res.get('MI_3')} cancelado pelo usuário", worker=wid, level="warning")
                    if cancel_event and cancel_event.is_set():
                        for fp in res.get("generated_files", []) or []:
                            try:
                                if fp and os.path.exists(fp):
                                    os.remove(fp)
                            except Exception:
                                pass
        else:
            for i, job in enumerate(jobs):
                mi3_job = job.get("mi3")
                if cancel_event and cancel_event.is_set():
                    res = {"status": "cancelado", "saida_filename": None, "generated_files": [], "worker_id": 1, "MI_3": mi3_job}
                else:
                    sub_cb = progress_cb if progress_cb else None
                    log_cb_local = progress_cb if progress_cb else None
                    res = process_one_feature(job, subprogress_cb=sub_cb, log_cb=log_cb_local, cancel_event=cancel_event, worker_id=1)
                results.append(res)
                step_inc = int(res.get("completed_steps") or 0)
                steps_completed_global = min(total_steps_expected, steps_completed_global + step_inc)
                if progress_cb:
                    progress_cb({
                        "type": "block_progress",
                        "block_idx": block_idx,
                        "completed": i + 1,
                        "total": total_jobs,
                        "steps_completed": steps_completed_global,
                        "total_steps": total_steps_expected,
                    })
                status = res.get("status") if isinstance(res, dict) else None
                wid = res.get("worker_id") if isinstance(res, dict) else None
                if status == "sem_pontos":
                    log_message(f"MI_3={res.get('MI_3')} sem pontos no recorte (nenhum arquivo gerado)", worker=wid, level="warning")
                elif status == "erro":
                    log_message(f"MI_3={res.get('MI_3')} erro: {res.get('mensagem')}", worker=wid, level="error")
                elif status == "pulado_existente":
                    log_message(f"MI_3={res.get('MI_3')} pulado (arquivos já existem)", worker=wid)
                elif status == "cancelado":
                    log_message(f"MI_3={res.get('MI_3')} cancelado pelo usuário", worker=wid, level="warning")
                if cancel_event and cancel_event is not None and cancel_event.is_set():
                    for fp in res.get("generated_files", []) or []:
                        try:
                            if fp and os.path.exists(fp):
                                os.remove(fp)
                        except Exception:
                            pass

        all_results.extend(results)
        log_message(f"✅ Bloco {block_idx} concluído: {len(results)} polígonos processados")
        if progress_cb:
            progress_cb({
                "type": "block_done",
                "block_idx": block_idx,
                "total": total_jobs,
                "steps_completed": min(steps_completed_global, total_steps_expected),
                "total_steps": total_steps_expected,
            })

    return all_results, warnings
def write_log_csv(output_folder: str, rows: List[Dict[str, Any]]) -> str:
    import csv
    log_path = os.path.join(output_folder, "log_processamento.csv")
    os.makedirs(output_folder, exist_ok=True)
    fieldnames = [
        "saida_filename", "MI_3", "lote", "bloco_digitado", "lote_do_shapefile", "bloco_do_shapefile",
        "n_tiles", "n_pts_np", "n_pts_npc", "n_pts_npc_terreno", "area_intersecao_m2", "epsg_usado", "data_hora", "duracao_s", "status", "mensagem"
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    return log_path
def write_summary(output_folder: str, results: List[Dict[str, Any]], warnings: List[str]) -> str:
    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped_existing = sum(1 for r in results if r.get("status") == "pulado_existente")
    skipped_no_tiles = sum(1 for r in results if r.get("status") == "sem_tiles")
    skipped_sem_pontos = sum(1 for r in results if r.get("status") == "sem_pontos")
    skipped_total = skipped_existing + skipped_no_tiles + skipped_sem_pontos
    errors = [r for r in results if r.get("status") == "erro"]
    total = len(results)

    # Count filtered polygons (those that didn't match LOTE/BLOCOS)
    filtered_count = len([w for w in warnings if "n�o corresponde ao digitado" in w])

    txt = [
        f"Total de pol�gonos processados: {total}",
        f"Sucesso: {ok}",
        f"Pulados: {skipped_total}",
        f"  - Pulados por arquivos j� existentes: {skipped_existing}",
        f"  - Pulados por sem tiles: {skipped_no_tiles}",
        f"  - Pulados sem pontos no recorte: {skipped_sem_pontos}",
        f"Pulados por filtro (sem corresponder ao lote/bloco): {filtered_count}",
        f"Erros: {len(errors)}",
        "",
        "Avisos:",
    ]
    txt.extend([f"- {w}" for w in warnings])
    if errors:
        txt.append("")
        txt.append("Erros detalhados:")
        for e in errors:
            txt.append(f"- {e.get('MI_3')}: {e.get('mensagem')}")
    path_out = os.path.join(output_folder, "relatorio_resumo.txt")
    with open(path_out, "w", encoding="utf-8") as f:
        f.write("\n".join(txt))
    return path_out

def _run_headless(params: AppParams) -> None:
    t0 = time.time()
    results, warns = run_processing(params)
    base_out = params.blocks[0].output_folder_default if params.blocks else params.input_folder
    log_csv = write_log_csv(base_out, results)
    summary = write_summary(base_out, results, warns)
    print(f"Concluído em {round(time.time()-t0,2)} s")
    print(f"Log: {log_csv}")
    print(f"Resumo: {summary}")
def build_gui_and_run():
    if _GUI_LIB == "psg":
        sg.theme("SystemDefault")
        
        # Aba 1: Recorte e exportação em .laz
        tab1_layout = [
            [sg.Text("Pasta de entrada (.laz 1 km²)"), sg.Input(key="in_folder"), sg.FolderBrowse("Browse")],
            [sg.Text("Shapefile de articulação (contém LOTE, BLOCOS, MI_3):"), sg.Input(key="art_shp", default_text=DEFAULT_ARTICULATION_SHP, size=(60,1)), sg.FileBrowse("Browse", file_types=(("Shapefile", "*.shp"),))],
            [sg.HSeparator()],
            [sg.Text("CONFIGURAÇÃO DO BLOCO", font=("Arial", 12, "bold"))],
            [sg.Text("Lote (XX) - filtra polígonos com este LOTE"), sg.Input(key="lote", size=(6,1)), sg.Text("Bloco (Y) - filtra polígonos com este BLOCOS"), sg.Input(key="bloco", size=(6,1))],
            [sg.Text("Pasta saída (cria subpastas NP e NPC)"), sg.Input(key="out_base"), sg.FolderBrowse("Browse")],
            [sg.HSeparator()],
            [sg.Text("CONFIGURAÇÕES GERAIS", font=("Arial", 12, "bold"))],
            [sg.Text("EPSG alvo"), sg.Input(key="epsg", size=(8,1), default_text="31982"),
             sg.Text("Buffer (m)"), sg.Input(key="buffer", size=(8,1), default_text="0"),
             sg.Checkbox("Sobrescrever", key="overwrite", default=False)],
            [sg.Checkbox("Gerar malhas regulares MDS/MDT", key="enable_surfaces", default=True)],
            [sg.Text("MDS passo X (m)"), sg.Input(key="mds_cell_x", size=(6,1), default_text="1.0"),
             sg.Text("Y"), sg.Input(key="mds_cell_y", size=(6,1), default_text="1.0"),
             sg.Text("MDT passo X (m)"), sg.Input(key="mdt_cell_x", size=(6,1), default_text="0.25"),
             sg.Text("Y"), sg.Input(key="mdt_cell_y", size=(6,1), default_text="0.25")],
            [sg.Text("Workers"), sg.Spin(values=[1,2,3,4,5,6,7,8], initial_value=min(4, os.cpu_count() or 1), key="workers"), sg.Checkbox("Processamento em paralelo", key="parallel", default=True)],
            [sg.Button("Executar", size=(15,2), key="EXEC"), sg.Button("Cancelar", size=(15,2), key="CANCEL"), sg.Button("Fechar", size=(15,2))],
            [sg.ProgressBar(max_value=100, orientation='h', size=(50, 20), key='PROG'), sg.Text("", key="PROG_TXT")],
            [sg.Multiline(size=(100,12), key="log", autoscroll=True)],
        ]
        # Aba 2: Conversão LAZ -> LAS 1.4
        tab2_layout = [
            [sg.Text("Pasta de entrada (.laz)"), sg.Input(key="conv_in_folder"), sg.FolderBrowse("Browse")],
            [sg.Text("Pasta de saída (.las 1.4)"), sg.Input(key="conv_out_folder"), sg.FolderBrowse("Browse")],
            [sg.Checkbox("Recursivo", key="conv_recursive", default=True),
             sg.Text("Workers"), sg.Spin(values=[1,2,3,4,5,6,7,8], initial_value=min(4, os.cpu_count() or 1), key="conv_workers")],
            [sg.Button("Converter", key="CONV_EXEC", size=(15,2))],
            [sg.ProgressBar(max_value=100, orientation='h', size=(50,20), key='CONV_PROG'), sg.Text("", key="CONV_PROG_TXT")],
            [sg.Multiline(size=(100,12), key="conv_log", autoscroll=True)],
        ]
        layout = [[
            sg.TabGroup([[sg.Tab('Exportar recorte (.laz)', tab1_layout), sg.Tab('Converter LAZ → LAS 1.4', tab2_layout)]])
        ]]
        win = sg.Window("Corte/Conversão LAZ/LAS (PDAL)", layout, finalize=True, resizable=True)
        
        # Estado de progresso (aba 1)
        current_block_total: int = 0
        current_block_completed: int = 0
        current_block_total_steps: int = 0
        current_block_steps_completed: int = 0
        cancel_evt: Optional[threading.Event] = None
        # Estado de progresso (aba 2)
        conv_total: int = 0
        conv_done: int = 0
        phase_colors = {
            "NP": "green",
            "NPC": "goldenrod",
            "MDS": "firebrick",
            "MDT": "royalblue",
        }
        def print_phase_log(prefix: str, phase: str, suffix: str) -> None:
            try:
                color = phase_colors.get(phase)
                win['log'].print(prefix, end='')
                win['log'].print(phase, text_color=color, end='')
                win['log'].print(suffix)
            except Exception:
                try:
                    win['log'].print(f"{prefix}{phase}{suffix}")
                except Exception:
                    pass
        def run_in_thread(params_local: AppParams, cancel_ev: Optional[threading.Event]) -> None:
            try:
                def progress_event(payload: Dict[str, Any]):
                    win.write_event_value('-PROGRESS-', payload)
                run_processing(params_local, progress_cb=progress_event, cancel_event=cancel_ev)
                win.write_event_value('-DONE-', None)
            except Exception as e:
                win.write_event_value('-ERROR-', str(e))
        # Conversão LAZ -> LAS 1.4
        def build_convert_pipeline(in_file: str, out_file: str) -> str:
            pipeline = [
                {"type": "readers.las", "filename": in_file},
                {"type": "writers.las", "filename": out_file, "minor_version": 4,
                 "scale_x": 0.01, "scale_y": 0.01, "scale_z": 0.01,
                 "offset_x": "auto", "offset_y": "auto", "offset_z": "auto"}
            ]
            return json.dumps({"pipeline": pipeline})
        def run_convert_thread(in_folder: str, out_folder: str, recursive: bool, workers: int) -> None:
            try:
                # Listar arquivos .laz
                laz_files: List[str] = []
                if recursive:
                    for root, _dirs, files in os.walk(in_folder):
                        for name in files:
                            if name.lower().endswith('.laz'):
                                laz_files.append(os.path.join(root, name))
                else:
                    for name in os.listdir(in_folder):
                        if name.lower().endswith('.laz'):
                            laz_files.append(os.path.join(in_folder, name))
                total = len(laz_files)
                win.write_event_value('-CONV-PROGRESS-', {"type": "start", "total": total})
                def convert_one(fp_in: str) -> Dict[str, Any]:
                    base = os.path.splitext(os.path.basename(fp_in))[0]
                    out_path = os.path.join(out_folder, base + '.las')
                    os.makedirs(out_folder, exist_ok=True)
                    pipe = pdal.Pipeline(build_convert_pipeline(fp_in, out_path))
                    pipe.execute()
                    return {"status": "ok", "out": out_path}
                results: List[Dict[str, Any]] = []
                if workers and workers > 1:
                    from concurrent.futures import ThreadPoolExecutor, as_completed as as_completed_thr
                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        futs = [ex.submit(convert_one, f) for f in laz_files]
                        done = 0
                        for fut in as_completed_thr(futs):
                            res = fut.result()
                            results.append(res)
                            done += 1
                            win.write_event_value('-CONV-PROGRESS-', {"type": "step", "done": done, "total": total, "file": res.get('out')})
                else:
                    done = 0
                    for f in laz_files:
                        res = convert_one(f)
                        results.append(res)
                        done += 1
                        win.write_event_value('-CONV-PROGRESS-', {"type": "step", "done": done, "total": total, "file": res.get('out')})
                win.write_event_value('-CONV-DONE-', {"results": results})
            except Exception as e:
                win.write_event_value('-CONV-ERROR-', str(e))
        while True:
            ev, values = win.read(timeout=150)
            if ev in (sg.WINDOW_CLOSED, "Fechar"):
                break
            elif ev == "EXEC":
                try:
                    # Create single block configuration
                    base_out = values.get("out_base") or ""
                    block = BlockConfig(
                        lote=str(values["lote"]).strip(),
                        bloco_text=str(values["bloco"]).strip(),
                        output_folder_default=base_out,
                        output_folder_classified=base_out,
                        output_folder_terrain=base_out,
                    )
                    enable_surfaces = bool(values.get("enable_surfaces"))
                    try:
                        mds_cell_x = float(values.get("mds_cell_x") or 1.0)
                        mds_cell_y = float(values.get("mds_cell_y") or 1.0)
                        mdt_cell_x = float(values.get("mdt_cell_x") or 0.25)
                        mdt_cell_y = float(values.get("mdt_cell_y") or 0.25)
                    except Exception:
                        sg.popup_error("Informe tamanhos de célula numéricos para MDS/MDT.")
                        continue
                    if any(v <= 0 for v in (mds_cell_x, mds_cell_y, mdt_cell_x, mdt_cell_y)):
                        sg.popup_error("Os tamanhos de célula devem ser positivos.")
                        continue
                    
                    params = AppParams(
                        input_folder=values["in_folder"],
                        blocks=[block],  # Single block for now
                        articulation_shp=values.get("art_shp") or DEFAULT_ARTICULATION_SHP,
                        target_epsg=int(values.get("epsg") or 31982),
                        buffer_m=float(values.get("buffer") or 0.0),
                        overwrite=bool(values.get("overwrite")),
                        workers=int(values.get("workers") or 1 if values.get("parallel") else 1),
                        convert_to_laz=False,
                        generate_surfaces=enable_surfaces,
                        mds_cell_size=(mds_cell_x, mds_cell_y),
                        mdt_cell_size=(mdt_cell_x, mdt_cell_y),
                    )
                    if not values.get("parallel"):
                        params.workers = 1
                    # iniciar processamento em background
                    win['EXEC'].update(disabled=True)
                    # desabilitar controles para evitar mudanças durante execução
                    for k in ("workers", "parallel", "in_folder", "art_shp", "lote", "bloco", "out_base",
                              "enable_surfaces", "mds_cell_x", "mds_cell_y", "mdt_cell_x", "mdt_cell_y"):
                        try:
                            win[k].update(disabled=True)
                        except Exception:
                            pass
                    current_block_total = 0
                    current_block_completed = 0
                    win['PROG'].update_bar(0)
                    win['PROG_TXT'].update("Iniciando...")
                    # habilitar botão Cancelar e iniciar thread com cancel_evt
                    try:
                        win['CANCEL'].update(disabled=False)
                    except Exception:
                        pass
                    cancel_evt = threading.Event()
                    t = threading.Thread(target=run_in_thread, args=(params, cancel_evt), daemon=True)
                    t.start()
                    continue
                    sg.popup("Processo concluído", keep_on_top=True)
                except Exception as e:
                    sg.popup_error(f"Erro: {e}")
            elif ev == 'CANCEL':
                try:
                    if cancel_evt is not None:
                        cancel_evt.set()
                        if 'CANCEL' in win.AllKeysDict:
                            win['CANCEL'].update(disabled=True)
                        win['PROG_TXT'].update("Cancelando... aguarde jobs em andamento")
                except Exception:
                    pass
            elif ev == '-PROGRESS-':
                payload = values.get('-PROGRESS-') or {}
                if payload.get('type') == 'block_start':
                    current_block_total = int(payload.get('total') or 0)
                    current_block_completed = 0
                    current_block_total_steps = int(payload.get('total_steps') or 0)
                    current_block_steps_completed = 0
                elif payload.get('type') == 'block_progress':
                    current_block_completed = int(payload.get('completed') or 0)
                    current_block_total = int(payload.get('total') or current_block_total)
                    if 'steps_completed' in payload:
                        current_block_steps_completed = int(payload.get('steps_completed') or 0)
                    if 'total_steps' in payload and payload.get('total_steps') is not None:
                        current_block_total_steps = int(payload.get('total_steps') or current_block_total_steps)
                elif payload.get('type') == 'block_done':
                    current_block_completed = current_block_total
                    if current_block_total_steps:
                        current_block_steps_completed = current_block_total_steps
                elif payload.get('type') == 'substep':
                    phase = payload.get('phase'); st = payload.get('status'); mi3 = payload.get('mi3'); wid = payload.get('worker_id')
                    try:
                        if st == 'start':
                            print_phase_log(f"Worker {wid}: iniciando ", phase, f" de MI_3={mi3}")
                        elif st == 'done':
                            print_phase_log(f"Worker {wid}: finalizou ", phase, f" de MI_3={mi3}")
                    except Exception:
                        pass
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
                    prefix = f"[W{worker}] " if worker else ''
                    try:
                        if color:
                            win['log'].print(prefix + message, text_color=color)
                        else:
                            win['log'].print(prefix + message)
                    except Exception:
                        try:
                            win['log'].print(prefix + message)
                        except Exception:
                            pass
                # update UI
                if current_block_total_steps:
                    total_steps = max(1, current_block_total_steps)
                    pct = int((current_block_steps_completed / total_steps) * 100)
                    display = f"{current_block_steps_completed}/{current_block_total_steps}"
                else:
                    total = max(1, current_block_total)
                    pct = int((current_block_completed / total) * 100) if total else 0
                    display = f"{current_block_completed}/{current_block_total}"
                win['PROG'].update_bar(pct)
                win['PROG_TXT'].update(f"{display} ({pct}%)")
            elif ev == '-DONE-':
                win['EXEC'].update(disabled=False)
                if 'CANCEL' in win.AllKeysDict:
                    win['CANCEL'].update(disabled=True)
                # reabilitar controles
                for k in ("workers", "parallel", "in_folder", "art_shp", "lote", "bloco", "out_base"):
                    try:
                        win[k].update(disabled=False)
                    except Exception:
                        pass
                sg.popup("Processo concluído", keep_on_top=True)
            elif ev == '-ERROR-':
                win['EXEC'].update(disabled=False)
                if 'CANCEL' in win.AllKeysDict:
                    win['CANCEL'].update(disabled=True)
                for k in ("workers", "parallel", "in_folder", "art_shp", "lote", "bloco", "out_base"):
                    try:
                        win[k].update(disabled=False)
                    except Exception:
                        pass
                sg.popup_error(f"Erro: {values.get('-ERROR-')}")
            elif ev == 'CONV_EXEC':
                try:
                    in_folder = values.get('conv_in_folder')
                    out_folder = values.get('conv_out_folder')
                    recursive = bool(values.get('conv_recursive'))
                    workers = int(values.get('conv_workers') or 1)
                    if not in_folder or not out_folder:
                        sg.popup_error('Informe pasta de entrada e saída')
                        continue
                    # reset conv progress UI
                    conv_total = 0
                    conv_done = 0
                    win['CONV_PROG'].update_bar(0)
                    win['CONV_PROG_TXT'].update('Iniciando...')
                    win['conv_log'].update("")
                    t = threading.Thread(target=run_convert_thread, args=(in_folder, out_folder, recursive, workers), daemon=True)
                    t.start()
                except Exception as e:
                    sg.popup_error(f"Erro: {e}")
            elif ev == '-CONV-PROGRESS-':
                payload = values.get('-CONV-PROGRESS-') or {}
                if payload.get('type') == 'start':
                    conv_total = int(payload.get('total') or 0)
                    conv_done = 0
                elif payload.get('type') == 'step':
                    conv_done = int(payload.get('done') or 0)
                    conv_total = int(payload.get('total') or conv_total)
                    file_done = payload.get('file')
                    if file_done:
                        win['conv_log'].print(f"{os.path.basename(file_done)} gerado")
                total = max(1, conv_total)
                pct = int((conv_done / total) * 100)
                win['CONV_PROG'].update_bar(pct)
                win['CONV_PROG_TXT'].update(f"{conv_done}/{conv_total} ({pct}%)")
            elif ev == '-CONV-DONE-':
                sg.popup('Conversão concluída', keep_on_top=True)
            elif ev == '-CONV-ERROR-':
                sg.popup_error(f"Erro na conversão: {values.get('-CONV-ERROR-')}")
        win.close()
    else:
        # Minimal Tk fallback
        root = tk.Tk()
        root.title("Corte LAZ → LAS por articulação (PDAL)")
        messagebox.showinfo("Info", "PySimpleGUI não disponível. Usando interface mínima Tkinter.")
        # Ask via dialogs
        in_folder = filedialog.askdirectory(title="Pasta de entrada (.laz)")
        art = DEFAULT_ARTICULATION_SHP
        out_base = filedialog.askdirectory(title="Pasta saída (base)") or ""
        lote = ""; bloco = ""
        try:
            lote = input("Lote (XX): ")
            bloco = input("Bloco (Y): ")
        except Exception:
            pass

        block = BlockConfig(
            lote=str(lote).strip(),
            bloco_text=str(bloco).strip(),
            output_folder_default=out_base,
            output_folder_classified=out_base,
            output_folder_terrain=out_base,
        )

        params = AppParams(
            input_folder=in_folder,
            blocks=[block],
            articulation_shp=art,
        )
        _run_headless(params)
        root.destroy()
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--nogui":
        # CLI mode for automation
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--in", dest="input_folder", required=True)
        ap.add_argument("--art", dest="articulation_shp", default=DEFAULT_ARTICULATION_SHP)
        ap.add_argument("--out-default", dest="output_folder_default", required=True)
        ap.add_argument("--out-classified", dest="output_folder_classified", required=True)
        ap.add_argument("--out-terrain", dest="output_folder_terrain", help="Pasta base para NPc_T (opcional)", default=None)
        ap.add_argument("--lote", required=True)
        ap.add_argument("--bloco", required=True)
        ap.add_argument("--epsg", type=int, default=31982)
        ap.add_argument("--buffer", type=float, default=0.0)
        ap.add_argument("--overwrite", action="store_true")
        ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
        ap.add_argument("--convert-to-laz", action="store_true", default=False, help="Convert LAS to LAZ after processing")
        ap.add_argument("--no-surfaces", action="store_true", help="Não gerar malhas MDS/MDT")
        ap.add_argument("--mds-cell", default="1.0,1.0", help="Passo da malha MDS no formato dx,dy (m)")
        ap.add_argument("--mdt-cell", default="0.25,0.25", help="Passo da malha MDT no formato dx,dy (m)")
        args = ap.parse_args()
        def _parse_cell_arg(value: str, fallback: Tuple[float, float]) -> Tuple[float, float]:
            try:
                parts = [float(p.strip()) for p in str(value).split(",") if p.strip() != ""]
            except Exception as exc:
                raise SystemExit(f"Erro ao interpretar tamanho de célula '{value}': {exc}") from exc
            if not parts:
                return fallback
            if len(parts) == 1:
                parts *= 2
            if len(parts) != 2:
                raise SystemExit(f"Informe valores de célula no formato dx,dy (recebido: '{value}')")
            if parts[0] <= 0 or parts[1] <= 0:
                raise SystemExit(f"Tamanhos de célula devem ser positivos (recebido: '{value}')")
            return (parts[0], parts[1])

        mds_cell = _parse_cell_arg(args.mds_cell, (1.0, 1.0))
        mdt_cell = _parse_cell_arg(args.mdt_cell, (0.25, 0.25))
        terrain_base = args.output_folder_terrain or args.output_folder_classified
        block = BlockConfig(
            lote=args.lote,
            bloco_text=args.bloco,
            output_folder_default=args.output_folder_default,
            output_folder_classified=args.output_folder_classified,
            output_folder_terrain=terrain_base,
        )
        params = AppParams(
            input_folder=args.input_folder,
            blocks=[block],
            articulation_shp=args.articulation_shp,
            target_epsg=args.epsg,
            buffer_m=args.buffer,
            overwrite=args.overwrite,
            workers=args.workers,
            convert_to_laz=args.convert_to_laz,
            generate_surfaces=not args.no_surfaces,
            mds_cell_size=mds_cell,
            mdt_cell_size=mdt_cell,
        )
        _run_headless(params)
    else:
        build_gui_and_run()
