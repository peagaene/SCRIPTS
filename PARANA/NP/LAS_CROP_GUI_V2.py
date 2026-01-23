import os
import sys
import json
import time
import math
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable
import threading

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

from concurrent.futures import ProcessPoolExecutor, as_completed


# Caminho fixo do shapefile de articulação (pode ser editado aqui se necessário)
DEFAULT_ARTICULATION_SHP = r"\\192.168.2.28\i\80225_PROJETO_IAT_PARANA\5 Processamento Laser\Articulacao_Auxiliar_5000+50.shp"

# Caminho do LASzip para conversão LAS -> LAZ
LASZIP_PATH = r'D:\LAStools\bin\laszip.exe'


@dataclass
class BlockConfig:
    """Configuration for a single block processing."""
    lote: str
    bloco_text: str
    output_folder_default: str  # Para pontos default (classe 1)
    output_folder_classified: str  # Para pontos classificados

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


@dataclass
class TileInfo:
    path: str
    bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy


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


def build_pdal_crop_pipeline(input_files: List[str], crop_polygon: Polygon, out_file: str, change_to_class: Optional[int] = None) -> str:
    readers = [{"type": "readers.las", "filename": f} for f in input_files]
    filters = [
        {"type": "filters.merge"},
        {"type": "filters.crop", "polygon": crop_polygon.wkt}
    ]
    if change_to_class is not None:
        filters.append({"type": "filters.assign", "assignment": f"Classification[:]={change_to_class}"})
    
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


def check_existing_files(output_folder: str, filename_base: str) -> bool:
    """Check if LAS or LAZ files already exist for this articulation."""
    las_file = os.path.join(output_folder, f"{filename_base}.las")
    laz_file = os.path.join(output_folder, f"{filename_base}.laz")
    
    return os.path.exists(las_file) or os.path.exists(laz_file)

def process_one_feature(feat: Dict[str, Any]) -> Dict[str, Any]:
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
    
    # More efficient tile selection - only process tiles that actually intersect
    selected_files = []
    for t in tiles:
        tminx, tminy, tmaxx, tmaxy = t.bounds
        # Check if tile intersects with polygon bbox
        if not (tmaxx < bbox[0] or tmaxy < bbox[1] or tminx > bbox[2] or tminy > bbox[3]):
            selected_files.append(t.path)

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
        "mensagem": ""
    }

    if not selected_files:
        result["status"] = "sem_tiles"
        result["mensagem"] = "Nenhum tile intersecta o polígono"
        return result

    # Use lote and bloco from shapefile if available, otherwise use input values
    lote_to_use = lote_from_shp if lote_from_shp else block_config.lote
    bloco_to_use = bloco_from_shp if bloco_from_shp else block_config.bloco_text
    
    # Create both output files (LAZ format)
    out_name_default = f"ES_L{str(lote_to_use).zfill(2)}_{bloco_to_use}_NP_{mi3}_R0"
    out_name_classified = f"ES_L{str(lote_to_use).zfill(2)}_{bloco_to_use}_NPC_{mi3}_R0"
    
    out_path_default = os.path.join(block_config.output_folder_default, f"{out_name_default}.laz")
    out_path_classified = os.path.join(block_config.output_folder_classified, f"{out_name_classified}.laz")

    # Check if files already exist (LAS or LAZ) and skip if not overwriting
    if not params.overwrite:
        # Check default file
        default_exists = check_existing_files(block_config.output_folder_default, out_name_default)
        # Check classified file  
        classified_exists = check_existing_files(block_config.output_folder_classified, out_name_classified)
        
        if default_exists and classified_exists:
            result["status"] = "pulado_existente"
            result["saida_filename"] = f"{out_name_default}.las/.laz; {out_name_classified}.las/.laz"
            result["mensagem"] = "Arquivos já existem nas pastas de entrega"
            print(f"⏭️ MI_3={mi3}: Arquivos já existem - pulando processamento")
            return result
        elif default_exists or classified_exists:
            # Log partial existence for debugging
            missing_files = []
            if not default_exists:
                missing_files.append("default")
            if not classified_exists:
                missing_files.append("classified")
            print(f"⚠️ MI_3={mi3}: Apenas alguns arquivos existem (faltam: {', '.join(missing_files)}) - processando...")

    try:
        os.makedirs(block_config.output_folder_default, exist_ok=True)
        os.makedirs(block_config.output_folder_classified, exist_ok=True)
        
        # Export default points (all points changed to class 1) -> LAZ
        pipe_json_default = build_pdal_crop_pipeline(selected_files, poly, out_path_default, change_to_class=1)
        pipe_default = pdal.Pipeline(pipe_json_default)
        pipe_default.execute()
        
        # Export all classified points (keep original classifications) -> LAZ
        pipe_json_classified = build_pdal_crop_pipeline(selected_files, poly, out_path_classified)
        pipe_classified = pdal.Pipeline(pipe_json_classified)
        pipe_classified.execute()
        
        result["saida_filename"] = f"{out_path_default}; {out_path_classified}"
            
    except Exception as e:
        result["status"] = "erro"
        result["mensagem"] = str(e)
    finally:
        result["duracao_s"] = round(time.time() - start, 3)
    return result


def run_processing(
    params: AppParams,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    all_results: List[Dict[str, Any]] = []

    # Load articulation shapefile only
    art = gpd.read_file(params.articulation_shp)
    art = ensure_crs(art, params.target_epsg)

    if "MI_3" not in art.columns:
        raise ValueError("Shapefile de articulação não contém coluna 'MI_3'.")
    if "LOTE" not in art.columns:
        raise ValueError("Shapefile de articulação não contém coluna 'LOTE'.")
    if "BLOCOS" not in art.columns:
        raise ValueError("Shapefile de articulação não contém coluna 'BLOCOS'.")

    # Show available LOTE/BLOCOS combinations for diagnosis
    print("=== COMBINAÇÕES LOTE/BLOCOS DISPONÍVEIS NO SHAPEFILE ===")
    combos = (art[['LOTE','BLOCOS']]
              .assign(LOTE=lambda d: d['LOTE'].map(norm_lote),
                      BLOCOS=lambda d: d['BLOCOS'].map(norm_bloco))
              .value_counts()
              .sort_index())
    combos = combos.iloc[0:0]
    for (lote, bloco), count in combos.items():
        print(f"LOTE={lote}, BLOCOS={bloco}: {count} polígonos")
    
    print(f"=== PROCESSANDO {len(params.blocks)} BLOCO(S) ===")
    for i, block in enumerate(params.blocks, 1):
        print(f"Bloco {i}: LOTE={block.lote}, BLOCOS={block.bloco_text}")
        print(f"  - Saída default: {block.output_folder_default}")
        print(f"  - Saída classificados: {block.output_folder_classified}")

    # Buffer if needed
    if params.buffer_m and params.buffer_m != 0:
        art["geometry"] = art.geometry.buffer(params.buffer_m)

    # Scan tiles and build index
    tiles = scan_tiles(params.input_folder)
    tindex = TileIndex(tiles)

    # Process each block
    for block_idx, block in enumerate(params.blocks, 1):
        print(f"\n🔄 PROCESSANDO BLOCO {block_idx}/{len(params.blocks)}: LOTE={block.lote}, BLOCOS={block.bloco_text}")
        
        # Prepare jobs - filter by matching LOTE and BLOCOS for this block
        jobs: List[Dict[str, Any]] = []
        skipped_existing_count = 0
        total_match = 0
        intersect_count = 0
        
        print("🔍 Verificando arquivos existentes nas pastas de entrega...")
        
        for _, row in art.iterrows():
            mi3 = row.get("MI_3")
            if mi3 is None or (isinstance(mi3, float) and math.isnan(mi3)) or str(mi3).strip() == "":
                warnings.append("Polígono sem MI_3 válido. Pulado.")
                continue
            
            # Get lote and bloco directly from articulation shapefile
            lote_from_shp, bloco_from_shp = get_block_from_articulation(row)
            
            
            
            # Filter: only process polygons that match the input LOTE and BLOCOS
            if lote_from_shp and bloco_from_shp:
                # Normalize values for comparison using robust functions
                lote_shape_norm = norm_lote(lote_from_shp)
                lote_input_norm = norm_lote(block.lote)
                bloco_shape_norm = norm_bloco(bloco_from_shp)
                bloco_input_norm = norm_bloco(block.bloco_text)
                
                # Compare normalized values
                lote_match = lote_shape_norm == lote_input_norm
                bloco_match = bloco_shape_norm == bloco_input_norm
                
                if not lote_match or not bloco_match:
                    warnings.append(f"Polígono MI_3={mi3} tem LOTE={lote_from_shp}, BLOCOS={bloco_from_shp} - não corresponde ao digitado ({block.lote}, {block.bloco_text}). Pulado.")
                    continue
            
            # Contagem de correspondência e interseção com tiles (antes de checar arquivos existentes)
            total_match += 1
            poly: Polygon = row.geometry
            minx, miny, maxx, maxy = poly.bounds
            buffer = 100.0  # 100m buffer for tile selection
            buffered_bbox = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
            candidate_tiles = tindex.query(buffered_bbox)
            if candidate_tiles:
                intersect_count += 1

            # Check if files already exist before processing
            lote_to_use = lote_from_shp if lote_from_shp else block.lote
            bloco_to_use = bloco_from_shp if bloco_from_shp else block.bloco_text
            out_name_default = f"ES_L{str(lote_to_use).zfill(2)}_{bloco_to_use}_NP_{mi3}_R0"
            out_name_classified = f"ES_L{str(lote_to_use).zfill(2)}_{bloco_to_use}_NPC_{mi3}_R0"
            
            if not params.overwrite:
                default_exists = check_existing_files(block.output_folder_default, out_name_default)
                classified_exists = check_existing_files(block.output_folder_classified, out_name_classified)
                
                if default_exists and classified_exists:
                    skipped_existing_count += 1
                    print(f"⏭️ MI_3={mi3}: Arquivos já existem - pulando")
                    continue
            
            jobs.append({
                "params": params.__dict__,
                "block_config": block.__dict__,
                "poly_wkt": poly.wkt,
                "mi3": str(mi3),
                "lote_from_shp": lote_from_shp,
                "bloco_from_shp": bloco_from_shp,
                "area_inter": 0.0,  # Not used anymore
                "tiles": [t.__dict__ for t in candidate_tiles]
            })

        results: List[Dict[str, Any]] = []
        total_jobs = len(jobs)
        # Resumo: quantas articulações intersectam os tiles carregados para o LOTE/BLOCO digitado
        print(f"Articulações que intersectam tiles para LOTE={block.lote}, BLOCOS={block.bloco_text}: {intersect_count} (de {total_match})")
        if progress_cb:
            progress_cb({
                "type": "block_start",
                "block_idx": block_idx,
                "total": total_jobs,
            })
        
        if skipped_existing_count > 0:
            print(f"⏭️ {skipped_existing_count} articulações puladas (arquivos já existem)")
        
        print(f"Processando {total_jobs} polígonos com {params.workers} workers...")
        
        if params.workers and params.workers > 1:
            with ProcessPoolExecutor(max_workers=params.workers) as ex:
                futs = [ex.submit(process_one_feature, j) for j in jobs]
                completed = 0
                for fut in as_completed(futs):
                    res = fut.result()
                    results.append(res)
                    completed += 1
                    if progress_cb:
                        progress_cb({
                            "type": "block_progress",
                            "block_idx": block_idx,
                            "completed": completed,
                            "total": total_jobs,
                        })
                    # Log de arquivo exportado (substitui linhas 'Progresso: ...')
                    try:
                        if isinstance(res, dict) and res.get("status") == "ok" and res.get("saida_filename"):
                            for part in str(res.get("saida_filename")).split(';'):
                                p = part.strip()
                                if p:
                                    print(f"{os.path.basename(p)} exportado")
                    except Exception:
                        pass
        else:
            for i, j in enumerate(jobs):
                res = process_one_feature(j)
                results.append(res)
                if progress_cb:
                    progress_cb({
                        "type": "block_progress",
                        "block_idx": block_idx,
                        "completed": i + 1,
                        "total": total_jobs,
                    })
                # Log de arquivo exportado (substitui linhas 'Progresso: ...')
                try:
                    if isinstance(res, dict) and res.get("status") == "ok" and res.get("saida_filename"):
                        for part in str(res.get("saida_filename")).split(';'):
                            p = part.strip()
                            if p:
                                print(f"{os.path.basename(p)} exportado")
                except Exception:
                    pass

        # Add results to overall results
        all_results.extend(results)
        print(f"✅ Bloco {block_idx} concluído: {len(results)} polígonos processados")

    return all_results, warnings


def write_log_csv(output_folder: str, rows: List[Dict[str, Any]]) -> str:
    import csv
    log_path = os.path.join(output_folder, "log_processamento.csv")
    os.makedirs(output_folder, exist_ok=True)
    fieldnames = [
        "saida_filename", "MI_3", "lote", "bloco_digitado", "lote_do_shapefile", "bloco_do_shapefile",
        "n_tiles", "area_intersecao_m2", "epsg_usado", "data_hora", "duracao_s", "status", "mensagem"
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
    skipped_total = skipped_existing + skipped_no_tiles
    errors = [r for r in results if r.get("status") == "erro"]
    total = len(results)
    
    # Count filtered polygons (those that didn't match LOTE/BLOCOS)
    filtered_count = len([w for w in warnings if "não corresponde ao digitado" in w])
    
    txt = [
        f"Total de polígonos processados: {total}",
        f"Sucesso: {ok}",
        f"Pulados: {skipped_total}",
        f"  - Pulados por arquivos já existentes: {skipped_existing}",
        f"  - Pulados por sem tiles: {skipped_no_tiles}",
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
    path = os.path.join(output_folder, "relatorio_resumo.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt))
    return path


def _run_headless(params: AppParams) -> None:
    t0 = time.time()
    results, warns = run_processing(params)
    # Use output_folder_default for logs and summary
    log_csv = write_log_csv(params.output_folder_default, results)
    summary = write_summary(params.output_folder_default, results, warns)
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
            [sg.Text("Pasta saída - pontos default (classe 1) - arquivos .laz"), sg.Input(key="out_folder_default"), sg.FolderBrowse("Browse")],
            [sg.Text("Pasta saída - pontos classificados - arquivos .laz"), sg.Input(key="out_folder_classified"), sg.FolderBrowse("Browse")],
            [sg.HSeparator()],
            [sg.Text("CONFIGURAÇÕES GERAIS", font=("Arial", 12, "bold"))],
            [sg.Text("EPSG alvo"), sg.Input(key="epsg", size=(8,1), default_text="31982"),
             sg.Text("Buffer (m)"), sg.Input(key="buffer", size=(8,1), default_text="0"),
             sg.Checkbox("Sobrescrever", key="overwrite", default=False)],
            [sg.Text("Workers"), sg.Spin(values=[1,2,3,4,5,6,7,8], initial_value=min(4, os.cpu_count() or 1), key="workers"), sg.Checkbox("Processamento em paralelo", key="parallel", default=True)],
            [sg.Button("Executar", size=(15,2), key="EXEC"), sg.Button("Fechar", size=(15,2))],
            [sg.ProgressBar(max_value=100, orientation='h', size=(50, 20), key='PROG'), sg.Text("", key="PROG_TXT")],
            [sg.Multiline(size=(100,12), key="log", autoscroll=True, reroute_stdout=True, reroute_stderr=True)],
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

        # Estado de progresso (aba 2)
        conv_total: int = 0
        conv_done: int = 0

        def run_in_thread(params_local: AppParams) -> None:
            try:
                def progress_event(payload: Dict[str, Any]):
                    win.write_event_value('-PROGRESS-', payload)
                run_processing(params_local, progress_cb=progress_event)
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
                    block = BlockConfig(
                        lote=str(values["lote"]).strip(),
                        bloco_text=str(values["bloco"]).strip(),
                        output_folder_default=values["out_folder_default"],
                        output_folder_classified=values["out_folder_classified"]
                    )
                    
                    params = AppParams(
                        input_folder=values["in_folder"],
                        blocks=[block],  # Single block for now
                        articulation_shp=values.get("art_shp") or DEFAULT_ARTICULATION_SHP,
                        target_epsg=int(values.get("epsg") or 31982),
                        buffer_m=float(values.get("buffer") or 0.0),
                        overwrite=bool(values.get("overwrite")),
                        workers=int(values.get("workers") or 1 if values.get("parallel") else 1),
                        convert_to_laz=False,
                    )
                    if not values.get("parallel"):
                        params.workers = 1
                    # iniciar processamento em background
                    win['EXEC'].update(disabled=True)
                    current_block_total = 0
                    current_block_completed = 0
                    win['PROG'].update_bar(0)
                    win['PROG_TXT'].update("Iniciando...")
                    t = threading.Thread(target=run_in_thread, args=(params,), daemon=True)
                    t.start()
                    continue
                    sg.popup("Processo concluído", keep_on_top=True)
                except Exception as e:
                    sg.popup_error(f"Erro: {e}")
            elif ev == '-PROGRESS-':
                payload = values.get('-PROGRESS-') or {}
                if payload.get('type') == 'block_start':
                    current_block_total = int(payload.get('total') or 0)
                    current_block_completed = 0
                elif payload.get('type') == 'block_progress':
                    current_block_completed = int(payload.get('completed') or 0)
                    current_block_total = int(payload.get('total') or current_block_total)
                elif payload.get('type') == 'block_done':
                    current_block_completed = current_block_total
                # update UI
                total = max(1, current_block_total)
                pct = int((current_block_completed / total) * 100)
                win['PROG'].update_bar(pct)
                win['PROG_TXT'].update(f"{current_block_completed}/{current_block_total} ({pct}%)")
            elif ev == '-DONE-':
                win['EXEC'].update(disabled=False)
                sg.popup("Processo concluído", keep_on_top=True)
            elif ev == '-ERROR-':
                win['EXEC'].update(disabled=False)
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
        out_folder_default = filedialog.askdirectory(title="Pasta saída - pontos default (classe 1)")
        out_folder_classified = filedialog.askdirectory(title="Pasta saída - pontos classificados")
        lote = ""; bloco = ""
        try:
            lote = input("Lote (XX): ")
            bloco = input("Bloco (Y): ")
        except Exception:
            pass
        params = AppParams(
            input_folder=in_folder, articulation_shp=art,
            lote=lote, bloco_text=bloco, 
            output_folder_default=out_folder_default,
            output_folder_classified=out_folder_classified
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
        ap.add_argument("--lote", required=True)
        ap.add_argument("--bloco", required=True)
        ap.add_argument("--epsg", type=int, default=31982)
        ap.add_argument("--buffer", type=float, default=0.0)
        ap.add_argument("--overwrite", action="store_true")
        ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
        ap.add_argument("--convert-to-laz", action="store_true", default=False, help="Convert LAS to LAZ after processing")
        args = ap.parse_args()
        params = AppParams(
            input_folder=args.input_folder,
            articulation_shp=args.articulation_shp,
            lote=args.lote,
            bloco_text=args.bloco,
            output_folder_default=args.output_folder_default,
            output_folder_classified=args.output_folder_classified,
            target_epsg=args.epsg,
            buffer_m=args.buffer,
            overwrite=args.overwrite,
            workers=args.workers,
            convert_to_laz=args.convert_to_laz,
        )
        _run_headless(params)
    else:
        build_gui_and_run()


