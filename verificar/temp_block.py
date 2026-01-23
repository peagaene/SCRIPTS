def process_one_feature(
    feat: Dict[str, Any],
    subprogress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
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
        "mensagem": "",
        "generated_files": []
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
        if subprogress_cb:
            subprogress_cb({"type": "substep", "phase": "NP", "status": "start", "mi3": mi3, "worker_id": worker_id})
        pipe_json_default = build_pdal_crop_pipeline(selected_files, poly, out_path_default, change_to_class=1)
        pipe_default = pdal.Pipeline(pipe_json_default)
        pipe_default.execute()
        if subprogress_cb:
            subprogress_cb({"type": "substep", "phase": "NP", "status": "done", "mi3": mi3, "worker_id": worker_id})
        
        # Export all classified points (keep original classifications) -> LAZ
        if subprogress_cb:
            subprogress_cb({"type": "substep", "phase": "NPC", "status": "start", "mi3": mi3, "worker_id": worker_id})
        pipe_json_classified = build_pdal_crop_pipeline(selected_files, poly, out_path_classified)
        pipe_classified = pdal.Pipeline(pipe_json_classified)
        pipe_classified.execute()
        if subprogress_cb:
            subprogress_cb({"type": "substep", "phase": "NPC", "status": "done", "mi3": mi3, "worker_id": worker_id})
        
        result["saida_filename"] = f"{out_path_default}; {out_path_classified}"
        result["generated_files"].extend([out_path_default, out_path_classified])
        # ===== MDS/MDT grid generation (triangulated Z) =====
        # Base root (parent of NP/NPC)
        root_out = os.path.dirname(block_config.output_folder_default)
        def triangulated_grid_to_txt_laz(points_arr, cell_size: float, out_txt_path: str, out_laz_path: str):
            import numpy as np
            if points_arr.size == 0:
                return False
            xs = points_arr['X'].astype(float)
            ys = points_arr['Y'].astype(float)
            zs = points_arr['Z'].astype(float)
            if xs.size < 3 or not _HAS_SCIPY:
                # Fallback: nearest-neighbor if não há SciPy/triângulos suficientes
                try:
                    from sklearn.neighbors import KDTree  # optional fallback
                    tree = KDTree(np.c_[xs, ys])
                    ok_nn = True
                except Exception:
                    ok_nn = False
                import math
                minx, maxx = float(np.min(xs)), float(np.max(xs))
                miny, maxy = float(np.min(ys)), float(np.max(ys))
                gx = np.arange(minx, maxx + cell_size*0.5, cell_size)
                gy = np.arange(miny, maxy + cell_size*0.5, cell_size)
                out_pts = []
                for y in gy:
                    for x in gx:
                        if ok_nn:
                            dist, ind = tree.query([[x, y]], k=1)
                            z = float(zs[ind[0][0]])
                        else:
                            # No fallback available; skip
                            continue
                        out_pts.append((x, y, z))
            else:
                # Triangulated interpolation via Delaunay
                import numpy as np
                tri = Delaunay(np.c_[xs, ys])
                minx, maxx = float(np.min(xs)), float(np.max(xs))
                miny, maxy = float(np.min(ys)), float(np.max(ys))
                gx = np.arange(minx, maxx + cell_size*0.5, cell_size)
                gy = np.arange(miny, maxy + cell_size*0.5, cell_size)
                out_pts = []
                # Precompute affine transforms
                for y in gy:
                    for x in gx:
                        p = np.array([x, y])
                        simplex = tri.find_simplex(p)
                        if simplex == -1:
                            continue
                        T = tri.transform[simplex]
                        r = T[:2].dot(p - T[2])
                        b0, b1 = r[0], r[1]
                        b2 = 1.0 - b0 - b1
                        verts = tri.simplices[simplex]
                        z = b0 * zs[verts[0]] + b1 * zs[verts[1]] + b2 * zs[verts[2]]
                        out_pts.append((x, y, float(z)))
            # Write TXT
            with open(out_txt_path, 'w', encoding='utf-8') as ftxt:
                ftxt.write("X Y Z\n")
                for x, y, z in out_pts:
                    ftxt.write(f"{x:.3f} {y:.3f} {z:.3f}\n")
            # TXT -> LAZ via PDAL
            pipe_json = {
                "pipeline": [
                    {"type": "readers.text", "filename": out_txt_path, "header": "X Y Z"},
                    {"type": "writers.las", "filename": out_laz_path,
                     "scale_x": 0.01, "scale_y": 0.01, "scale_z": 0.01,
                     "offset_x": "auto", "offset_y": "auto", "offset_z": "auto"}
                ]
            }
            pipe = pdal.Pipeline(json.dumps(pipe_json))
            pipe.execute()
            return True
        # Read classified crop to arrays once per grid
        try:
            # MDS: classes != 7, cell 1.0m, triangulated Z
            if subprogress_cb:
                subprogress_cb({"type": "substep", "phase": "MDS", "status": "start", "mi3": mi3, "worker_id": worker_id})
            mds_pipe = pdal.Pipeline(json.dumps({
                "pipeline": [
                    {"type": "readers.las", "filename": out_path_classified},
                    {"type": "filters.range", "limits": "Classification![7:7]"}
                ]
            }))
            mds_pipe.execute()
            mds_arrs = mds_pipe.arrays
            if mds_arrs:
                # Diretorios: 6_MDS/LOTE_xx/BLOCO_Y/{1_LAS,2_ASCII}
                lote_dir = f"LOTE_{str(lote_to_use).zfill(2)}"
                bloco_dir = f"BLOCO_{bloco_to_use}"
                mds_las_dir = os.path.join(root_out, "6_MDS", lote_dir, bloco_dir, "1_LAS")
                mds_txt_dir = os.path.join(root_out, "6_MDS", lote_dir, bloco_dir, "2_ASCII")
                os.makedirs(mds_las_dir, exist_ok=True)
                os.makedirs(mds_txt_dir, exist_ok=True)
                name_base = f"ES_L{str(lote_to_use).zfill(2)}_{bloco_to_use}_MDS_{mi3}_R0"
                mds_txt = os.path.join(mds_txt_dir, f"{name_base}.txt")
                mds_laz = os.path.join(mds_las_dir, f"{name_base}.laz")
                if triangulated_grid_to_txt_laz(mds_arrs[0], 1.0, mds_txt, mds_laz):
                    result["generated_files"].extend([mds_txt, mds_laz])
            if subprogress_cb:
                subprogress_cb({"type": "substep", "phase": "MDS", "status": "done", "mi3": mi3, "worker_id": worker_id})
            # MDT: only class 2, cell 0.25m, triangulated Z
            if subprogress_cb:
                subprogress_cb({"type": "substep", "phase": "MDT", "status": "start", "mi3": mi3, "worker_id": worker_id})
            mdt_pipe = pdal.Pipeline(json.dumps({
                "pipeline": [
                    {"type": "readers.las", "filename": out_path_classified},
                    {"type": "filters.range", "limits": "Classification[2:2]"}
                ]
            }))
            mdt_pipe.execute()
            mdt_arrs = mdt_pipe.arrays
            if mdt_arrs:
                lote_dir = f"LOTE_{str(lote_to_use).zfill(2)}"
                bloco_dir = f"BLOCO_{bloco_to_use}"
                mdt_las_dir = os.path.join(root_out, "7_MDT", lote_dir, bloco_dir, "1_LAS")
                mdt_txt_dir = os.path.join(root_out, "7_MDT", lote_dir, bloco_dir, "2_ASCII")
                os.makedirs(mdt_las_dir, exist_ok=True)
                os.makedirs(mdt_txt_dir, exist_ok=True)
                name_base = f"ES_L{str(lote_to_use).zfill(2)}_{bloco_to_use}_MDT_{mi3}_R0"
                mdt_txt = os.path.join(mdt_txt_dir, f"{name_base}.txt")
                mdt_laz = os.path.join(mdt_las_dir, f"{name_base}.laz")
                if triangulated_grid_to_txt_laz(mdt_arrs[0], 0.25, mdt_txt, mdt_laz):
                    result["generated_files"].extend([mdt_txt, mdt_laz])
            if subprogress_cb:
                subprogress_cb({"type": "substep", "phase": "MDT", "status": "done", "mi3": mi3, "worker_id": worker_id})
        except Exception as _e:
            # Silenciar erro de grid para não interromper o fluxo principal
            pass
            
    except Exception as e:
        result["status"] = "erro"
        result["mensagem"] = str(e)
    finally:
        result["duracao_s"] = round(time.time() - start, 3)
    return result
def run_processing(
    params: AppParams,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_event: Optional[threading.Event] = None,
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
            
            # Apenas criar job se houver candidatos (interseção provável com tiles)
            if candidate_tiles:
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
            # ThreadPool para permitir callbacks e melhor cancelamento cooperativo
            with ThreadPoolExecutor(max_workers=params.workers) as ex:
                # worker id slotting
                slot_lock = threading.Lock()
                slots = list(range(1, (params.workers or 1) + 1))
                def acquire_slot() -> int:
                    with slot_lock:
                        return slots.pop(0) if slots else 0
                def release_slot(s: int) -> None:
                    if s:
                        with slot_lock:
                            slots.append(s)
                def do_job(j: Dict[str, Any]):
                    wid = acquire_slot()
                    mi3j = j.get("mi3")
                    try:
                        # Só notificar início se houver tiles candidatos
                        # Log somente via substep para evitar duplicidade
                        if cancel_event and cancel_event.is_set():
                            return {"status": "cancelado", "saida_filename": None, "generated_files": []}
                        return process_one_feature(j, subprogress_cb=(lambda ev: progress_cb(ev) if progress_cb else None), cancel_event=cancel_event, worker_id=wid)
                    finally:
                        release_slot(wid)
                futs = [ex.submit(do_job, j) for j in jobs]
                completed = 0
                for fut in as_completed(futs):
                    try:
                        res = fut.result()
                    except Exception as e:
                        # Não interromper: registrar erro como resultado e seguir
                        res = {"status": "erro", "mensagem": str(e), "saida_filename": None, "generated_files": []}
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
                        if isinstance(res, dict) and res.get("status") == "ok":
                            printed_any = False
                            for fp in (res.get("generated_files") or []):
                                if fp:
                                    print(f"{os.path.basename(fp)} exportado")
                                    printed_any = True
                            if (not printed_any) and res.get("saida_filename"):
                                for part in str(res.get("saida_filename")).split(';'):
                                    p = part.strip()
                                    if p:
                                        print(f"{os.path.basename(p)} exportado")
                        # Se cancelado, apagar saídas geradas deste job
                        if cancel_event and cancel_event.is_set():
                            for fp in res.get("generated_files", []) or []:
                                try:
                                    if fp and os.path.exists(fp):
                                        os.remove(fp)
                                except Exception:
                                    pass
                    except Exception:
                        pass
        else:
            for i, j in enumerate(jobs):
                if cancel_event and cancel_event.is_set():
                    res = {"status": "cancelado", "saida_filename": None, "generated_files": []}
                else:
                    res = process_one_feature(j, subprogress_cb=(lambda ev: progress_cb(ev) if progress_cb else None), cancel_event=cancel_event, worker_id=1)
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
                    if isinstance(res, dict) and res.get("status") == "ok":
                        printed_any = False
                        for fp in (res.get("generated_files") or []):
                            if fp:
                                print(f"{os.path.basename(fp)} exportado")
                                printed_any = True
                        if (not printed_any) and res.get("saida_filename"):
                            for part in str(res.get("saida_filename")).split(';'):
                                p = part.strip()
                                if p:
                                    print(f"{os.path.basename(p)} exportado")
                    if cancel_event and cancel_event.is_set():
                        for fp in res.get("generated_files", []) or []:
                            try:
                                if fp and os.path.exists(fp):
                                    os.remove(fp)
                            except Exception:
                                pass
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
