import os

import geopandas as gpd
import pandas as pd

# === CONFIGURAÇÕES ===
EPSG_UTM22S = "EPSG:31982"


def carregar_shape_blocos(dir_execucao_voo):
    shp_blocos = os.path.join(dir_execucao_voo, "PLANO_DE_VOO", "Blocos_PR_utm.shp")
    if os.path.exists(shp_blocos):
        return gpd.read_file(shp_blocos).to_crs(EPSG_UTM22S)
    return None


def gerar_trajetoria_unificada(dir_execucao_voo, blocos_info, datas_filtro=None, log_func=None):
    gdf_blocos = carregar_shape_blocos(dir_execucao_voo)

    for info in blocos_info:
        dir_bloco = info["dir_bloco"]
        bloco_nome = info["bloco_nome"]
        lote_id = info["lote_id"]
        dir_lote = info["dir_lote"]

        if not os.path.isdir(dir_bloco):
            if log_func:
                log_func(f"[AVISO] Diretório do bloco não encontrado: {dir_bloco}\n")
            continue

        if log_func:
            log_func(f"[INFO] Exportando trajetórias de {bloco_nome} ({info['lote_nome']})...\n")

        trajetorias = []

        for pasta in sorted(os.listdir(dir_bloco)):
            if not pasta.startswith("2025"):
                continue

            try:
                data_str, _ = pasta.split("_", 1)
                data_dt = pd.to_datetime(data_str, format="%Y%m%d").date()
            except Exception:
                continue

            if datas_filtro and data_dt not in datas_filtro:
                continue

            dir_dia = os.path.join(dir_bloco, pasta)
            trj_shps = [f for f in os.listdir(dir_dia) if f.endswith(".shp") and "_TRJ" in f]
            for trj in trj_shps:
                caminho_trj = os.path.join(dir_dia, trj)
                try:
                    gdf_trj = gpd.read_file(caminho_trj)
                    if gdf_trj.crs is None:
                        gdf_trj.set_crs(EPSG_UTM22S, inplace=True)
                    else:
                        gdf_trj = gdf_trj.to_crs(EPSG_UTM22S)
                    trajetorias.append(gdf_trj)
                except Exception as e:
                    if log_func:
                        log_func(f"[ERRO] Falha ao ler {caminho_trj}: {e}\n")

        if not trajetorias:
            if log_func:
                log_func(f"[AVISO] Nenhuma trajetória encontrada em {bloco_nome}.\n")
            continue

        gdf_trj_merged = gpd.GeoDataFrame(pd.concat(trajetorias, ignore_index=True), crs=EPSG_UTM22S)

        bloco_id = bloco_nome.split("_")[-1]
        lote_final = lote_id
        if gdf_blocos is not None and not gdf_blocos.empty:
            centroide_trj = gdf_trj_merged.unary_union.centroid
            filtro = gdf_blocos[gdf_blocos.contains(centroide_trj)]
            if "LOTE" in gdf_blocos.columns:
                try:
                    lote_num = int(lote_id)
                except ValueError:
                    lote_num = None
                if lote_num is not None:
                    filtro = filtro[filtro["LOTE"] == lote_num]
            if not filtro.empty:
                bloco_id = filtro.iloc[0]["BLOCOS"]
                lote_final = str(filtro.iloc[0]["LOTE"]).zfill(2)
            elif log_func:
                log_func(f"[AVISO] Bloco não identificado no shape para {bloco_nome}. Usando nome e lote informados.\n")

        nome_saida = f"ES_L{lote_final}_{bloco_id}_FAIXAS_VOO_LASER_R0"

        dir_saida_base = os.path.join(dir_lote, "ENTREGA", "4_TRAJETORIA", bloco_nome)
        dir_saida_shp = os.path.join(dir_saida_base, "SHP")
        dir_saida_kmz = os.path.join(dir_saida_base, "KMZ")
        os.makedirs(dir_saida_shp, exist_ok=True)
        os.makedirs(dir_saida_kmz, exist_ok=True)

        shp_path = os.path.join(dir_saida_shp, nome_saida + ".shp")
        kmz_path = os.path.join(dir_saida_kmz, nome_saida + ".kmz")

        try:
            gdf_trj_merged.to_file(shp_path)
            gdf_trj_merged.to_file(kmz_path, driver="KML")
            if log_func:
                log_func(f"[OK] Exportado:\n  {shp_path}\n {kmz_path}\n")
        except Exception as e:
            if log_func:
                log_func(f"[ERRO] Falha ao exportar trajetórias de {bloco_nome}: {e}\n")
