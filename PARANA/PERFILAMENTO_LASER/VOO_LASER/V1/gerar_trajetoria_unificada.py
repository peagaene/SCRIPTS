import os
import geopandas as gpd
import pandas as pd

# === CONFIGURAÇÕES ===
EPSG_UTM22S = "EPSG:31982"
DIR_BASE = r"\\192.168.2.28\i\80225_PROJETO_IAT_PARANA\3 Execução de voo"
SHAPE_BLOCOS = os.path.join(DIR_BASE, "PLANO_DE_VOO", "Blocos_PR_utm.shp")
DIR_SAIDA_TRAJ = os.path.join(DIR_BASE, "ENTREGA", "4_TRAJETORIA")

def gerar_trajetoria_unificada(dir_execucao_voo, blocos, datas_filtro=None, log_func=None):
    gdf_blocos = gpd.read_file(SHAPE_BLOCOS).to_crs(EPSG_UTM22S)

    for bloco_nome in blocos:
        dir_bloco = os.path.join(dir_execucao_voo, bloco_nome)
        if not os.path.isdir(dir_bloco):
            continue

        if log_func:
            log_func(f"[INFO] Exportando trajetórias do {bloco_nome}...\n")

        trajetorias = []

        for pasta in sorted(os.listdir(dir_bloco)):
            if not pasta.startswith("2025"):
                continue

            try:
                data_str, _ = pasta.split("_")
                data_dt = pd.to_datetime(data_str, format="%Y%m%d").date()
            except:
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

        centroide_trj = gdf_trj_merged.unary_union.centroid
        bloco_info = gdf_blocos[gdf_blocos.contains(centroide_trj)]

        if not bloco_info.empty:
            bloco_id = bloco_info.iloc[0]['BLOCOS']
            lote_id = str(bloco_info.iloc[0]['LOTE']).zfill(2)
        else:
            bloco_id = bloco_nome[-1]
            lote_id = "XX"
            if log_func:
                log_func(f"[ERRO] Não foi possível identificar o lote para {bloco_nome}. Usando XX.\n")

        nome_saida = f"ES_L{lote_id}_{bloco_id}_FAIXAS_VOO_LASER_R0"

        dir_saida_base = os.path.join(DIR_SAIDA_TRAJ, bloco_nome)
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
                log_func(f"[OK] Exportado:\n  → {shp_path}\n  → {kmz_path}\n")
        except Exception as e:
            if log_func:
                log_func(f"[ERRO] Falha ao exportar trajetórias do {bloco_nome}: {e}\n")
