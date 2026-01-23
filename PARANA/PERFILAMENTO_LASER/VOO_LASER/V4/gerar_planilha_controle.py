import os
from datetime import datetime

import geopandas as gpd
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font

EPSG_UTM22S = "EPSG:31982"
CAMINHO_TEMPLATE_CONTROLE = r"\\192.168.2.28\i\80225_PROJETO_IAT_PARANA\3 Execução de voo\ES_PC_LASER_LXX_Y_R0_ddmmaaaa.xlsx"


def carregar_blocos_area(dir_plano_voo):
    shp_blocos = os.path.join(dir_plano_voo, "Blocos_PR_utm.shp")
    if os.path.exists(shp_blocos):
        gdf = gpd.read_file(shp_blocos).to_crs(EPSG_UTM22S)
        gdf["AREA_KM2"] = (gdf.geometry.area / 1e6).round(3)
        return gdf
    return None


def cm_to_px(cm):
    return int(cm * 96 / 2.54)


def gerar_planilha_controle(dir_execucao_voo, blocos_info, datas_filtro):
    dir_plano_voo = os.path.join(dir_execucao_voo, "PLANO_DE_VOO")
    gdf_blocos = carregar_blocos_area(dir_plano_voo)

    for info in blocos_info:
        dir_bloco = info["dir_bloco"]
        bloco_nome = info["bloco_nome"]
        bloco_letra = bloco_nome.split("_")[-1]
        lote_id = info["lote_id"]
        dir_lote = info["dir_lote"]

        if not os.path.isdir(dir_bloco):
            print(f"[AVISO] Diretório do bloco não encontrado: {dir_bloco}")
            continue

        for pasta in os.listdir(dir_bloco):
            if not pasta.startswith("2025"):
                continue

            data_str, _ = pasta.split("_", 1)
            data_dt = datetime.strptime(data_str, "%Y%m%d")
            if datas_filtro and data_dt.date() not in datas_filtro:
                continue

            data_formatada = data_dt.strftime("%d%m%Y")
            dir_dia = os.path.join(dir_bloco, pasta)

            nome_arquivo_controle = f"ES_PC_LASER_L{lote_id}_{bloco_letra}_R0_{data_formatada}.xlsx"

            dir_saida = os.path.join(dir_lote, "ENTREGA", "1_PLANILHA_CONTROLE", bloco_nome)
            os.makedirs(dir_saida, exist_ok=True)
            caminho_saida_controle = os.path.join(dir_saida, nome_arquivo_controle)

            wb_ctrl = load_workbook(CAMINHO_TEMPLATE_CONTROLE)
            aba_dados = wb_ctrl["Dados Gerais"]

            faixas_shp = [f for f in os.listdir(dir_dia) if f.endswith(".shp") and "_TRJ" not in f]
            if not faixas_shp:
                print(f"[AVISO] Nenhum shapefile de faixa encontrado em {dir_dia}")
                continue

            gdf_faixas = gpd.read_file(os.path.join(dir_dia, faixas_shp[0])).to_crs(EPSG_UTM22S)
            velocidades = (
                gdf_faixas["Speed_[m/s"]
                .apply(lambda x: x * 1.94384 if pd.notna(x) else None)
                .dropna()
                .tolist()
            )
            if velocidades:
                vel_media = round(sum(velocidades) / len(velocidades), 1)
                vel_media_str = f"{vel_media:.1f}".replace(".", ",") + " knots"
                aba_dados["C7"] = vel_media_str

            aba_dados["C12"] = len(gdf_faixas)

            caminho_print = os.path.join(dir_lote, "PRINT", bloco_nome, f"{data_dt.strftime('%Y%m%d')}_TRJ.png")
            if os.path.exists(caminho_print):
                aba_trajetoria = wb_ctrl["Trajetoria Ajustada"]
                img = XLImage(caminho_print)
                img.width = cm_to_px(20)
                img.height = cm_to_px(13)
                aba_trajetoria.add_image(img, "C3")

            aba_np = wb_ctrl["Print NP no Bloco"]
            caminho_ele = os.path.join(dir_lote, "PRINT", bloco_nome, f"{data_dt.strftime('%Y%m%d')}_ELE.png")
            if os.path.exists(caminho_ele):
                img_ele = XLImage(caminho_ele)
                img_ele.height = cm_to_px(15)
                img_ele.width = cm_to_px(25)
                aba_np.add_image(img_ele, "B3")

            caminho_int = os.path.join(dir_lote, "PRINT", bloco_nome, f"{data_dt.strftime('%Y%m%d')}_INT.png")
            if os.path.exists(caminho_int):
                img_int = XLImage(caminho_int)
                img_int.height = cm_to_px(15)
                img_int.width = cm_to_px(25)
                aba_np.add_image(img_int, "R3")

            if gdf_blocos is not None:
                filtro = gdf_blocos[gdf_blocos["BLOCOS"] == bloco_letra]
                if "LOTE" in gdf_blocos.columns:
                    try:
                        lote_num = int(lote_id)
                    except ValueError:
                        lote_num = None
                    if lote_num is not None:
                        filtro = filtro[filtro["LOTE"] == lote_num]
                if not filtro.empty:
                    area_km2 = filtro.iloc[0]["AREA_KM2"]
                    if pd.notna(area_km2):
                        texto_area = f"\u00c1REA: {area_km2:.3f} KM²".replace(".", ",")
                        cell_area = aba_np["C40"]
                        cell_area.value = texto_area
                        cell_area.font = Font(name="Calibri", size=18)

            wb_ctrl.save(caminho_saida_controle)
            print(f"[OK] Planilha de Controle salva: {caminho_saida_controle}")
