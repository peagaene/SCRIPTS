import os
import laspy
import numpy as np
import pandas as pd
from datetime import datetime

# === CONFIGURAÇÕES ===
EXT = (".las", ".laz")
TAMANHO_JANELA_M = 50
FATOR_DECIMACAO = 12
MODO_RETORNO = "todos"  # "todos" | "first" | "last"
DIR_SAIDA = r"\\192.168.2.28\i\80225_PROJETO_IAT_PARANA\3 Execução de voo\DENSIDADE_LAS"


def metros_para_graus(lat_ref, metros):
    lat_rad = np.radians(lat_ref)
    grau_lon_m = 111320 * np.cos(lat_rad)
    grau_lat_m = 111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
    return metros / grau_lon_m, metros / grau_lat_m


def filtro_retorno(las, modo):
    if modo == "first":
        return las.return_number == 1
    elif modo == "last":
        return las.return_number == las.number_of_returns
    else:
        return np.ones(len(las.x), dtype=bool)


def calcular_densidade(las_path, tamanho_m=50, modo="todos", fator=1):
    las = laspy.read(las_path)
    x, y = las.x, las.y
    mask_ret = filtro_retorno(las, modo)
    x, y = x[mask_ret], y[mask_ret]

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    ref_lat = (ymin + ymax) / 2
    dx_grau, dy_grau = metros_para_graus(ref_lat, tamanho_m)
    metade_x, metade_y = dx_grau / 2, dy_grau / 2

    regioes = [
        ("Q1", xmin + (xmax - xmin) * 0.25, ymax - (ymax - ymin) * 0.25),
        ("Q2", xmin + (xmax - xmin) * 0.75, ymax - (ymax - ymin) * 0.25),
        ("Q3", xmin + (xmax - xmin) * 0.25, ymin + (ymax - ymin) * 0.25),
        ("Q4", xmin + (xmax - xmin) * 0.75, ymin + (ymax - ymin) * 0.25),
    ]

    dens = []
    area_m2 = tamanho_m ** 2
    for _, cx, cy in regioes:
        dentro = (x >= cx - metade_x) & (x <= cx + metade_x) & \
                 (y >= cy - metade_y) & (y <= cy + metade_y)
        num = int(np.sum(dentro))
        densidade = (num / area_m2) * fator
        dens.append(round(densidade, 2))

    return dens


def gerar_relatorio_densidade(dir_execucao_voo, blocos, datas_filtro, log_func=None):
    os.makedirs(DIR_SAIDA, exist_ok=True)

    for bloco in blocos:
        dir_bloco = os.path.join(dir_execucao_voo, bloco)
        densidades_voo = {}

        for pasta in os.listdir(dir_bloco):
            if not pasta.startswith("2025"):
                continue

            try:
                data_str, _ = pasta.split("_")
                data_dt = datetime.strptime(data_str, "%Y%m%d").date()
            except:
                continue

            if datas_filtro and data_dt not in datas_filtro:
                continue

            caminho_voo = os.path.join(dir_bloco, pasta, "6 - ALTM_LOGS")
            if not os.path.isdir(caminho_voo):
                continue

            densidades_arquivos = []
            for subpasta in os.listdir(caminho_voo):
                caminho_rt = os.path.join(caminho_voo, subpasta, "Rt")
                if not os.path.isdir(caminho_rt):
                    continue

                for arquivo in os.listdir(caminho_rt):
                    if not arquivo.lower().endswith(EXT):
                        continue

                    caminho_las = os.path.join(caminho_rt, arquivo)
                    if log_func:
                        log_func(f"[INFO] Processando {bloco} / {pasta} / {arquivo}\n")
                    try:
                        dens = calcular_densidade(caminho_las, TAMANHO_JANELA_M, MODO_RETORNO, FATOR_DECIMACAO)
                        densidades_arquivos.append(dens)
                    except Exception as e:
                        if log_func:
                            log_func(f"[ERRO] {arquivo}: {e}\n")

            if densidades_arquivos:
                densidades_array = np.array(densidades_arquivos)
                medias = np.round(densidades_array.mean(axis=0), 2)
                densidades_voo[data_dt.strftime("%Y-%m-%d")] = medias.tolist()

        if densidades_voo:
            df_final = pd.DataFrame.from_dict(
                densidades_voo, orient="index",
                columns=["Densidade Q1 (ppm²)", "Q2", "Q3", "Q4"]
            ).reset_index().rename(columns={"index": "Data_Voo"})

            caminho_saida = os.path.join(DIR_SAIDA, f"densidade_{bloco}.xlsx")
            df_final.to_excel(caminho_saida, index=False)
            if log_func:
                log_func(f"[OK] Densidade salva: {caminho_saida}\n")
        else:
            if log_func:
                log_func(f"[AVISO] Nenhum dado para o bloco {bloco}\n")
