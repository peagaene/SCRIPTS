import os
import pandas as pd
import numpy as np
from dbfread import DBF

def ler_arquivo_com_cabecalho_e_margem(caminho_arquivo, linhas_cabecalho=39):
    with open(caminho_arquivo, "r") as arquivo:
        for _ in range(linhas_cabecalho):
            next(arquivo)
        linhas = arquivo.readlines()
    linhas = [linha.strip().split(None, 1)[-1] for linha in linhas]
    return linhas

def extract_info_from_filename(filename):
    year = int(filename[:4])
    month = int(filename[4:6])
    day = int(filename[6:8])

    # Adiciona um zero se o mês ou o dia for menor que 10
    month_str = f"{month:02d}"
    day_str = f"{day:02d}"
    return year, month_str, day_str

def process_txt_file(file_path, diretorio2):
    with open(file_path, "r") as file:
        lines = file.readlines()
        header = lines[0].strip().split('\t')
        gps_time_index = header.index("GPS Time")
        longitude_index = header.index("Longitude")
        latitude_index = header.index("Latitude")
        filename_index = header.index("Filename")
        time_gps_index = header.index("Weeks:Seconds")
        results = []

        for line in lines[1:]:
            values = line.strip().split('\t')
            gps_time = values[gps_time_index]
            longitude = float(values[longitude_index])
            latitude = float(values[latitude_index])
            filename = values[filename_index]
            time_gps = values[time_gps_index][5:]

            if filename.lower().endswith(".iiq"):
                filename = filename[:-4]

            hora, minuto, segundo = map(float, gps_time.split(":"))
            result_row = {
                "Hora": f"{int(hora-3):02d}:{int(minuto):02d}:{int(segundo):02d}",
                "Latitude": latitude,
                "Longitude": longitude,
                "Imagem analisada": filename,
                "Tempo GPS": float(time_gps),
            }
            results.append(result_row)

        df_log_foto = pd.DataFrame(results)

    ano, mes, dia = extract_info_from_filename(os.path.basename(file_path))
    caminho_arquivo = os.path.join(diretorio2, f"{ano}{mes}{dia}.txt")
    linhas = ler_arquivo_com_cabecalho_e_margem(caminho_arquivo, linhas_cabecalho=39)

    terceira_coluna_valores = []
    quarta_coluna_valores = []
    quinta_coluna_valores = []
    sexta_coluna_valores = []
    setima_coluna_valores = []
    oitava_coluna_valores = []
    deriva_coluna = []
    faixa_coluna = []

    for linha in linhas:
        valores = linha.strip().split()
        if len(valores) >= 8:
            valor_segunda_coluna = float(valores[0])
            valor_terceira_coluna = float(valores[1])
            valor_quarta_coluna = float(valores[2])
            valor_quinta_coluna = float(valores[3])
            valor_sexta_coluna = float(valores[4])
            valor_setima_coluna = float(valores[5])
            valor_oitava_coluna = float(valores[6])
            valor_kappa = valor_oitava_coluna + 155 if valor_oitava_coluna < 0 else valor_oitava_coluna - 25

            terceira_coluna_valores.append(valor_terceira_coluna)
            quarta_coluna_valores.append(valor_quarta_coluna)
            quinta_coluna_valores.append(valor_quinta_coluna)
            sexta_coluna_valores.append(valor_sexta_coluna)
            setima_coluna_valores.append(valor_setima_coluna)
            oitava_coluna_valores.append(valor_oitava_coluna)
            deriva_coluna.append(valor_kappa)
            faixa_coluna.append(valor_segunda_coluna)

    df_terceira_coluna = pd.DataFrame({
        "E": terceira_coluna_valores,
        "N": quarta_coluna_valores,
        "Z": quinta_coluna_valores,
        "Omega": sexta_coluna_valores,
        "Phi": setima_coluna_valores,
        "Kappa": oitava_coluna_valores,
        "Deriva": deriva_coluna
    })

    faixa_faixa = []
    sobreposicao = []
    current_faixa = 1
    lado = 1392.96
    valor_y = []

    for i in range(len(faixa_coluna)-1):
        time_diff = float(faixa_coluna[i]) - float(faixa_coluna[i-1])
        point2 = np.array([terceira_coluna_valores[i], quarta_coluna_valores[i]])
        point1 = np.array([terceira_coluna_valores[i+1], quarta_coluna_valores[i+1]])
        dist = np.linalg.norm(point1 - point2)
        x = lado - dist
        valor_y.append(round((x/lado)*100, 2))

        if time_diff > 20:
            current_faixa += 1
        faixa_faixa.append(current_faixa)
        sobreposicao.append(valor_y[i-1] if (valor_y[i] > 100 or valor_y[i] < 0) else valor_y[i])

    time_diff = float(faixa_coluna[-1]) - float(faixa_coluna[-2])
    point2 = np.array([terceira_coluna_valores[-1], quarta_coluna_valores[-1]])
    point1 = np.array([terceira_coluna_valores[-2], quarta_coluna_valores[-2]])
    dist = np.linalg.norm(point1 - point2)
    x = lado - dist
    y = (round((x/lado)*100, 2))

    if time_diff > 20:
        current_faixa += 1

    faixa_faixa.append(current_faixa)
    sobreposicao.append(0 if y > 100 else y)

    df_terceira_coluna["Faixa2"] = faixa_faixa
    df_terceira_coluna["Sobreposição"] = sobreposicao

    return df_log_foto, df_terceira_coluna

def main():
    diretorio = r"D:\Pedro\PROJETOS\GOV_SP\SI_15"
    diretorio2 = r"D:\Pedro\PROJETOS\GOV_SP\SI_15\EO"

    for file_name in os.listdir(diretorio):
        if file_name.endswith(".txt"):
            df_log_foto, df_terceira_coluna = process_txt_file(os.path.join(diretorio, file_name), diretorio2)

            # Combine os resultados
            df_combined = pd.concat([df_log_foto, df_terceira_coluna], axis=1)

            # Verifique se o arquivo de referência existe
            caminho_arquivo_bordo = os.path.join(diretorio, file_name[:-4] + '.xlsx')
            
            if os.path.exists(caminho_arquivo_bordo):
                df_bordo = pd.read_excel(caminho_arquivo_bordo)

                for index, row in df_combined.iterrows():
                    imagem_analisada = row["Imagem analisada"]
                    matches = df_bordo[df_bordo["Foto Inicial"].str.strip().str.lower().isin([imagem_analisada]) | df_bordo["Foto Final"].str.strip().str.lower().isin([imagem_analisada])]
                    if not matches.empty:
                        faixa = matches.iloc[0]["Faixa"]
                        df_combined.loc[(index, 'Faixa')] = faixa

            mapeamento = {}
            for index, row in df_combined.iterrows():
                if pd.notna(row['Faixa']):
                    faixa = row['Faixa']
                    faixa2 = row['Faixa2']
                    mapeamento[faixa2] = faixa 

            df_combined['Faixa'] = df_combined['Faixa2'].map(mapeamento)

            columns_interest = ["Imagem analisada", "Tempo GPS", "E", "N", "Z", "Omega", "Phi", "Kappa", "Latitude", "Longitude", "Sobreposição", "Hora", "Deriva", "Faixa"]

            df_fim = df_combined[columns_interest]
            output_file_name_combined = os.path.join (diretorio, f"Preliminar_{file_name[:8]}.xlsx")
            df_fim.to_excel(output_file_name_combined, index=False)

            print("Os resultados da faixa", file_name[:-4], "foram exportados")

if __name__ == "__main__":
    main()
