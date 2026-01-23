from datetime import datetime, timedelta
import os
import pandas as pd
from dbfread import DBF
from openpyxl import load_workbook, Workbook

# Define o diretório base onde estão localizados os arquivos
BASE_DIRECTORY = r'D:\Pedro\PROJETOS\GOV_SP\SI_09' 

def find_nearest_time(gps_list, target_gps, mode='both'):
    if mode == 'both':
        nearest_greater = min(filter(lambda x: x >= target_gps - 20, gps_list), default=None)
        nearest_lesser = max(filter(lambda x: x <= target_gps + 20, gps_list), default=None)

        if nearest_greater is not None and nearest_lesser is not None:
            return nearest_greater if abs(target_gps - nearest_greater) <= abs(target_gps - nearest_lesser) else nearest_lesser
        elif nearest_greater is not None:
            return nearest_greater
        elif nearest_lesser is not None:
            return nearest_lesser
        else:
            return None

# Função para converter GPS Time para Hora Local
def convert_gps_to_hora_local(gps_seconds):
    total_seconds = 604800 + float(gps_seconds)
    gps_time = timedelta(seconds=total_seconds)

    # Subtrair 3 horas para obter o UTC-3 (Horário Padrão de Brasília)
    utc_minus_3_time = gps_time - timedelta(hours=3)

    # Retornar apenas o componente de hora no formato HH:MM:SS
    hora_local = utc_minus_3_time.total_seconds() % 86400
    horas = int(hora_local // 3600)
    minutos = int((hora_local % 3600) // 60)
    segundos = int(hora_local % 60)

    return f"{horas:02d}:{minutos:02d}:{segundos:02d}"

def add_nomenclatura(aba_df, txt_file_path):
    # Carregar o arquivo txt com as colunas "Filename" e "Week:Seconds"
    with open(txt_file_path, 'r') as file:
        # Ler o arquivo linha por linha
        lines = file.readlines()
    
    # Criar um dicionário para mapear os tempos GPS para as nomenclaturas das fotos
    gps_dict = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            gps_week_seconds = parts[3].split(':')
            filename = parts[0].split('.')
            if len(gps_week_seconds) == 2:
                gps_week_seconds = float(gps_week_seconds[-1])  # Pega apenas o valor à direita do ":"
                filename = str(filename[0])
                gps_dict[gps_week_seconds] = filename

    # Listas para armazenar as nomenclaturas associadas à hora inicial e hora final
    nomenclaturas_iniciais = []
    nomenclaturas_finais = []

    for _, row in aba_df.iterrows():
        hora_inicial = row['Hora Inicial']
        hora_final = row['Hora Final']

        gps_inicial = find_nearest_time(list(gps_dict.keys()), float(hora_inicial), mode='both')
        gps_final = find_nearest_time(list(gps_dict.keys()), float(hora_final), mode='both')

        nomenclatura_inicial = gps_dict.get(gps_inicial, '')
        nomenclatura_final = gps_dict.get(gps_final, '')

        nomenclaturas_iniciais.append(nomenclatura_inicial)
        nomenclaturas_finais.append(nomenclatura_final)

    # Adicionar as colunas de nomenclaturas ao DataFrame
    aba_df['Foto Inicial'] = nomenclaturas_iniciais
    aba_df['Foto Final'] = nomenclaturas_finais
    return aba_df

def merge_same_faixa_rows(df):
    merged_rows = []
    current_row = None

    for _, row in df.iterrows():
        if current_row is None:
            current_row = row.copy()
        else:
            if row['Faixa'] == current_row['Faixa']:
                current_row['Hora Final'] = row['Hora Final']
                current_row['Foto Final'] = row['Foto Final']
            else:
                merged_rows.append(current_row)
                current_row = row.copy()

    if current_row is not None:
        merged_rows.append(current_row)

    return pd.DataFrame(merged_rows)

def process_flight(voo, dbf_file_path, txt_file_path, formatted_date):
    dbf_data = DBF(dbf_file_path)
    records = []
    voo = voo.split('_')[0]

    for record in dbf_data:
        hora_inicial = record['GPS_Start_']
        hora_final = record['GPS_Stop_T']
        freq_emitida = record['Scanner_Fr']
        freq_varredura = record['System_PRF']
        velocidade_de_voo = record['Speed_[m/s']
        altura_de_voo = record['Planned_AG']
        angulo_abertura = record['Scanner_Ha']
        comprimento_faixa = ((float(hora_final) - float(hora_inicial)) * float(velocidade_de_voo))
        record['Hora Inicial'] = hora_inicial
        record['Hora Final'] = hora_final
        record['Frequencia Emitida'] = freq_emitida
        record['Frequencia de Varredura'] = freq_varredura
        record['Velocidade de voo'] = velocidade_de_voo
        record['Altura de voo'] = altura_de_voo
        record['Angulo de Abertura'] = angulo_abertura
        record['Comprimento da Faixa'] = comprimento_faixa
        records.append(record)

    df_voo = pd.DataFrame(records)
    df_voo['Data'] = formatted_date
    df_voo['Faixa'] = voo + '_' + df_voo['Flight_Lin'].astype(str)
    df_voo = df_voo.loc[:, ['Data', 'Hora Inicial', 'Hora Final', 'Faixa']]
    df_laser = pd.DataFrame(records)
    df_laser['Numero de Faixas'] = None
    df_laser = df_laser.loc[:, ['Frequencia Emitida', 'Frequencia de Varredura', 'Velocidade de voo', 'Altura de voo', 'Angulo de Abertura', 'Comprimento da Faixa', 'Numero de Faixas']]

    # Calcular as médias dos valores do DataFrame de laser
    df_laser_mean = df_laser.mean()
    
    df_laser_mean['Velocidade de voo'] = round(df_laser_mean['Velocidade de voo'], 1)

    comprimento_faixa_max = df_laser['Comprimento da Faixa'].max()  # Calcula o valor máximo da coluna de comprimento da faixa

    # Substituir o valor da média pelo valor máximo no DataFrame df_laser_mean
    df_laser_mean['Comprimento da Faixa'] = comprimento_faixa_max

    # Adicionar as nomenclaturas do arquivo .txt correspondente
    df_voo = add_nomenclatura(df_voo, txt_file_path)

    # Ordenar o DataFrame pelo valor da coluna 'Hora Inicial' em ordem crescente
    df_voo = df_voo.sort_values(by='Hora Inicial')

    # Aplicar a conversão para hora local nas colunas 'Hora Inicial' e 'Hora Final'
    df_voo['Hora Inicial'] = df_voo.apply(lambda row: convert_gps_to_hora_local(row['Hora Inicial']), axis=1)
    df_voo['Hora Final'] = df_voo.apply(lambda row: convert_gps_to_hora_local(row['Hora Final']), axis=1)

    # Unir as linhas onde a faixa for igual
    df_voo = merge_same_faixa_rows(df_voo)

    # Calcular o número de faixas
    numero_faixas = len(df_voo)
    df_laser_mean['Numero de Faixas'] = numero_faixas

    # Salvar o DataFrame em um arquivo Excel individual com o nome do voo
    output_file_path = os.path.join(BASE_DIRECTORY, f'{voo}_F1_12SEN309.xlsx')
    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        df_voo.to_excel(writer, sheet_name=f'Bordo_{voo}', index=False)

        # Criar um DataFrame com as médias e formatá-lo corretamente
        df_laser_mean_formatted = pd.DataFrame({
            'Item': df_laser.columns,
            'Parametros': df_laser_mean
        })
        # Salvar o DataFrame de laser
        df_laser_mean_formatted.to_excel(writer, sheet_name=f'Laser_{voo}', index=False)

def main():
    df_excel = pd.read_excel(os.path.join(BASE_DIRECTORY, 'relatorio_de_voo.xlsx'), engine='openpyxl')
    
    for index, row in df_excel.iterrows():
        voo = str(row['VOO']) + "_12SEN309"
        voodate = str(row['VOO']).split('_')[0]

        # Check if 'voo' is a valid date before converting
        try:
            formatted_date = pd.to_datetime(voodate, format='%Y%m%d').strftime('%d/%m/%Y')
        except ValueError as e:
            print(f"Issue with date format in VOO {voo}: {str(e)}")
            continue  # Skip processing for this entry
        
        dbf_file_path = os.path.join(BASE_DIRECTORY, f'{voo}.dbf')
        txt_file_path = os.path.join(BASE_DIRECTORY, f'{voo}.txt')

        if not os.path.exists(dbf_file_path) or not os.path.exists(txt_file_path):
            print(f"Arquivos ausentes para o voo {voo}. Pulando o processamento.")
            continue

        try:
            process_flight(voo, dbf_file_path, txt_file_path, formatted_date)
        except Exception as e:
            print(f"Erro ao processar o voo {voo}: {str(e)}")

    print("Os resultados foram exportados")

if __name__ == "__main__":
    main()
