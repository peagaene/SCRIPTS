import pandas as pd
import os
import logging

def listar_arquivos_em_diretorio(diretorio):
    return [os.path.join(diretorio, nome_arquivo) for nome_arquivo in os.listdir(diretorio) if nome_arquivo.endswith('.txt')]

def carregar_dataframes(arquivos):
    dfs = {}
    for arquivo in arquivos:
        logging.debug(f"Carregando DataFrame do arquivo {arquivo}")
        try:
            df = pd.read_csv(arquivo, sep='\t')
            dfs[arquivo] = df
        except Exception as e:
            logging.error(f"Falha ao carregar arquivo {arquivo}: {str(e)}")
    return dfs

def comparar_e_salvar_correspondencias(diretorio1, diretorio2):
    arquivos_diretorio1 = listar_arquivos_em_diretorio(diretorio1)
    arquivos_diretorio2 = listar_arquivos_em_diretorio(diretorio2)
    
    dfs = carregar_dataframes(arquivos_diretorio1 + arquivos_diretorio2)

    unmatched_result_data = []

    for arquivo1 in arquivos_diretorio1:
        for arquivo2 in arquivos_diretorio2:
            if os.path.basename(arquivo1) == os.path.basename(arquivo2):
                logging.debug(f"Comparando arquivos {arquivo1} e {arquivo2}")

                # Carregar os DataFrames
                df1 = dfs[arquivo1]
                df2 = dfs[arquivo2]

                # Ajustar a coluna 'GPS Time' para considerar apenas os primeiros 8 caracteres
                df1['GPS Time'] = df1['GPS Time'].astype(str).str[:8]
                df2['GPS Time'] = df2['GPS Time'].astype(str).str[:8]

                # Realizar a mesclagem com base nos primeiros 8 caracteres de 'GPS Time'
                merged_df = df1.merge(df2, left_on=['GPS Time'], right_on=['GPS Time'], how='outer', suffixes=('_1', '_2'))

                # Restante do código permanece inalterado
                non_matching_rows = merged_df[merged_df['Filename_1'].isnull() | merged_df['Filename_2'].isnull()]
                for index, row in non_matching_rows.iterrows():
                    if pd.notnull(row['Filename_1']):
                        unmatched_result_data.append({
                            'Imagem analisada': row['Filename_1'][:-4],
                            'Voo': int(os.path.basename(arquivo1)[:-4])
                        })
                    elif pd.notnull(row['Filename_2']):
                        unmatched_result_data.append({
                            'Imagem analisada': row['Filename_2'][:-4],
                            'Voo': int(os.path.basename(arquivo2)[:-4])
                        })

    unmatched_result_df = pd.DataFrame(unmatched_result_data)

    caminho_arquivo_sem_correspondencia = os.path.join(diretorio1, 'Sem_Correspondencia_RGB_NIR.xlsx')

    logging.info("Salvando arquivos sem correspondência em arquivo Excel...")
    unmatched_result_df.to_excel(caminho_arquivo_sem_correspondencia, index=False)

    logging.info("Processo concluído.")

if __name__ == "__main__":
    logging.basicConfig(filename='comparacao_logs.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    diretorio1 = r'D:\Pedro\PROJETOS\GOV_SP\SI_13'
    diretorio2 = r'D:\Pedro\PROJETOS\GOV_SP\SI_13\NIR'
    
    logging.info("Iniciando processo de comparação...")
    comparar_e_salvar_correspondencias(diretorio1, diretorio2)