import os
from tqdm import tqdm
import pandas as pd

def gerar_planilha_imagens_faltantes(lista_txt, pastas_origem, arquivo_planilha):
    # Lê a lista de imagens do arquivo .txt
    with open(lista_txt, "r") as file:
        lista_imagens = {linha.strip() for linha in file if linha.strip()}

    # Lista para armazenar imagens faltantes
    imagens_faltantes_lista = []

    # Contador total de arquivos para exibição de progresso
    total_faltantes = 0
    for pasta_origem in pastas_origem:
        if os.path.exists(pasta_origem):
            arquivos_origem = set(os.listdir(pasta_origem))
            total_faltantes += len(arquivos_origem - lista_imagens)

    # Processa cada pasta de origem
    with tqdm(total=total_faltantes, desc="Identificando imagens faltantes", unit="imagem") as pbar:
        for pasta_origem in pastas_origem:
            print(f"Processando pasta: {pasta_origem}")
            if not os.path.exists(pasta_origem):
                print(f"Pasta não encontrada: {pasta_origem}")
                continue

            # Lê os arquivos presentes na pasta de origem
            arquivos_origem = set(os.listdir(pasta_origem))

            # Identifica as imagens que estão na pasta de origem, mas não na lista
            imagens_faltantes = arquivos_origem - lista_imagens

            # Adiciona à lista de imagens faltantes
            for imagem in imagens_faltantes:
                imagens_faltantes_lista.append({"Imagem": imagem, "Pasta de Origem": pasta_origem})

                # Atualiza a barra de progresso
                pbar.update(1)

    # Salva a lista de imagens faltantes em uma planilha Excel
    df_faltantes = pd.DataFrame(imagens_faltantes_lista)
    df_faltantes.to_excel(arquivo_planilha, index=False)

    print(f"Processo concluído. Planilha gerada em: {arquivo_planilha}")

# Exemplo de uso
lista_txt = r'H:\Lista_imagens_Aerotri_SI_16_HD04.txt'  # Arquivo .txt com os nomes das imagens
pastas_origem = [
    r'H:\IMAGENS_RGBI_SI16',
    r'I:\IMAGENS_RGBI_SI16'
]  # Lista de pastas onde estão as imagens
arquivo_planilha = r'E:\imagens_faltantes.xlsx'  # Caminho para salvar a planilha

gerar_planilha_imagens_faltantes(lista_txt, pastas_origem, arquivo_planilha)
