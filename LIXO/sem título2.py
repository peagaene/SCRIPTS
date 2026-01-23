import os
import shutil
from tqdm import tqdm
import pandas as pd

def copiar_imagens_faltantes(lista_txt, pastas_origem, pasta_destino, planilha_saida):
    # Lê a lista de imagens do arquivo .txt
    with open(lista_txt, "r") as file:
        lista_imagens = {linha.strip() for linha in file if linha.strip()}

    # Certifica-se de que a pasta de destino existe
    os.makedirs(pasta_destino, exist_ok=True)

    # Lista para registrar as imagens copiadas
    imagens_copiadas = []

    # Contador total de arquivos para exibição de progresso
    total_faltantes = 0
    for pasta_origem in pastas_origem:
        if os.path.exists(pasta_origem):
            arquivos_origem = set(os.listdir(pasta_origem))
            total_faltantes += len(arquivos_origem - lista_imagens)

    # Processa cada pasta de origem
    with tqdm(total=total_faltantes, desc="Copiando imagens faltantes", unit="imagem") as pbar:
        for pasta_origem in pastas_origem:
            print(f"Processando pasta: {pasta_origem}")
            if not os.path.exists(pasta_origem):
                print(f"Pasta não encontrada: {pasta_origem}")
                continue

            # Lê os arquivos presentes na pasta de origem
            arquivos_origem = set(os.listdir(pasta_origem))

            # Identifica as imagens que estão na pasta de origem, mas não na lista
            imagens_faltantes = arquivos_origem - lista_imagens

            # Copia as imagens faltantes para a pasta de destino
            for imagem in imagens_faltantes:
                origem = os.path.join(pasta_origem, imagem)
                destino = os.path.join(pasta_destino, imagem)
                if os.path.isfile(origem):  # Garante que é um arquivo, não uma subpasta
                    shutil.copy2(origem, destino)
                    imagens_copiadas.append(imagem)  # Adiciona à lista de imagens copiadas
                pbar.update(1)  # Atualiza a barra de progresso

    # Exporta a lista de imagens copiadas para uma planilha
    if imagens_copiadas:
        df = pd.DataFrame(imagens_copiadas, columns=["Imagens Copiadas"])
        df.to_excel(planilha_saida, index=False)
        print(f"Planilha gerada: {planilha_saida}")
    else:
        print("Nenhuma imagem foi copiada.")

    print("Processo concluído.")

# Exemplo de uso
lista_txt = r'D:\Lista_imagens_Aerotri_SI_16_HD04.txt'  # Arquivo .txt com os nomes das imagens
pastas_origem = [
    r'\\192.168.2.28\h\IMAGENS_RGBI_SI16',
    r'\\192.168.2.28\i\IMAGENS_RGBI_SI_16'
]  # Lista de pastas onde estão as imagens
pasta_destino = r'\\192.168.2.28\e\SI_16'  # Pasta onde as imagens serão copiadas
planilha_saida = r'D:\imagens_copiadas.xlsx'  # Caminho da planilha de saída

copiar_imagens_faltantes(lista_txt, pastas_origem, pasta_destino, planilha_saida)