# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 18:06:03 2024

@author: sai
"""

import os
import shutil

# Defina os caminhos das pastas
pasta_A = r'K:\SP22_BE_13_03052024_HD03\SI_12\IR' # PASTA COM MAIS FOTOS
pasta_B = r'K:\SP22_BE_13_03052024_HD03\SI_12\RGB' #PASTA COM MENOS FOTOS
pasta_destino = r'K:\SP22_BE_13_03052024_HD03\SI_12\apagar_ir'

# Verifica se a pasta de destino existe, se não, cria
if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)

# Lista arquivos nas pastas A e B
arquivos_A = set(os.listdir(pasta_A))
arquivos_B = set(os.listdir(pasta_B))

# Identifica arquivos presentes em A que não estão em B
arquivos_para_mover = arquivos_A - arquivos_B

# Move os arquivos da pasta A para a pasta de destino
for arquivo in arquivos_para_mover:
    origem = os.path.join(pasta_A, arquivo)
    destino = os.path.join(pasta_destino, arquivo)
    if os.path.isfile(origem):  # Certifica que é um arquivo
        shutil.move(origem, destino)  # Move o arquivo para a pasta de destino
        print(f'Arquivo {arquivo} movido para {pasta_destino}')
