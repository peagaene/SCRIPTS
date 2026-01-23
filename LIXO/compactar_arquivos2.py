# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:35:08 2024

@author: sai
"""

import os
import subprocess

# Defina o caminho da pasta que contém os arquivos que você deseja compactar
pasta_arquivos = r'\\192.168.2.26\g\03 - DEVELOP\20240430_F1_12SEN309_NIR'

# Crie uma pasta para armazenar os arquivos compactados
pasta_zipada = r'\\192.168.2.26\g\03 - DEVELOP\20240430_F1_12SEN309_NIR'
os.makedirs(pasta_zipada, exist_ok=True)

# Caminho para o executável do rar
caminho_rar = r"C:\Program Files\WinRAR\rar.exe"  # Substitua pelo caminho correto do seu executável

# Percorra todos os arquivos na pasta
for nome_arquivo in os.listdir(pasta_arquivos):
    caminho_completo_arquivo = os.path.join(pasta_arquivos, nome_arquivo)
    
    # Verifique se é um arquivo (e não um diretório)
    if os.path.isfile(caminho_completo_arquivo):
        # Defina o caminho completo para o arquivo rar
        caminho_zipado = os.path.join(pasta_zipada, f"{nome_arquivo}.rar")
        
        # Comando para criar o arquivo rar
        comando = [caminho_rar, 'a', '-ep', caminho_zipado, caminho_completo_arquivo]
        
        # Execute o comando
        subprocess.run(comando, check=True)
        
        print(f"Arquivo {nome_arquivo} compactado com sucesso!")

print("Todos os arquivos foram compactados.")