import os
import subprocess
import shutil
import time
from tqdm import tqdm  # Biblioteca para a barra de progresso

# Defina o caminho da pasta que contém os arquivos que você deseja compactar
pasta_arquivos = r'G:\STRIPALIGIN'

# Defina o caminho da pasta onde os arquivos compactados serão armazenados
pasta_zipada = r'G:\STRIPALIGIN'
os.makedirs(pasta_zipada, exist_ok=True)

# Caminho para o executável do rar
caminho_rar = r"C:\Program Files\WinRAR\rar.exe"  # Substitua pelo caminho correto do seu executável

def espaco_disponivel(pasta):
    """Retorna o espaço disponível no disco onde a pasta está localizada, em bytes."""
    total, usado, livre = shutil.disk_usage(pasta)
    return livre

def tamanho_arquivo(caminho):
    """Retorna o tamanho do arquivo em bytes."""
    return os.path.getsize(caminho)

# Obtenha a lista de arquivos na pasta e inicie a barra de progresso
arquivos = os.listdir(pasta_arquivos)
total_arquivos = len(arquivos)

# Inicia a barra de progresso
with tqdm(total=total_arquivos, desc="Compactando arquivos", unit="arquivo") as pbar:
    for nome_arquivo in arquivos:
        caminho_completo_arquivo = os.path.join(pasta_arquivos, nome_arquivo)

        # Verifique se o arquivo já é um .rar ou se um .rar correspondente já existe na pasta de destino
        if nome_arquivo.lower().endswith('.rar'):
            print(f"O arquivo {nome_arquivo} já é um arquivo .rar, ignorando...")
            pbar.update(1)  # Atualiza a barra de progresso
            continue

        caminho_zipado = os.path.join(pasta_zipada, f"{nome_arquivo}.rar")
        if os.path.exists(caminho_zipado):
            print(f"O arquivo {caminho_zipado} já existe, ignorando...")
            pbar.update(1)  # Atualiza a barra de progresso
            continue
        
        # Verifique se é um arquivo (e não um diretório)
        if os.path.isfile(caminho_completo_arquivo):
            # Estime o espaço necessário (geralmente, o tamanho original do arquivo é uma boa aproximação)
            espaco_necessario = tamanho_arquivo(caminho_completo_arquivo)

            # Verifique se há espaço suficiente no disco
            while espaco_disponivel(pasta_zipada) < espaco_necessario:
                print(f"Espaço insuficiente para compactar {nome_arquivo}. Aguardando liberação de espaço...")
                time.sleep(60)  # Espera 60 segundos antes de verificar novamente
            
            # Comando para criar o arquivo rar
            comando = [caminho_rar, 'a', '-ep', caminho_zipado, caminho_completo_arquivo]
            
            # Execute o comando
            subprocess.run(comando, check=True)
            
            print(f"Arquivo {nome_arquivo} compactado com sucesso!")

        pbar.update(1)  # Atualiza a barra de progresso

# Exibe o espaço disponível no disco após a compactação
espaco_restante = espaco_disponivel(pasta_zipada) / (1024 * 1024 * 1024)  # Converte para GB
print(f"Espaço disponível no disco após a compactação: {espaco_restante:.2f} GB")
