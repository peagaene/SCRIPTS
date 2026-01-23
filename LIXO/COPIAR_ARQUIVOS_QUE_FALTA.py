import os
import shutil
from tqdm import tqdm
import time

def listar_arquivos_unicos(pastas_grupo1, pastas_grupo2, pasta_destino):
    # Função para listar apenas arquivos em uma pasta
    def listar_arquivos(pasta):
        try:
            return {arquivo for arquivo in os.listdir(pasta) if os.path.isfile(os.path.join(pasta, arquivo))}
        except FileNotFoundError:
            print(f"A pasta {pasta} não foi encontrada.")
            return set()
        except PermissionError:
            print(f"Sem permissão para acessar a pasta {pasta}.")
            return set()

    # Obter conjuntos de arquivos para cada grupo
    arquivos_grupo1 = set()
    arquivos_grupo2 = set()

    # Adicionar arquivos de todas as pastas do grupo 1
    for pasta in pastas_grupo1:
        arquivos_grupo1.update(listar_arquivos(pasta))
    
    # Adicionar arquivos de todas as pastas do grupo 2
    for pasta in pastas_grupo2:
        arquivos_grupo2.update(listar_arquivos(pasta))
    
    # Encontrar diferença simétrica de arquivos entre os dois grupos
    arquivos_unicos = arquivos_grupo1.symmetric_difference(arquivos_grupo2)
    
    # Criar pasta de destino, se não existir
    os.makedirs(pasta_destino, exist_ok=True)
    
    # Copiar arquivos únicos para a pasta de destino com barra de progresso
    print(f"Copiando {len(arquivos_unicos)} arquivos únicos para {pasta_destino}...")
    for arquivo in tqdm(arquivos_unicos, desc="Progresso", unit="arquivo", colour="blue"):
        for pasta in pastas_grupo1 + pastas_grupo2:  # Procurar o arquivo em todas as pastas
            caminho_origem = os.path.join(pasta, arquivo)
            if os.path.exists(caminho_origem):  # Certifique-se de que o arquivo existe
                caminho_destino = os.path.join(pasta_destino, arquivo)
                shutil.copy2(caminho_origem, caminho_destino)  # Copiar mantendo metadados
                break  # Copiar o arquivo uma única vez
            time.sleep(0.01)  # Pequena pausa para suavizar a barra de progresso
    
    return list(arquivos_unicos)

# Exemplo de uso
pastas_grupo1 = [r'\\192.168.2.27\h\SP22_BE_13_03052024_HD03\SI_09\ORTO\TIFF\RGB']
pastas_grupo2 = [r'\\192.168.2.27\g\SI_09_ORTO\ORTO\TIFF\ir']
pasta_destino = r'\\192.168.2.27\g\SI_09_ORTO\ORTO\TIFF\rgb'  # Defina a pasta onde os arquivos únicos serão copiados

# Gerar lista de arquivos únicos e copiá-los para a pasta de destino
arquivos_unicos = listar_arquivos_unicos(pastas_grupo1, pastas_grupo2, pasta_destino)

# Exibir resultados
print(f"Arquivos únicos copiados para {pasta_destino}: {arquivos_unicos}")
