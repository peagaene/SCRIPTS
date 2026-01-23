import os
import shutil
import geopandas as gpd

def ajustar_nome(nome):
    """
    Ajusta o nome removendo a extensão .tif (caso presente) e garantindo que o nome tenha 6 caracteres:
      - Se tiver 4 caracteres, adiciona '00' à esquerda.
      - Se tiver 5 caracteres, adiciona '0' à esquerda.
      - Se tiver 7 caracteres, remove o primeiro caractere.
      - Se tiver 6 caracteres, mantém inalterado.
    """
    # Remove a extensão ".tif", se presente (não diferencia maiúsculas/minúsculas)
    if nome.lower().endswith('.tif'):
        nome = nome[:-4]
    
    # Ajusta o nome para que tenha exatamente 6 caracteres
    if len(nome) == 4:
        nome = '00' + nome
    elif len(nome) == 5:
        nome = '0' + nome
    elif len(nome) == 7:
        nome = nome[1:]
    
    return nome

# Defina os caminhos para o shapefile e a pasta de imagens
caminho_shapefile = r'\\192.168.2.26\g\SI_09\ARTICULACAO_INTERNA.shp'
pasta_imagens = r'\\192.168.2.27\h\SI09\RGB'
pasta_divisa = os.path.join(pasta_imagens, 'divisa')

# Cria a pasta "divisa" caso ela não exista
if not os.path.exists(pasta_divisa):
    os.makedirs(pasta_divisa)

# Leitura do shapefile
gdf = gpd.read_file(caminho_shapefile)

# Extrai os nomes da coluna "nome_img", retirando valores nulos
nomes_shape = gdf['Nome_img'].dropna().unique()

# Ajusta os nomes lidos do shapefile para padronização (remove o ".tif" e ajusta os caracteres)
nomes_shape_ajustados = { ajustar_nome(nome) for nome in nomes_shape }

# Processa os arquivos na pasta de imagens
for arquivo in os.listdir(pasta_imagens):
    caminho_arquivo = os.path.join(pasta_imagens, arquivo)
    # Verifica se é um arquivo (evita pastas)
    if os.path.isfile(caminho_arquivo):
        # Se o arquivo possuir extensão, separa o nome e a extensão
        base, ext = os.path.splitext(arquivo)
        # Ajusta o nome do arquivo de acordo com as regras
        base_ajustada = ajustar_nome(base)
        
        # Define o novo nome, mantendo a extensão original (caso haja)
        novo_nome = base_ajustada + ext
        novo_caminho = os.path.join(pasta_imagens, novo_nome)
        
        # Renomeia o arquivo se o nome atual for diferente do nome ajustado
        if arquivo != novo_nome:
            os.rename(caminho_arquivo, novo_caminho)
            caminho_arquivo = novo_caminho  # Atualiza o caminho para o arquivo renomeado

        # Verifica se o nome ajustado está presente na lista do shapefile
        if base_ajustada not in nomes_shape_ajustados:
            # Se não estiver, move o arquivo para a pasta "divisa"
            destino = os.path.join(pasta_divisa, novo_nome)
            shutil.move(caminho_arquivo, destino)
            print(f"Arquivo {novo_nome} movido para 'divisa'.")
        else:
            print(f"Arquivo {novo_nome} está presente no shapefile.")
