import os
import geopandas as gpd
from shutil import copy2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def move_image(input_path, output_path):
    try:
        # Move o arquivo para a pasta de saída
        copy2(input_path, output_path)
        print(f"Imagem movida para {output_path}.")
    except Exception as e:
        print(f"Erro ao mover a imagem: {e}")

def move_images(input_folder, output_folder, articulation_path):
    # Verifica se a pasta de saída existe, caso contrário, cria-a
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lê o shapefile de articulação
    articulation = gpd.read_file(articulation_path)

    # Filtra os polígonos da articulação com área menor que 1.000.000 m²
    small_polygons = articulation[articulation.geometry.area < 1_000_000]

    # Gera uma lista de nomes de arquivos associados aos polígonos pequenos
    small_polygon_files = small_polygons['Nome_img'].apply(lambda x: x if x.endswith('.tif') else f"{x}.tif").tolist()

    # Lista os arquivos na pasta de entrada
    raster_files = [f for f in os.listdir(input_folder) if f.endswith(('.tif', '.TIF'))]

    # Filtra os arquivos que correspondem aos nomes no shapefile de articulação
    raster_files_to_move = [f for f in raster_files if f in small_polygon_files]

    # Prepara caminhos completos para mover
    tasks = [
        (os.path.join(input_folder, filename), os.path.join(output_folder, filename))
        for filename in raster_files_to_move
    ]

    # Usa ThreadPoolExecutor para paralelizar o processo
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda args: move_image(*args), tasks), total=len(tasks), desc="Movendo imagens", leave=True))

# Exemplo de uso
input_folder = r'K:\SP22_BE_13_03052024_HD03\SI_04\RGB'
output_folder = r'K:\SP22_BE_13_03052024_HD03\SI_04\RGB\output'
articulation_path = r'D:\ARTICULACAO_CORTADA_NA_AREA.shp'

move_images(input_folder, output_folder, articulation_path)
