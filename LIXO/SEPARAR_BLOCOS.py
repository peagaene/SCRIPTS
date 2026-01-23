import os
import shutil
import geopandas as gpd

def match_shapefile_images(shapefile_path, image_folder, destination_folder):
    # Carregar shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Verificar se a coluna 'name' existe
    if 'NAME' not in gdf.columns:
        raise ValueError("A coluna 'name' não foi encontrada no shapefile.")
    
    # Obter os nomes únicos da coluna 'name'
    names = set(gdf['NAME'].astype(str))
    
    # Listar todas as imagens .tif na pasta
    tif_files = [f for f in os.listdir(image_folder) if f.endswith('.tjf')]
    
    # Criar um dicionário para armazenar correspondências
    matched_files = {}
    
    for name in names:
        expected_filename = f"_{name}.tif"
        if expected_filename in tif_files:
            matched_files[name] = expected_filename
            
            # Mover a imagem para a pasta de destino
            src_path = os.path.join(image_folder, expected_filename)
            dst_path = os.path.join(destination_folder, expected_filename)
            shutil.move(src_path, dst_path)
    
    return matched_files

# Exemplo de uso
shapefile_path = r"F:\SI_13\MOSAICO.shp"
image_folder = r"\\192.168.2.28\e\1314\rgb"
destination_folder = r"\\192.168.2.28\e\13\ir\tiff"

matched_images = match_shapefile_images(shapefile_path, image_folder, destination_folder)
print(matched_images)
