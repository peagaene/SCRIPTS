import os
import shutil
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
from tqdm import tqdm

def normalize_filename(name):
    """Remove zeros à esquerda do nome do arquivo para correspondência correta."""
    return name.lstrip("0").lower()

def clip_images(input_folder, output_folder, polygon_path, articulation_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    not_found_folder = os.path.join(output_folder, "nao_encontradas")
    if not os.path.exists(not_found_folder):
        os.makedirs(not_found_folder)
    
    polygons = gpd.read_file(polygon_path)
    articulation = gpd.read_file(articulation_path)
    
    # Normaliza os nomes dos arquivos na articulação
    all_polygon_files = set(articulation['Nome_img'].apply(lambda x: normalize_filename(x if x.endswith('.tif') else f"{x}.tif")))
    small_polygons = articulation[articulation.geometry.area < 1_000_000]
    small_polygon_files = set(small_polygons['Nome_img'].apply(lambda x: normalize_filename(x if x.endswith('.tif') else f"{x}.tif")))
    
    # Lista os arquivos na pasta de entrada
    raster_files = [f for f in os.listdir(input_folder) if f.endswith(('.tif', '.TIF'))]   
    
    for filename in tqdm(raster_files, desc="Processando imagens", leave=True):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        normalized_filename = normalize_filename(filename)
        
        if normalized_filename in small_polygon_files:
            try:
                with rasterio.open(input_path) as src:
                    # Verifica se o raster tem projeção
                    crs = src.crs if src.crs is not None else articulation.crs
                    
                    # Reprojeta o polígono para coincidir com o CRS do raster
                    polygons_reproj = polygons.to_crs(crs)
                    
                    # Suaviza a geometria do polígono usando buffer negativo e positivo
                    smoothed_polygons = polygons_reproj.buffer(-0.01).buffer(0.01)
                    
                    # Aplica o mask com os polígonos suavizados
                    out_image, out_transform = mask(src, [mapping(geom) for geom in smoothed_polygons], crop=True, nodata=255)
                    
                    # Atualiza os metadados
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "nodata": 255,
                        "crs": crs
                    })
                    
                    # Salva a imagem recortada
                    with rasterio.open(output_path, "w", **out_meta) as dest:
                        dest.write(out_image)
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
        elif normalized_filename not in all_polygon_files:
            # Move a imagem para a pasta de não encontradas
            shutil.move(input_path, os.path.join(not_found_folder, filename))
            print(f"Imagem {filename} não encontrada no shapefile de articulação e movida para {not_found_folder}.")
    
input_folder = r'\\192.168.2.26\e\SI_13\RGB\divisa'
output_folder = r'\\192.168.2.26\e\SI_13\RGB\divisa\BORDA'
polygon_path = r'\\192.168.2.29\d\BLOCO.shp'
articulation_path = r'\\192.168.2.29\d\ARTICULACAO_CORTADA_NA_AREA.shp'

clip_images(input_folder, output_folder, polygon_path, articulation_path)
