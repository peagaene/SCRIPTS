import os
import geopandas as gpd
from osgeo import gdal, osr
from tqdm import tqdm

def georeference_images(shapefile_path, image_folder, name_column):
    # Carregar o shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Verificar se a coluna existe
    if name_column not in gdf.columns:
        raise ValueError(f"A coluna '{name_column}' não existe no shapefile.")
    
    # Criar um dicionário para mapear nomes a geometrias
    image_georef_data = {}
    
    for _, row in gdf.iterrows():
        image_name = str(row[name_column])
        geometry = row.geometry
        epsg = gdf.crs.to_epsg()  # Obtém o código EPSG do shapefile
        
        if geometry is None or geometry.is_empty:
            continue
        
        # Pega a extensão (bounding box) do polígono
        minx, miny, maxx, maxy = geometry.bounds
        
        image_georef_data[image_name] = {
            "minx": minx, "maxx": maxx, "miny": miny, "maxy": maxy, "epsg": epsg
        }
    
    # Processar cada imagem na pasta
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.tif', '.TIF'))]
    
    for img_file in tqdm(image_files, desc="Georreferenciando imagens", leave=True):
        img_name, ext = os.path.splitext(img_file)
        
        if img_name in image_georef_data:
            img_path = os.path.join(image_folder, img_file)
            
            # Abrir a imagem com GDAL
            dataset = gdal.Open(img_path, gdal.GA_Update)
            if dataset is None:
                print(f"Erro ao abrir a imagem: {img_file}")
                continue
            
            # Verifica se a imagem já possui georreferenciamento
            if dataset.GetGeoTransform() != (0, 1, 0, 0, 0, 1):
                print(f"Imagem {img_file} já georreferenciada. Pulando...")
                continue
            
            # Obtém os parâmetros geoespaciais
            georef = image_georef_data[img_name]
            minx, maxx, miny, maxy, epsg = georef.values()
            
            # Calcular resolução da imagem (assume-se que o pixel é quadrado)
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            pixel_size_x = (maxx - minx) / width
            pixel_size_y = (maxy - miny) / height
            
            # Criar transformação geoespacial
            geotransform = [minx, pixel_size_x, 0, maxy, 0, -pixel_size_y]
            dataset.SetGeoTransform(geotransform)
            
            # Definir projeção
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsg)
            dataset.SetProjection(srs.ExportToWkt())
            
            # Salvar e fechar
            dataset.FlushCache()
            dataset = None
            print(f"Imagem {img_file} georreferenciada com sucesso!")

# Exemplo de uso:
shapefile_path = r"E:\\2212_GOV_SAO_PAULO\\VOO\\BLOCOS ARTICULACAO\\ARTICULACAO_DEMO.shp"
image_folder = r"G:\\SI_04"
name_column = "Nome_img"

georeference_images(shapefile_path, image_folder, name_column)
