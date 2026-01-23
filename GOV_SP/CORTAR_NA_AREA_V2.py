import os
import rasterio
from rasterio.mask import mask
from rasterio.transform import Affine
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import mapping, box
from tqdm import tqdm

def read_tfw(tfw_path):
    try:
        with open(tfw_path, 'r') as f:
            lines = [float(line.strip()) for line in f.readlines()]
            return Affine(lines[0], lines[2], lines[4],
                          lines[1], lines[3], lines[5])
    except Exception as e:
        print(f"Erro ao ler {tfw_path}: {e}")
        return None

def clip_images(input_folder, output_folder, polygon_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lê o shapefile e garante CRS EPSG:31983
    polygons = gpd.read_file(polygon_path).to_crs(epsg=31983)
    union_geom = polygons.unary_union  # mais rápido para interseção

    raster_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tif')]

    for filename in tqdm(raster_files, desc="Processando imagens"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        tfw_path = os.path.splitext(input_path)[0] + ".tfw"

        try:
            # Abre a imagem inicialmente para extrair info
            with rasterio.open(input_path) as src:
                crs = src.crs
                transform = src.transform
                width, height = src.width, src.height
                meta = src.meta.copy()

            # Se CRS for inválido ou transform vazio, tenta .tfw
            if crs is None or crs.to_epsg() is None or transform == Affine.identity():
                transform = read_tfw(tfw_path)
                if transform is None:
                    print(f"{filename}: sem transform válido. Pulando.")
                    continue
                crs = CRS.from_epsg(31983)

            # Calcula bounding box real da imagem
            bounds = rasterio.transform.array_bounds(height, width, transform)
            raster_geom = box(*bounds)

            # Verifica interseção com o polígono
            if not raster_geom.intersects(union_geom):
                print(f"{filename}: fora do polígono. Pulando.")
                continue

            # Reabre a imagem aplicando transform e crs corrigidos (se necessário)
            with rasterio.open(input_path) as src:
                image_data = src.read()
                count = src.count
                dtype = src.dtypes[0]

            # Ajusta metadados
            meta.update({
                "crs": crs,
                "transform": transform,
                "nodata": 255,
                "dtype": dtype,
                "count": count
            })

            # Recorta usando rasterio com transform manual
            with rasterio.open(input_path, 'r', crs=crs, transform=transform) as src:
                out_image, out_transform = mask(
                    src,
                    shapes=[mapping(geom) for geom in polygons.to_crs(crs).geometry],
                    crop=True,
                    nodata=255
                )

            # Atualiza para salvar
            meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            # Salva
            with rasterio.open(output_path, "w", **meta) as dest:
                dest.write(out_image)

            print(f"{filename}: recortada e salva.")

        except Exception as e:
            print(f"{filename}: Erro inesperado - {e}")

# Caminhos
input_folder = r'E:\RGB'
output_folder = r'E:\RGB\output'
polygon_path = r'D:\2212_GOV_SAO_PAULO\BLOCO.shp'

clip_images(input_folder, output_folder, polygon_path)
