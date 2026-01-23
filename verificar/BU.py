import os
import shutil
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box, mapping  # ✅ adicione esta linha
from tqdm import tqdm

# === CONFIGURAÇÕES ===
PASTA_IMAGENS = r"\\192.168.2.27\i\ORTOMOSAICOS\10\RGB\TIFF"
SHAPE_MASK = r"D:\2212_GOV_SAO_PAULO\BLOCO.shp"
PASTA_SAIDA = r"\\192.168.2.27\i\ORTOMOSAICOS\10\RGB\TIFF\SAIDA"

os.makedirs(PASTA_SAIDA, exist_ok=True)

# === CARREGAR MÁSCARA ===
gdf = gpd.read_file(SHAPE_MASK)
gdf_mask_union = gdf.unary_union  # União de todos os polígonos
geoms = [mapping(gdf_mask_union)]

# === PERCORRER IMAGENS ===
for nome_arquivo in tqdm(os.listdir(PASTA_IMAGENS)):
    if not nome_arquivo.lower().endswith(".tif"):
        continue

    caminho_img = os.path.join(PASTA_IMAGENS, nome_arquivo)
    nome_base = os.path.splitext(nome_arquivo)[0]
    caminho_saida_recorte = os.path.join(PASTA_SAIDA, nome_base + "_recortado.tif")

    # Se já foi processada, pular
    if os.path.exists(caminho_saida_recorte):
        print(f"⏭️ {nome_arquivo} já processada. Pulando.")
        continue

    with rasterio.open(caminho_img) as src:
        bbox_raster = box(*src.bounds)

        # Se não houver interseção, ignorar
        if not gdf_mask_union.intersects(bbox_raster):
            print(f"⏭️ {nome_arquivo} não cruza com o shapefile. Ignorado.")
            continue

        # Se a imagem estiver totalmente dentro, pular
        if gdf_mask_union.contains(bbox_raster):
            print(f"✅ {nome_arquivo} está totalmente dentro do polígono. Pulando (sem copiar).")
            continue

        # Caso contrário, recortar
        out_image, out_transform = mask(
            src,
            geoms,
            crop=True,
            nodata=src.nodata,
            filled=True
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": src.nodata,
            "compress": "DEFLATE"
        })

        with rasterio.open(caminho_saida_recorte, "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"✂️ {nome_arquivo} recortado com sucesso.")

print("🏁 Processamento concluído.")