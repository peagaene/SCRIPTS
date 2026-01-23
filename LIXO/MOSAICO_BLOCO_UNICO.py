import os
import rasterio as rio
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box, mapping
import geopandas as gpd
from tqdm import tqdm
import numpy as np
from rasterio.fill import fillnodata
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
import logging
import matplotlib.pyplot as plt

# Configurando logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Diretórios fixos
DIR_IMG = [r'J:\SI_01\RGB']
DIR_SHP_MOSAICO = r'D:\SI_01\SI_01.shp'
DIR_SAIDA = r'G:\ORTO\SI_01\1_RGB\1_GEOTIFF'
CRS_DESEJADO = "EPSG:31983"
DIR_CLIP_OUTPUT = r'J:\SI_01\RGB\output'
POLYGON_PATH = r'D:\BLOCO.shp'
ARTICULATION_PATH = r'D:\ARTICULACAO_CORTADA_NA_AREA.shp'

def visualizar_gaps(merged_data, filled_data, meta):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Máscara dos gaps antes do preenchimento
    ax[0].imshow(merged_data[0] == meta['nodata'], cmap="gray")
    ax[0].set_title("Gaps Antes do Preenchimento")

    # Máscara dos gaps após o preenchimento
    ax[1].imshow(filled_data[0] == meta['nodata'], cmap="gray")
    ax[1].set_title("Gaps Após o Preenchimento")

    plt.tight_layout()
    plt.show()

def gerar_tfw_prj(output_path, crs):
    """
    Gera os arquivos TFW e PRJ para a imagem GeoTIFF.
    """
    # Gera o arquivo TFW
    tfw_path = output_path.replace(".tif", ".tfw")
    with rio.open(output_path) as src:
        transform = src.transform
        with open(tfw_path, "w") as tfw:
            tfw.write(f"{transform.a}\n")  # Tamanho do pixel em X
            tfw.write(f"0.0\n")            # Rotação (linha -> coluna)
            tfw.write(f"0.0\n")            # Rotação (coluna -> linha)
            tfw.write(f"{transform.e}\n")  # Tamanho do pixel em Y
            tfw.write(f"{transform.c}\n")  # Coordenada X do canto superior esquerdo
            tfw.write(f"{transform.f}\n")  # Coordenada Y do canto superior esquerdo

    # Gera o arquivo PRJ
    prj_path = output_path.replace(".tif", ".prj")
    with open(prj_path, "w") as prj:
        prj.write(crs.to_wkt())

    logging.info(f"Arquivos auxiliares TFW e PRJ gerados: {tfw_path}, {prj_path}")


def listar_imagens(diretorios, clip_output_dir):
    """
    Lista as imagens TIFF únicas, priorizando aquelas no diretório de saída (clip_output_dir).
    """
    path_imagens = {}
    total_arquivos = sum(len(os.listdir(dir_img)) for dir_img in diretorios) + len(os.listdir(clip_output_dir))
    
    with tqdm(total=total_arquivos, desc="Lendo imagens", unit="img") as pbar:
        # Verifica as imagens cortadas (prioritárias)
        for nome_arquivo in os.listdir(clip_output_dir):
            if nome_arquivo.endswith(".tif"):
                path_imagens[nome_arquivo] = os.path.join(clip_output_dir, nome_arquivo)
            pbar.update(1)

        # Verifica imagens originais, adicionando apenas as que não estão no clip_output_dir
        for dir_img in diretorios:
            for nome_arquivo in os.listdir(dir_img):
                if nome_arquivo.endswith(".tif") and nome_arquivo not in path_imagens:
                    path_imagens[nome_arquivo] = os.path.join(dir_img, nome_arquivo)
                pbar.update(1)

    return list(path_imagens.values())

# Função: Ler Shapefiles e Reprojetar
def ler_shapefile_reprojetar(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Shapefile não encontrado: {caminho}")
    shp = gpd.read_file(caminho)
    if shp.crs != CRS_DESEJADO:
        logging.info(f"Reprojetando shapefile {caminho} para {CRS_DESEJADO}.")
        shp = shp.to_crs(CRS_DESEJADO)
    return shp

# Função: Reprojetar Imagem em Memória e Reduzir Bandas
def reprojetar_e_reduzir(src):
    if src.crs != CRS_DESEJADO:
        transform, width, height = calculate_default_transform(
            src.crs, CRS_DESEJADO, src.width, src.height, *src.bounds)
        meta = src.meta.copy()
        meta.update({"crs": CRS_DESEJADO, "transform": transform, "width": width, "height": height, "count": 3})
        data = np.zeros((3, height, width), dtype=src.dtypes[0])
        for i in range(1, 4):
            reproject(
                source=rio.band(src, i),
                destination=data[i-1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=CRS_DESEJADO,
                resampling=Resampling.nearest)
        return data, meta
    return src.read(out_shape=(3, src.height, src.width)), src.meta

def merge_imagens(imagens):
    rasters = []
    memfiles = []

    for img in tqdm(imagens, desc="Reprojetando e processando imagens", unit="img", leave=True):
        with rio.open(img) as src:
            data, meta = reprojetar_e_reduzir(src)
            memfile = MemoryFile()
            dataset = memfile.open(**meta)
            dataset.write(data)

            memfiles.append(memfile)
            rasters.append(dataset)

    try:
        logging.info("Realizando merge das imagens...")
        merged_data, merged_transform = merge(rasters, nodata=0)  # Força NoData=0 para o merge
        meta = rasters[0].meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": merged_data.shape[1],
            "width": merged_data.shape[2],
            "transform": merged_transform,
            "nodata": 0  # Define nodata como 0
        })

        # Preenche os gaps (pixels nodata) com base na vizinhança
        logging.info("Preenchendo gaps entre imagens...")
        filled_data = np.zeros_like(merged_data, dtype=merged_data.dtype)
        for band in range(merged_data.shape[0]):  # Iterar pelas bandas
            mask = merged_data[band] == meta['nodata']  # Cria uma máscara para nodata
            filled_data[band] = fillnodata(merged_data[band], mask=mask, max_search_distance=10, smoothing_iterations=2)

    finally:
        for dataset in rasters:
            dataset.close()
        for memfile in memfiles:
            memfile.close()
    visualizar_gaps(merged_data, filled_data, meta)
    return filled_data, meta

# Função: Clip Imagens
def clip_images(input_folder, output_folder, polygon_path, articulation_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    polygons = gpd.read_file(polygon_path)
    articulation = gpd.read_file(articulation_path)
    small_polygons = articulation[articulation.geometry.area < 1_000_000]

    small_polygon_files = small_polygons['Nome_img'].apply(lambda x: x if x.endswith('.tif') else f"{x}.tif").tolist()
    raster_files = [f for f in os.listdir(input_folder) if f.endswith(('.tif', '.TIF'))]
    raster_files_to_process = [f for f in raster_files if f in small_polygon_files]

    for filename in tqdm(raster_files_to_process, desc="Cortando imagens", leave=True):
        output_path = os.path.join(output_folder, filename)

        # Verifica se a imagem já existe no caminho de saída
        if os.path.exists(output_path):
            logging.info(f"Imagem {filename} já existe. Pulando...")
            continue  # Pula para a próxima iteração

        input_path = os.path.join(input_folder, filename)

        with rio.open(input_path) as src:
            try:
                polygons_reproj = polygons.to_crs(src.crs)
                smoothed_polygons = polygons_reproj.buffer(-0.01).buffer(0.01)
                out_image, out_transform = mask(src, [mapping(geom) for geom in smoothed_polygons], crop=True, nodata=255)
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": 255
                })
                with rio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)

                logging.info(f"Imagem {filename} recortada e salva em {output_path}.")
            except Exception as e:
                logging.error(f"Erro ao processar {filename}: {e}")
                
# Função: Cortar Final no Shapefile Articulação Mosaico
def cortar_final(data, meta, shape_mosaico, output_path):
    geometria_mosaico = [feature['geometry'] for feature in shape_mosaico.__geo_interface__['features']]
    
    # Abrir o array NumPy como um dataset temporário
    with MemoryFile() as memfile:
        with memfile.open(**meta) as temp_dataset:
            temp_dataset.write(data)
            
            # Aplicar o mask diretamente no dataset temporário
            out_image, out_transform = mask(temp_dataset, geometria_mosaico, crop=True)
    
    # Atualizar os metadados após o corte
    meta.update({
        "transform": out_transform,
        "height": out_image.shape[1],
        "width": out_image.shape[2]
    })

    # Salvar resultado final
    with rio.open(output_path, "w", **meta) as dest:
        for i in range(1, out_image.shape[0] + 1):
            dest.write(out_image[i-1], i)
    
    gerar_tfw_prj(output_path, meta['crs'])
    logging.info(f"Ortomosaico salvo em {output_path}")

def processar_ortomosaicos():
    logging.info("Iniciando processamento de ortomosaicos...")
    
    # Listar imagens, priorizando as cortadas
    imagens = listar_imagens(DIR_IMG, DIR_CLIP_OUTPUT)
    shp_mosaico = ler_shapefile_reprojetar(DIR_SHP_MOSAICO)

    # Chama a função clip_images antes do processamento principal
    logging.info("Cortando imagens pequenas com base no shapefile...")
    clip_images(DIR_IMG[0], DIR_CLIP_OUTPUT, POLYGON_PATH, ARTICULATION_PATH)

    with tqdm(total=len(shp_mosaico['NAME'].unique()), desc="Processando mosaicos", unit="mosaico", leave=True) as pbar:
        for nome_mosaico in shp_mosaico['NAME'].unique():
            logging.info(f"Processando mosaico: {nome_mosaico}")
            mosaico_geom = shp_mosaico[shp_mosaico['NAME'] == nome_mosaico]
            output_path = os.path.join(DIR_SAIDA, f"{nome_mosaico}_ORTO_RGB.tif")

            # Filtra as imagens que intersectam com o mosaico
            imagens_validas = [img for img in imagens if box(*rio.open(img).bounds).intersects(mosaico_geom.geometry.union_all())]
            
            # Realiza merge e corte final
            merged_data, meta = merge_imagens(imagens_validas)
            cortar_final(merged_data, meta, mosaico_geom, output_path)
            pbar.update(1)

    logging.info("Processamento concluído com sucesso!")

# Entrada do Script
if __name__ == "__main__":
    processar_ortomosaicos()
    
