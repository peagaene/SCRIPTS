# Importar pacotes
import os
import sys
import rasterio as rio
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import *
from rasterio.warp import calculate_default_transform, Resampling
import geopandas as gpd
from tqdm import tqdm
import warnings

# Remover os avisos
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Área
nome_area = 'SI_10_ORTOMOSAICO'

# Fator de compressão
compression_factor = 10

# Diretório(s) com as imagens IR
dir_img = [r'\\192.168.2.26\e\SI_10\TPHOTO\IR']

# Diretório temporário
dir_temp = r'D:\SI_10\ir'

# Diretório dos SHP (Ortofoto e Bloco)
dir_shp_mosaico = r'D:\SI_10\MOSAICO_SI10.shp'
dir_shp_bloco = r'D:\SI_10\BLOCO.shp'

# Diretório onde serão salvos as imagens IRG
dir_irb = r'I:\5_ORTOMOSAICOS\SI_10\2_IR'

# Criar uma lista com o nome das ortofotos que estão dentro do(s) diretório(s)
path_imagens = []
nomes_imagens = []

for dir_img_i in dir_img:
    nomes_imagens_i = []
    for nome_arquivo in os.listdir(dir_img_i):
        if nome_arquivo.endswith(".tif") and nome_arquivo not in nomes_imagens:  # Evitar que imagens repetidas sejam adicionadas
            nomes_imagens_i.append(nome_arquivo)
            nomes_imagens.append(nome_arquivo)

    num_img = len(nomes_imagens_i)

    for i in range(num_img):
        path_i = f'{dir_img_i}\\{nomes_imagens_i[i]}'
        path_imagens.append(path_i)

# Leitura dos shp com o a área dos ortomosaicos e do Bloco
shp_mosaico = gpd.read_file(dir_shp_mosaico)
shp_bloco = gpd.read_file(dir_shp_bloco)

# Certificar que ambos os shapefiles estão no mesmo CRS
shp_bloco = shp_bloco.to_crs(shp_mosaico.crs)

# Criar uma lista com nome dos ortomosaicos que estão dentro do shp
nomes_mosaico = shp_mosaico['NAME'].tolist()

# Leitura do shapefile das imagens
shp_img = gpd.read_file(r'D:\SI_10\overlay_si10.shp')

# Certificar que o CRS do shapefile de imagem seja igual ao CRS do mosaico
shp_img = shp_img.to_crs(shp_mosaico.crs)

# Seleção das ortofotos que estão nos respectivos ortomosaicos
shp_new = gpd.overlay(shp_img, shp_mosaico, how='intersection')

# Criar os ortomosaicos e salvar em suas respectivas pastas
arquivo_mosaico = []
for nome_arquivo in os.listdir(f'{dir_temp}'):
    if nome_arquivo.endswith(".tif"):
        arquivo_mosaico.append(nome_arquivo)

if not arquivo_mosaico:
    for i in tqdm(range(len(nomes_mosaico)), desc="Gerando os Ortomosaicos"):

        # Selecionar as ortofotos associadas aos ortomosaicos
        orto_mos_filt_1 = shp_new[shp_new['NAME'] == nomes_mosaico[i]]

        # Remover fotos duplicadas da lista
        orto_mos_filt = orto_mos_filt_1.drop_duplicates(subset='geometry')

        # Verificar se há ortofotos associadas
        if orto_mos_filt.empty:
            print(f"Não foram encontradas ortofotos para o mosaico {nomes_mosaico[i]}")
            continue

        # Armazenar o nome das ortofotos em uma lista
        orto_mos = orto_mos_filt['Nome_img'].tolist()

        # Confirmar que os arquivos de imagem estão no diretório correto
        dir_orto_mos = []
        for dir_img_j in dir_img:
            for j in range(len(orto_mos)):
                dir_orto_mos_i = f'{dir_img_j}\\{orto_mos[j]}'
                if os.path.exists(dir_orto_mos_i):  # Verifica se o arquivo realmente existe
                    dir_orto_mos.append(dir_orto_mos_i)
                else:
                    print(f"A imagem {orto_mos[j]} não foi encontrada no diretório {dir_img_j}")

        # Verificar se a lista de ortofotos associadas não está vazia
        if not dir_orto_mos:
            print(f"Não foram encontradas imagens raster para o mosaico {nomes_mosaico[i]}")
            continue

        # Carregar as imagens raster
        raster_to_mosaic = []
        for j in dir_orto_mos:
            raster_to_mosaic.append(rio.open(j))

        # Merge raster images
        mosaic, output = merge(raster_to_mosaic)

        # Filtrar a feição do shp do bloco correspondente ao mosaico
        shp_bloco_mosaico = shp_mosaico[shp_mosaico['NAME'] == nomes_mosaico[i]]

        # Verificar se há um bloco correspondente ao mosaico
        if not shp_bloco_mosaico.empty:
            geom_bloco_mosaico = shp_bloco_mosaico.unary_union

            # Atualizar metadados para o mosaico recortado
            clip_meta = raster_to_mosaic[0].meta.copy()
            clip_meta.update({"count": 3,
                              "dtype": 'uint8',
                              "height": mosaic.shape[1],
                              "width": mosaic.shape[2],
                              "driver": "GTiff",
                              "nodata": None,
                              "crs": shp_mosaico.crs,
                              "transform": output})

            # Definir o caminho de saída do mosaico
            output_rgb_pasta_i = f'{dir_temp}'
            output_rgb_tif = f'{output_rgb_pasta_i}\\{nomes_mosaico[i]}.tif'
            os.makedirs(os.path.dirname(output_rgb_pasta_i), exist_ok=True)

            # Salvar o mosaico resultante em um arquivo raster
            with rio.open(output_rgb_tif, 'w', **clip_meta) as clipped_dest:
                clipped_dest.write(mosaic)
        else:
            print(f"Não foi encontrado bloco correspondente ao mosaico {nomes_mosaico[i]}.")

# Recortar o shp do mosaico utilizando o bloco como máscara
shp_mosaico_rec = gpd.overlay(shp_mosaico, shp_bloco, how='intersection')

# Recortar o ortomosaico resultante com o shp do bloco
output_rgb_pasta_i = f'{dir_temp}'

path_mosaico = []
for nome_arquivo in os.listdir(output_rgb_pasta_i):
    if nome_arquivo.endswith(".tif"):
        path_mosaico.append(nome_arquivo)

for i in tqdm(range(len(path_mosaico)), desc="Recortando e Salvando os Ortomosaicos"):

    shp_mosaico_it = shp_mosaico_rec[shp_mosaico_rec['NAME'] == path_mosaico[i][:-4]]

    with rio.open(f'{output_rgb_pasta_i}\\{path_mosaico[i]}') as src:
        img = src.read()
        out_img, out_transform = mask(src, shp_mosaico_it['geometry'], crop=True)
        clip_meta = src.meta

        # Parâmetros dos arquivos PRJ
        transform = src.transform
        ta = transform.a
        tb = transform.b
        tc = transform.c
        td = transform.d
        te = transform.e
        tf = transform.f

        tfw_content = f"{ta}\n0.0\n0.0\n{te}\n{tc+ta*.5}\n{tf+te*.5}"

        # Parâmetros do arquivos PRJ
        crs_s = src.crs

    # Selecionar apenas as bandas IRG
    out_img_irg = out_img[[0, 1, 2], :, :]

    # Salvar o arquivo final IRG/GeoTIFF
    clip_meta.update({"count": 3,
                      "dtype": 'uint8',
                      "height": out_img_irg.shape[1],
                      "width": out_img_irg.shape[2],
                      "driver": "GTiff",
                      "nodata": None,
                      "crs": crs_s,
                      "transform": out_transform})

    output_rgb = f'{dir_irb}\\{path_mosaico[i]}'
    with rio.open(output_rgb, 'w', **clip_meta) as dest:
        dest.write(out_img_irg)

print("Processo finalizado!")
