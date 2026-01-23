#pip install rasterio shapely geopandas tqdm 

import os
import sys
import rasterio as rio
from rasterio.merge import merge
from rasterio.enums import Resampling
from shapely.geometry import *
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, Resampling
import geopandas as gpd
from tqdm import tqdm
import warnings

#Remover os avisos
warnings.filterwarnings('ignore', category = DeprecationWarning)

#Area
nome_area = 'SI_06_ORTOMOSAICO'

#Fator de compressão
compression_factor = 10

#Diretório(s) com as imagens RGBI
dir_img =[r'\\192.168.2.26\g\OUT\RGB'] #r'\\192.168.2.28\e\TPHOTO\OUT'

#Dir temporario
dir_temp = r'D:\TESTE\TIF_SI_13\RGB'

#Diretorio dos SHP (Ortofoto e Bloco)
dir_shp_mosaico = r'D:\TESTE\shp\SI_13_ORTOMOSAICO_RGB_CONTINENTE.shp'
dir_shp_bloco = r'D:\TESTE\shp\BLOCO_SI_13.shp'

#Diretorio onde serão salvos as imagens RBB e IRG
dir_rgb = r'D:\ORTOMOSAICO\SI_13\1_RGB' #Usar \\ ou / caso haja numero no início do nome da pasta
#dir_irb = r'\\192.168.2.28\g\5_ORTOMOSAICOS\SI_13\2_IR'

#Criar uma lista com o nome das ortofotos que estão dentro do(s) diretório(s)
path_imagens = []
nomes_imagens = []

for dir_img_i in dir_img:
    nomes_imagens_i = []
    for nome_arquivo in os.listdir(dir_img_i):
      if nome_arquivo.endswith(".tif") and nome_arquivo not in nomes_imagens: #Evitar que imagens repetidas sejam adicionadas
        nomes_imagens_i.append(nome_arquivo)
        nomes_imagens.append(nome_arquivo)
        
    #print (nomes_imagens_i)
    
    num_img = len(nomes_imagens_i)
    
    for i in range(num_img):
        path_i = f'{dir_img_i}\{nomes_imagens_i[i]}'
        path_imagens.append(path_i)
    
#print(path_imagens)

#Leitura dos shp com o a área dos ortomosaicos e do Bloco
shp_mosaico = gpd.read_file(dir_shp_mosaico)
shp_bloco = gpd.read_file(dir_shp_bloco)

#Criar uma lista com nome dos ortomosaicos que estão dentro do shp
nomes_mosaico = shp_mosaico['NAME'].tolist()

#Criar um SHP com a BBOX das ortofotos
'''extensao = []
orto_nomes = []

for i in tqdm(range(len(path_imagens)), desc = "Calculando Bbox das ortofotos"):
    with rio.open(path_imagens[i]) as src:
        img = src.read()   
        bound = src.bounds #Extrair os vértices dos rasters

    bbox = box(*bound) #Criar um bbox com as coordenadas dos vértices do raster
    extensao.append(bbox)
    orto_nomes.append(nomes_imagens[i])
   
shp_img = gpd.GeoDataFrame(geometry = extensao, crs='epsg:31983')
shp_img['Nome_Orto'] = orto_nomes

shp_img.to_file(r'D:\TESTE\shp\overlay_bloco13.shp')'''


shp_img = gpd.read_file(r'D:\TESTE\shp\overlay_bloco13.shp')

#Seleção das ortofotos que estão nos respectivos ortomosaicos
shp_new = gpd.overlay(shp_img, shp_mosaico, how='intersection' )

#print(nomes_mosaico)
#Criar os ortomosaicos e salvar em suas respectivas pastas

arquivo_mosaico = []
for nome_arquivo in os.listdir(f'{dir_temp}'): #dir_rgb
  if nome_arquivo.endswith(".tif"): # 
    arquivo_mosaico.append(nome_arquivo)

if not arquivo_mosaico:
    for i in tqdm(range(len(nomes_mosaico)), desc = "Gerando os Ortomosaicos"):

        #Selecionar as ortofotos associadas aos ortomosaicos
        orto_mos_filt_1 = shp_new[shp_new['NAME'] == nomes_mosaico[i]]
        
        #Remover fotos duplicadas da lista
        orto_mos_filt = orto_mos_filt_1.drop_duplicates(subset='geometry')

        # Verificar se há ortofotos associadas
        if orto_mos_filt.empty:
            #print(f"Não foram encontradas ortofotos para o mosaico {nomes_mosaico[i]}")
            continue

        #Armazenar o nome das ortofotos em uma lista
        orto_mos = orto_mos_filt['Nome_Orto'].tolist()
        
        #Merge na lista de ortofotos associadas ao ortomosaico
        dir_orto_mos = []
        
        # !!! ATENÇÃO !!! Adicionar na lista somente os arquivos que existem no diretório 
        for dir_img_j in dir_img:
            for j in range(len(orto_mos)):
                dir_orto_mos_i = f'{dir_img_j}\{orto_mos[j]}'
                
                if os.path.exists(dir_orto_mos_i):
                    dir_orto_mos.append(dir_orto_mos_i)

        raster_to_mosaic = []
        for j in dir_orto_mos:
            raster_to_mosaic.append(rio.open(j))

        #Merge raster images
        mosaic, output = merge(raster_to_mosaic)

        # Filtrar a feiçaõ do shp do bloco correspondente ao mosaico
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
                            "crs": "EPSG:31983",
                            "transform": output}) 
                              
            # Definir o caminho de saída do mosaico
            output_rgb_pasta_i = f'{dir_temp}'
            output_rgb_tif = f'{output_rgb_pasta_i}\{nomes_mosaico[i]}.tif'
            os.makedirs(os.path.dirname(output_rgb_pasta_i), exist_ok=True)

            # Salvar o mosaico resultante em um arquivo raster
            with rio.open(output_rgb_tif, 'w', **clip_meta) as clipped_dest:
                clipped_dest.write(mosaic) 
                #original: clipped_dest.write(mosaic)
        else:
            print(f"Não foi encontrado bloco correspondente ao mosaico {nomes_mosaico[i]}.")

#Recortar o shp do mosaico utilizando o bloco como máscara
shp_mosaico_rec = gpd.overlay(shp_mosaico, shp_bloco, how='intersection' )

#Recortar o ortomosaico resultante com o shp do bloco
output_rgb_pasta_i = f'{dir_temp}'#alterar aqui depois

path_mosaico = []
for nome_arquivo in os.listdir(output_rgb_pasta_i): 
  if nome_arquivo.endswith(".tif"):
    path_mosaico.append(nome_arquivo)

#print(path_mosaico)
#print(path_mosaico)

for i in tqdm(range(len(path_mosaico)), desc = "Recortando e Salvando os Ortomosaicos"):

    shp_mosaico_it = shp_mosaico_rec[shp_mosaico_rec['NAME'] == path_mosaico[i][:-4]]

    with rio.open(f'{output_rgb_pasta_i}\{path_mosaico[i]}') as src:
        img = src.read()
        out_img, out_transform = mask(src, shp_mosaico_it['geometry'], crop = True)
        clip_meta = src.meta
        
        #Parâmetros dos arquivos PRJ
        transform = src.transform
        
        ta = transform.a
        tb = transform.b
        tc = transform.c
        td = transform.d
        te = transform.e
        tf = transform.f
        
        tfw_content = f"{ta}\n0.0\n0.0\n{te}\n{tc+ta*.5}\n{tf+te*.5}"
        
        #Parâmetros do arquivos PRJ
        crs_s = src.crs.from_epsg("31983")
        
    # Selecionar apenas as bandas 1, 2 e 3 (RGB)
    out_img_rgb = out_img[[0,1,2], :, :]

    # Selecionar apenas as bandas 4, 3 e 2 (IRG)
    #out_img_irg = out_img[[3,2,1], :, :]

    #>> Salvar o arquivo final em GTiff <<
    
    # >> Salvar o arquivo final RGB/Gtiff <<
    # Iterar sobre cada banda
 
    clip_meta.update({"count": 3,
                        "dtype": 'uint8',
                        "height": out_img_rgb.shape[1],
                        "width": out_img_rgb.shape[2],
                        "driver": "GTiff",
                        "nodata": 255, #None
                        "crs": "EPSG:31983",
                        "transform": out_transform})

    # Definir o caminho de saída do mosaico
    output_rgb_pasta = f'{dir_rgb}\\1_GEOTIFF'
    output_rgb_tif = f'{output_rgb_pasta}\{path_mosaico[i][:-4]}_ORTO_RGB.tif'
    os.makedirs(os.path.dirname(output_rgb_pasta), exist_ok=True)

    # Salvar o mosaico resultante em um arquivo raster
    with rio.open(output_rgb_tif, 'w', **clip_meta) as clipped_dest:
        clipped_dest.write(out_img_rgb)
    
    # >> Salvar o arquivo final em RGB/JPEG2000 << 
    quality = 100 // 10
    clip_meta.update({"count": 3,
                        "dtype": 'uint8',
                        "height": out_img_rgb.shape[1],
                        "width": out_img_rgb.shape[2],
                        "driver": "JP2OpenJPEG",
                        "compression": "JPEG2000",
                        "QUALITY": quality,
                        "crs": "EPSG:31983",
                        "nodata": None,
                        "transform": out_transform})

    # Definir o caminho de saída do mosaico
    output_rgb_pasta = f'{dir_rgb}\\2_JPG2000'
    output_rgb_jp2 = f'{output_rgb_pasta}\{path_mosaico[i][:-4]}_ORTO_RGB.jp2'
    os.makedirs(os.path.dirname(output_rgb_pasta), exist_ok=True)

    # Salvar o mosaico resultante em um arquivo raster
    with rio.open(output_rgb_jp2, 'w', **clip_meta) as clipped_dest:
        clipped_dest.write(out_img_rgb)

    #Salvar o arquivo TFW (ESRI World File) e PRJ
    dir_tfw = [output_rgb_tif,output_rgb_jp2]
    
    for nome_dir_tfw in dir_tfw:
        
        nome_dir_tfw_i = f'{nome_dir_tfw[:-4]}.tfw'
        nome_dir_prj_i = f'{nome_dir_tfw[:-4]}.prj'
        
        with open(nome_dir_tfw_i, 'w') as tfw_file:
            tfw_file.write(tfw_content)
    
        with open(nome_dir_prj_i, 'w') as prj_file:
                prj_file.write(crs_s.to_wkt())
    
