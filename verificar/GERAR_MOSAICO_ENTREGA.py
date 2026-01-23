import os
import rasterio as rio
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box, mapping
import geopandas as gpd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

# ===== CONFIGURAÇÕES =====
CONFIG = {
    "nome_area": "SI_04_ORTOMOSAICO",
    "compression_factor": 10,
    "dir_img": [
        r'\\192.168.2.27\f\si_04\RGB', r'\\192.168.2.27\f\si_04\RGB\divisa'
    ],
    "dir_shp_articulacoes": r'K:\SI_04\mosaico_interno.shp',
    "dir_shp_imagens": r'D:\ARTICULACAO_CORTADA_NA_AREA.shp',
    "epsg_code": 31983,
    "dir_output_rgb": r'D:\SI_04\RGB',
    "dir_output_jp2": r'D:\SI_04\JP2',
    "id_field": "NAME",
    "relatorio_saida": r'D:\SI_04\relatorio_exportacao.txt',
}

# ===== PREPARAÇÃO =====
os.makedirs(CONFIG["dir_output_rgb"], exist_ok=True)
os.makedirs(CONFIG["dir_output_jp2"], exist_ok=True)

# Ler shapefiles
shp_articulacoes = gpd.read_file(CONFIG["dir_shp_articulacoes"])
shp_imagens = gpd.read_file(CONFIG["dir_shp_imagens"])

# Coletar imagens disponíveis
path_imagens = []
nomes_imagens = []
for dir_img_i in CONFIG["dir_img"]:
    for nome_arquivo in os.listdir(dir_img_i):
        if nome_arquivo.lower().endswith(".tif"):
            path_i = os.path.join(dir_img_i, nome_arquivo)
            path_imagens.append(path_i)
            nomes_imagens.append(nome_arquivo)

# Criar BBOX de imagens disponíveis
extensao = []
for path_i in tqdm(path_imagens, desc="Calculando BBOX das ortofotos"):
    with rio.open(path_i) as src:
        bound = src.bounds
    extensao.append(box(*bound))

shp_img_disk = gpd.GeoDataFrame(geometry=extensao, crs=f"EPSG:{CONFIG['epsg_code']}")
shp_img_disk['Nome_Orto'] = nomes_imagens

# ===== VARIÁVEIS DE RELATÓRIO =====
exportacoes_sucesso = []
exportacoes_falha = []
imagens_nao_encontradas = set()

# ===== PROCESSAMENTO =====
for idx, row in tqdm(shp_articulacoes.iterrows(), total=len(shp_articulacoes), desc="Exportando articulações"):
    raster_to_mosaic = []  # 🔵 Inicializa aqui para garantir que exista
    try:
        polygon = row.geometry
        id_value = row.get(CONFIG["id_field"], idx)

        # Checar se já existem os arquivos de saída
        output_tif = os.path.join(CONFIG["dir_output_rgb"], f"{id_value}.tif")
        output_jp2 = os.path.join(CONFIG["dir_output_jp2"], f"{id_value}.jp2")

        if os.path.exists(output_tif) and os.path.exists(output_jp2):
            tqdm.write(f"\033[96m🟡 Imagens de {id_value} já existem, pulando...\033[0m")
            exportacoes_sucesso.append(id_value)
            continue

        tqdm.write(f"\033[94m🔵 Processando articulação {id_value}\033[0m")

        # Selecionar imagens que cruzam
        imagens_intersectam = shp_img_disk[shp_img_disk.intersects(polygon)]

        if imagens_intersectam.empty:
            msg = f"Nenhuma imagem encontrada para {id_value}"
            tqdm.write(f"\033[93m⚠️ {msg}\033[0m")
            exportacoes_falha.append((id_value, msg))
            continue

        imagens_necessarias = imagens_intersectam['Nome_Orto'].tolist()

        for nome_orto in imagens_necessarias:
            path_orto = next((p for p in path_imagens if os.path.basename(p) == nome_orto), None)
            if path_orto:
                raster_to_mosaic.append(rio.open(path_orto))
            else:
                imagens_nao_encontradas.add(nome_orto)

        if not raster_to_mosaic:
            msg = f"Nenhuma imagem aberta para {id_value} (faltando todas)"
            tqdm.write(f"\033[93m⚠️ {msg}\033[0m")
            exportacoes_falha.append((id_value, msg))
            continue

        # Criar mosaico virtual
        if len(raster_to_mosaic) == 1:
            src_vrt = raster_to_mosaic[0]
        else:
            mosaic, out_trans = merge(raster_to_mosaic, method='first')
            temp_meta = raster_to_mosaic[0].meta.copy()
            temp_meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "count": mosaic.shape[0],
                "driver": "GTiff",
                "crs": f"EPSG:{CONFIG['epsg_code']}",
                "nodata": None
            })
            temp_memfile = rio.io.MemoryFile()
            src_vrt = temp_memfile.open(**temp_meta)
            src_vrt.write(mosaic)

        # Cortar o polígono preenchendo nodata=255
        out_image, out_transform = mask(
            src_vrt,
            [mapping(polygon)],
            crop=True,
            filled=True,
            nodata=255
        )

        # Atualizar metadados
        out_meta = src_vrt.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "count": out_image.shape[0],
            "nodata": 255
        })

        # Salvar TIFF
        with rio.open(output_tif, 'w', **out_meta) as dest:
            dest.write(out_image)

        # Salvar JP2
        jp2_meta = out_meta.copy()
        jp2_meta.update({
            "driver": "JP2OpenJPEG",
            "quality": CONFIG["compression_factor"],
            "photometric": "RGB" if out_image.shape[0] >= 3 else "MINISBLACK",
            "nodata": 255
        })

        with rio.open(output_jp2, 'w', **jp2_meta) as dest_jp2:
            dest_jp2.write(out_image)

        tqdm.write(f"\033[92m✅ Articulação {id_value} exportada\033[0m")
        exportacoes_sucesso.append(id_value)

    except Exception as e:
        msg = str(e)
        tqdm.write(f"\033[91m❌ Erro ao processar {id_value}: {msg}\033[0m")
        exportacoes_falha.append((id_value, msg))

    finally:
        for dataset in raster_to_mosaic:
            try:
                dataset.close()
            except:
                pass

# ===== RELATÓRIO FINAL =====
with open(CONFIG["relatorio_saida"], 'w', encoding='utf-8') as relatorio:
    relatorio.write(f"Relatório de Exportação - {CONFIG['nome_area']}\n\n")
    relatorio.write(f"Total de articulações: {len(shp_articulacoes)}\n")
    relatorio.write(f"Exportações com sucesso: {len(exportacoes_sucesso)}\n")
    relatorio.write(f"Falhas: {len(exportacoes_falha)}\n\n")

    relatorio.write("=== Exportações Bem-sucedidas ===\n")
    for sucesso in exportacoes_sucesso:
        relatorio.write(f"- {sucesso}\n")

    relatorio.write("\n=== Exportações com Falha ===\n")
    for falha in exportacoes_falha:
        relatorio.write(f"- {falha[0]}: {falha[1]}\n")

    relatorio.write("\n=== Imagens Não Encontradas ===\n")
    for img in sorted(imagens_nao_encontradas):
        relatorio.write(f"- {img}\n")

print("\n\033[92m✅ Processamento concluído. Relatório salvo em:\033[0m", CONFIG["relatorio_saida"])
