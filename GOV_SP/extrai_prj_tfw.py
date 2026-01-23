import os
from osgeo import gdal

def create_prj_tfw_for_images(source_folder):
    """
    Cria arquivos .prj e .tfw para cada imagem em uma pasta com informações baseadas em seus metadados geoespaciais.

    :param source_folder: Caminho para a pasta que contém as imagens.
    """
    image_extensions = {".tif", ".tiff"}  # Foco em GeoTIFF ou imagens com dados geoespaciais

    for file_name in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file_name)

        if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in image_extensions:
            base_name = os.path.splitext(file_name)[0]
            tfw_file = os.path.join(source_folder, f"{base_name}.tfw")

            # Usa GDAL para abrir a imagem e obter informações geoespaciais
            dataset = gdal.Open(file_path)
            if not dataset:
                print(f"Erro ao abrir a imagem: {file_path}")
                continue

            geotransform = dataset.GetGeoTransform()

            if geotransform:
                # Extrai informações do GeoTransform para o arquivo .tfw
                tfw_content = (
                    f"{geotransform[1]}\n"  # Tamanho do pixel no eixo X
                    f"{geotransform[2]}\n"  # Rotação (geralmente 0)
                    f"{geotransform[4]}\n"  # Rotação (geralmente 0)
                    f"{geotransform[5]}\n"  # Tamanho do pixel no eixo Y (geralmente negativo)
                    f"{geotransform[0]}\n"  # Coordenada X do canto superior esquerdo
                    f"{geotransform[3]}\n"  # Coordenada Y do canto superior esquerdo
                )

                with open(tfw_file, "w") as tfw:
                    tfw.write(tfw_content)


    print(f"Arquivos .prj e .tfw criados para as imagens na pasta: {source_folder}")

# Configuração da pasta
source_folder = r"H:\4 - PROCESSAMENTO\04 ORTO\PONTAL\CORRECAO"  # Substitua pelo caminho da pasta de origem

# Executa a função
create_prj_tfw_for_images(source_folder)
