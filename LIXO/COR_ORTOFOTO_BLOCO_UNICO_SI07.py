import cv2
import numpy as np
import rasterio
import os
from tqdm import tqdm  # Importar tqdm para a barra de progresso

# Função para ler a imagem raster e retornar as três primeiras bandas RGB se houver 4 bandas
def read_raster(image_path):
    with rasterio.open(image_path) as src:
        all_bands = src.read()
        num_bands = all_bands.shape[0]
        
        if num_bands == 5:
            image_rgb = all_bands[:3]
        elif num_bands == 4:
            image_rgb = all_bands[:3]
        elif num_bands == 3:
            image_rgb = all_bands
        else:
            raise ValueError("A imagem deve ter 3 ou 4 bandas.")

        image_rgb = np.moveaxis(image_rgb, 0, -1)
        profile = src.profile
        profile.update(count=3)

        return image_rgb, profile

# Função para escrever a imagem processada com metadados usando rasterio
def write_raster(image, profile, output_path):
    image = np.moveaxis(image, -1, 0)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(image)

# Função para aplicar correção gama
def apply_gamma_correction(image, gamma=1.0):
    image_normalized = image / 255.0
    image_corrected = np.power(image_normalized, gamma)
    image_corrected = np.clip(image_corrected * 255, 0, 255).astype(np.uint8)
    return image_corrected

# Função para ajustar o brilho da imagem
def adjust_brightness(image, beta=0):
    brightness_adjusted = cv2.convertScaleAbs(image, beta=beta)
    return brightness_adjusted

# Função para ajustar o contraste
def adjust_contrast(image, alpha=1.0):
    contrast_adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return contrast_adjusted

# Função para ajustar a intensidade (escala geral da imagem)
def adjust_intensity(image, factor=1.0):
    return np.clip(image * factor, 0, 255).astype(np.uint8)

# Função para ajustar cada canal individualmente (R, G, B)
def adjust_rgb(image, red=0, green=0, blue=0):
    image[:, :, 0] = np.clip(image[:, :, 0] + red, 0, 255)  # Banda R
    image[:, :, 1] = np.clip(image[:, :, 1] + green, 0, 255)  # Banda G
    image[:, :, 2] = np.clip(image[:, :, 2] + blue, 0, 255)  # Banda B
    return image

# Função para remover o efeito de neblina
def remove_haze(image, alpha=1.2):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.convertScaleAbs(yuv[:, :, 0], alpha=alpha, beta=0)
    haze_removed = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return haze_removed

# Função para aplicar o filtro gaussiano
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Função para ajustar a saturação da imagem
def adjust_saturation(image, saturation_factor=1.0):
    # Converter a imagem de BGR para HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Ajustar a saturação
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    
    # Converter de volta para BGR
    saturated_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return saturated_image

# Função principal para carregar a imagem, aplicar todas as alterações e salvar o resultado
def process_image(source_path, output_path, gamma=1.0, brightness_adjustment=0, contrast=1.0, intensity_factor=1.0, red=0, green=0, blue=0, saturation_factor=1.0, apply_fog_removal=False, apply_blur=False):
    source_image, source_profile = read_raster(source_path)

    # Aplicar as transformações solicitadas
    image = apply_gamma_correction(source_image, gamma)
    image = adjust_brightness(image, brightness_adjustment)
    image = adjust_contrast(image, contrast)
    image = adjust_intensity(image, intensity_factor)
    image = adjust_rgb(image, red, green, blue)
    
    # Aplicar ajuste de saturação
    image = adjust_saturation(image, saturation_factor)

    # Se necessário, aplicar remoção de neblina
    if apply_fog_removal:
        image = remove_haze(image)
    
    # Se necessário, aplicar filtro gaussiano
    if apply_blur:
        image = apply_gaussian_blur(image)

    # Salvar a imagem processada
    write_raster(image, source_profile, output_path)
    print(f"Imagem ajustada e salva em {output_path}")

# Listar os arquivos .tif ou .jp2 da pasta
formato = '.tif'
dir_fotos = r'\\192.168.2.29\d\SI_07\SI_06\SI07_IR'
dir_output = r'\\192.168.2.29\d\SI_07\SI_06\SI07_COR'

if not os.path.exists(dir_output):
    os.makedirs(dir_output)

lista_dir = [f'{dir_fotos}\\{filename}' for filename in os.listdir(dir_fotos) if filename.lower().endswith(formato)]
lista_nome = [filename for filename in os.listdir(dir_fotos) if filename.lower().endswith(formato)]

# Barra de progresso com tqdm
for i in tqdm(range(len(lista_dir)), desc="Processando imagens", unit="imagem"):
    output_file = f'{dir_output}\\{lista_nome[i]}'
    if os.path.exists(output_file):
        print(f"Imagem {output_file} já existe. Pulando.")
        continue  # Pular se a imagem já existir

    # Ajuste de parâmetros para cada imagem
    process_image(
        lista_dir[i], output_file,
        gamma=0.75,               # Ajuste de gama
        brightness_adjustment=10,  # Ajuste de brilho
        contrast=0.95,            # Ajuste de contraste
        intensity_factor=1,    # Ajuste de intensidade
        red=-28, green=-40, blue=-40, # Ajuste individual dos canais RGB
        saturation_factor=1.35,   # Aumentar saturação
        apply_fog_removal=False,  # Habilitar remoção de neblina
        apply_blur=False          # Habilitar filtro gaussiano
    )

