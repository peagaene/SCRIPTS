import cv2
import numpy as np
import rasterio
import os
from tqdm import tqdm

# Função para ler a imagem raster e retornar as três primeiras bandas RGB se houver 4 bandas
def read_raster(image_path):
    with rasterio.open(image_path) as src:
        all_bands = src.read()
        num_bands = all_bands.shape[0]

        if num_bands == 4:
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

# Função para calcular a correspondência de cores entre duas imagens
def color_match(source):
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2Lab)

    source_mean, source_std = (np.array([[135.66057895],
                                         [127.02153941],
                                         [137.3524971]]), 
                               np.array([[54.08344986],
                                         [7.91640627],
                                         [8.639408]]))

    reference_mean, reference_std = (np.array([[105.04309],
                                               [125.019844],
                                               [129.477234]]), 
                                     np.array([[28.37144549],
                                               [3.73610442],
                                               [5.7678592]]))

    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    reference_mean = reference_mean.reshape(1, 1, 3)
    reference_std = reference_std.reshape(1, 1, 3)

    matched_lab = (source_lab - source_mean) * (reference_std / source_std) + reference_mean
    matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)

    matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_Lab2BGR)
    s_final = cv2.merge([source[:, :, 0], matched_bgr[:, :, 1], matched_bgr[:, :, 2]])  # [r, G, B]

    return matched_bgr

# Função para aplicar correção gama
def apply_gamma_correction(image, gamma=1.0):
    image_normalized = image / 255.0
    image_corrected = np.power(image_normalized, gamma)
    image_corrected = np.clip(image_corrected * 255, 0, 255).astype(np.uint8)
    return image_corrected

# Função para remover o efeito de neblina
def remove_haze(image, alpha=1.5):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.convertScaleAbs(yuv[:, :, 0], alpha=alpha, beta=0)
    haze_removed = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return haze_removed

# Função para aplicar filtro gaussiano
def apply_gaussian_blur(image, kernel_size=(3, 3)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Função para ajustar o brilho da imagem
def adjust_brightness(image, beta=0):
    brightness_adjusted = cv2.convertScaleAbs(image, beta=beta)
    return brightness_adjusted

# Função para subtrair valores específicos de cada banda
def subtract_values(image, x, y, z):
    image[:, :, 0] = np.clip(image[:, :, 0] - x, 0, 255)  # Banda R
    image[:, :, 1] = np.clip(image[:, :, 1] - y, 0, 255)  # Banda G
    image[:, :, 2] = np.clip(image[:, :, 2] - z, 0, 255)  # Banda B
    return image

# Função principal para carregar a imagem, ajustar cores, remover neblina, aplicar filtro gaussiano, ajustar brilho, aplicar correção gama e salvar o resultado
def process_image(source_path, output_path, gamma, brightness_adjustment, x, y, z):
    source_image, source_profile = read_raster(source_path)
    adjusted_image = color_match(source_image)
    haze_removed_image = remove_haze(adjusted_image)
    blurred_image = apply_gaussian_blur(haze_removed_image)
    gamma_corrected_image = apply_gamma_correction(blurred_image, gamma)
    brightness_adjusted_image = adjust_brightness(gamma_corrected_image, beta=brightness_adjustment)
    final_image = subtract_values(brightness_adjusted_image.copy(), x, y, z)
    write_raster(final_image, source_profile, output_path)

# Listar os arquivos .tif da pasta
dir_fotos = r'D:\SI_11'
dir_output = r'D:\SI_11\teste'
lista_dir = [f'{dir_fotos}\{filename}' for filename in os.listdir(dir_fotos) if filename.lower().endswith('.tif')]
lista_nome = [filename for filename in os.listdir(dir_fotos) if filename.lower().endswith('.tif')]

print(f"Total de imagens a processar: {len(lista_nome)}")

# Loop para processar as imagens, verificando se a imagem já foi processada
for i in tqdm(range(len(lista_dir)), desc="Corrigindo Cor: "):
    output_file = f'{dir_output}\\{lista_nome[i]}'
   
    process_image(lista_dir[i], output_file, gamma=.8, brightness_adjustment=-50, x=15, y=-10, z=-20)
