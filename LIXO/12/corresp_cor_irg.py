import cv2
import numpy as np
import rasterio
import os

# Função para ler a imagem raster e retornar as três primeiras bandas RGB se houver 4 bandas
def read_raster(image_path):
    with rasterio.open(image_path) as src:
        all_bands = src.read()
        num_bands = all_bands.shape[0]

        image_rgb = all_bands  # [R, G, B]
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
def color_match(sourcew):
    num_bands = sourcew.shape[2]

    # Verifica se existem 4 ou 5 bandas (NIR, R, G, B, + uma extra)
    if num_bands == 5:
        b, g, r, nir, _ = cv2.split(sourcew)  # Ignora a quinta banda
    elif num_bands == 4:
        b, g, r, nir = cv2.split(sourcew)  # cv2.split retorna B, G, R e NIR na ordem OpenCV
    else:
        raise ValueError(f"Esperado 4 ou 5 bandas (B, G, R, NIR), mas a imagem tem {num_bands} bandas.")

    # Realiza o ajuste de cores utilizando as bandas RGB (descartando o NIR)
    source = cv2.merge([b, g, r])  # Reorganiza como [R, G, B]
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2Lab)

    source_mean, source_std = (np.array([[113.08665894],
                                         [126.28659312],
                                         [118.2191386]]), 
                               np.array([[46.09646432],
                                         [ 7.3922675],
                                         [10.01725102]]))

    reference_mean, reference_std = (np.array([[118.48824545],
                                               [123.28214963],
                                               [128.94486877]]), 
                                     np.array([[20.41027262],
                                               [ 3.37733758],
                                               [ 5.32481135]]))

    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    reference_mean = reference_mean.reshape(1, 1, 3)
    reference_std = reference_std.reshape(1, 1, 3)

    matched_lab = (source_lab - source_mean) * (reference_std / source_std) + reference_mean
    matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)

    matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_Lab2BGR)

    # Retorna a imagem combinada com a banda NIR no formato [NIR, R, G]
    s_final = cv2.merge([nir, matched_bgr[:, :, 0], matched_bgr[:, :, 1]])  # [NIR, R, G]

    return s_final

# Função para remover neblina (dehaze)
def remove_haze(image, alpha=1.5):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.convertScaleAbs(yuv[:, :, 0], alpha=alpha, beta=0)
    haze_removed = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return haze_removed

# Função para aplicar filtro gaussiano
def apply_gaussian_blur(image, kernel_size=(3, 3)):
    return cv2.GaussianBlur(image, kernel_size, 0)

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

# Função para subtrair valores específicos de cada banda
def subtract_values(image, x, y, z):
    image[:, :, 0] = np.clip(image[:, :, 0] - x, 0, 255)  # Banda R
    image[:, :, 1] = np.clip(image[:, :, 1] - y, 0, 255)  # Banda G
    image[:, :, 2] = np.clip(image[:, :, 2] - z, 0, 255)  # Banda B
    return image

# Função principal para carregar a imagem, ajustar cores, remover neblina, aplicar filtro gaussiano, ajustar brilho, aplicar correção gama e salvar o resultado
def process_image(source_path, output_path, gamma, brightness_adjustment, x, y, z):
    # Verificar se a imagem de saída já existe
    if os.path.exists(output_path):
        print(f"Imagem já processada: {output_path}. Pulando...")
        return

    source_image, source_profile = read_raster(source_path)
    adjusted_image = color_match(source_image)
    haze_removed_image = remove_haze(adjusted_image)  # Aplicar remoção de neblina

    blurred_image = apply_gaussian_blur(haze_removed_image)
    gamma_corrected_image = apply_gamma_correction(blurred_image, gamma)
    brightness_adjusted_image = adjust_brightness(gamma_corrected_image, beta=brightness_adjustment)
    
    # Subtrair valores 
    final_image = subtract_values(brightness_adjusted_image.copy(), x, y, z)

    write_raster(final_image, source_profile, output_path)
    print(f"Imagem ajustada e salva em {output_path}")

# Função para exibir o progresso
def show_progress(current, total):
    progress = (current / total) * 100
    print(f"Progresso: {progress:.2f}%")

# Executar o processo
formato = '.tif'
dir_fotos = r'D:\SI_12'
dir_output = r'D:\SI_12\Nova pasta'

lista_dir = [f'{dir_fotos}\{filename}' for filename in os.listdir(dir_fotos) if filename.lower().endswith(formato)]
lista_nome = [filename for filename in os.listdir(dir_fotos) if filename.lower().endswith(formato)]

total_imagens = len(lista_dir)

for i in range(total_imagens):
    process_image(lista_dir[i], f'{dir_output}\\{lista_nome[i]}', gamma=1.2, brightness_adjustment=-30, x=0, y=0, z=0)
    show_progress(i + 1, total_imagens)
