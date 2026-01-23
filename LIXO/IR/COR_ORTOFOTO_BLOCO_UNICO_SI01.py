import cv2
import numpy as np
import rasterio
import os
from tqdm import tqdm

# Lista para guardar as imagens com erro
imagens_com_erro = []

def read_raster(image_path):
    with rasterio.open(image_path) as src:
        if src.count < 4:
            raise ValueError("A imagem deve ter pelo menos 4 bandas para usar as bandas 3, 0 e 1.")
        band_order = [4, 1, 2]
        image_rgb = np.stack([src.read(b) for b in band_order], axis=-1)
        profile = src.profile
        profile.update(count=3)
        return image_rgb, profile

def write_raster(image, profile, output_path):
    image = np.moveaxis(image, -1, 0)
    profile.update(dtype=rasterio.uint8)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(image)

def apply_gamma_correction(image, gamma=1.0):
    image_normalized = image / 255.0
    image_corrected = np.power(image_normalized, gamma)
    return np.clip(image_corrected * 255, 0, 255).astype(np.uint8)

def adjust_brightness(image, beta=0):
    return cv2.convertScaleAbs(image, beta=beta)

def adjust_contrast(image, alpha=1.0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def adjust_intensity(image, factor=1.0):
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def adjust_rgb(image, red=0, green=0, blue=0):
    adjusted = image.copy()
    adjusted[:, :, 0] = np.clip(adjusted[:, :, 0] + red, 0, 255)
    adjusted[:, :, 1] = np.clip(adjusted[:, :, 1] + green, 0, 255)
    adjusted[:, :, 2] = np.clip(adjusted[:, :, 2] + blue, 0, 255)
    return adjusted

def remove_haze(image, alpha=1.2):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.convertScaleAbs(yuv[:, :, 0], alpha=alpha, beta=0)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def adjust_saturation(image, saturation_factor=1.0):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def process_image(source_path, output_path, gamma=1.0, brightness_adjustment=0, contrast=1.0, intensity_factor=1.0, red=0, green=0, blue=0, saturation_factor=1.0, apply_fog_removal=False, apply_blur=False):
    try:
        source_image, source_profile = read_raster(source_path)

        image = apply_gamma_correction(source_image, gamma)
        image = adjust_brightness(image, brightness_adjustment)
        image = adjust_contrast(image, contrast)
        image = adjust_intensity(image, intensity_factor)
        image = adjust_rgb(image, red, green, blue)
        image = adjust_saturation(image, saturation_factor)

        if apply_fog_removal:
            image = remove_haze(image)

        if apply_blur:
            image = apply_gaussian_blur(image)

        write_raster(image, source_profile, output_path)
        print(f"[OK] Imagem ajustada e salva em {output_path}")

    except Exception as e:
        print(f"[ERRO] Falha ao processar {source_path}: {e}")

        # Se deu erro e já criou arquivo de saída, remove
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"Arquivo {output_path} removido devido a erro.")
            except Exception as err_remocao:
                print(f"Falha ao tentar remover {output_path}: {err_remocao}")

        # Guarda o nome da imagem com erro para o relatório
        imagens_com_erro.append(os.path.basename(source_path))

# Diretórios
formato = '.tif'
dir_fotos = r'K:\SP22_BE_13_03052024_HD03\SI_08\OUT'
dir_output = r'\\192.168.2.27\f\si_08\cir'

if not os.path.exists(dir_output):
    os.makedirs(dir_output)

# Listar os arquivos
lista_dir = [os.path.join(dir_fotos, filename) for filename in os.listdir(dir_fotos) if filename.lower().endswith(formato)]
lista_nome = [filename for filename in os.listdir(dir_fotos) if filename.lower().endswith(formato)]

# Processar
for img_path, img_name in tqdm(zip(lista_dir, lista_nome), desc="Processando imagens", total=len(lista_dir), unit="imagem"):
    output_file = os.path.join(dir_output, img_name)

    # Verificar se já existe
    if os.path.exists(output_file):
        print(f"[SKIP] {img_name} já processado.")
        continue

    # Processar imagem
    process_image(
        img_path, output_file,
        gamma=1.0,
        brightness_adjustment=0,
        contrast=1.0,
        intensity_factor=1.0,
        red=0, green=0, blue=0,
        saturation_factor=1.0,
        apply_fog_removal=False,
        apply_blur=False
    )

# Após tudo, salvar o relatório de imagens que falharam
relatorio_erro = os.path.join(dir_output, 'relatorio_erros.txt')
if imagens_com_erro:
    with open(relatorio_erro, 'w') as f:
        f.write("Imagens que não foram processadas corretamente:\n\n")
        for nome in imagens_com_erro:
            f.write(f"{nome}\n")
    print(f"\n[RELATÓRIO] {len(imagens_com_erro)} imagens com erro. Relatório salvo em {relatorio_erro}")
else:
    print("\nNenhuma imagem com erro. Processamento finalizado com sucesso!")
