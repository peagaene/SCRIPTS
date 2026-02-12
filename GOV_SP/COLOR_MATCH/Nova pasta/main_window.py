import cv2
import os
import numpy as np
import json
from osgeo import gdal
from PyQt5.QtWidgets import (
    QFileDialog, QVBoxLayout, QPushButton, QProgressBar, QApplication
)

from image_window import ImageWindow
from parameter_window import ParameterWindow
from utils import save_image_with_epsg

# Remove haze effect using a simple luminance boost.
def remove_haze(image, alpha=1.2):
    # Convert BGR to YUV, adjust Y channel, and convert back.
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.convertScaleAbs(yuv[:, :, 0], alpha=alpha, beta=0)
    haze_removed = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return haze_removed

# Apply gaussian blur using a sigma value.
def apply_gaussian_blur(image, sigma):
    if sigma <= 0:
        return image
    ksize = int(max(3, 2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

class MainWindow:
    def __init__(self):
        self.image_window = ImageWindow()
        self.param_window = ParameterWindow(
            self.load_reference_image,
            self.load_image_to_edit,
            self.reset_image,
            self.update_image,
            self.save_parameters,
            self.load_parameters,
            self.color_transfer_callback,
        )
        # Barra de progresso
        self.progress_bar = QProgressBar()
        self.param_window.layout().addWidget(self.progress_bar)

        self.reference_image = None
        self.edited_image = None
        self.original_edited_image = None

        self.color_params = None  # Parametros LAB calculados para a transferencia de cor

        self.input_folder = None
        self.output_folder = None

        self.image_window.show()
        self.param_window.show()
        
        # Botao para reabrir a janela de visualizacao
        self.btn_reopen_image = QPushButton("Abrir Image Viewer")
        self.param_window.layout().addWidget(self.btn_reopen_image)
        self.btn_reopen_image.clicked.connect(self.reopen_image_view)
        
        # Botoes para processamento em lote
        self.add_batch_processing_buttons()
        
        # Atributos para thread (se houver processamento assincrono)
        self.future = None
        self.watcher = None
        
    def reopen_image_view(self):
        # Exibe a janela de visualizacao se estiver oculta
        self.image_window.show()
        
    # ==================================================
    # 1) TRANSFERENCIA DE COR (USANDO MASK THRESHOLD)
    # ==================================================
    def color_transfer_callback(self):
        if self.edited_image is None or self.reference_image is None:
            print("Precisa de imagem a editar e imagem de referencia.")
            return
    
        # Calcula os parametros de transferencia de cor usando todos os pixels
        self.color_params = self.compute_color_transfer_params(self.edited_image, self.reference_image)
        # Aplica a transferencia de cor em toda a imagem
        transferred = self.apply_color_transfer_params(self.edited_image, self.color_params)
        self.edited_image = transferred
        self.original_edited_image = transferred.copy()
        self.image_window.update_edited_image(transferred)
        print("Transferencia de cor aplicada e parametros LAB salvos.")

    def compute_color_transfer_params(self, source_rgb, reference_rgb):
        """
        Calcula os parametros de transferencia de cor da imagem source_rgb para reference_rgb
        utilizando todos os pixels da imagem fonte.
        
        Retorna um dicionario com as medias e desvios padrao dos canais L, a e b.
        """
        # Converte as imagens para o espaco LAB (em float32)
        source_lab = cv2.cvtColor(source_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype("float32")
        reference_lab = cv2.cvtColor(reference_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype("float32")
        
        # Usa todos os pixels da imagem fonte
        valid_pixels = source_lab.reshape(-1, 3)
        lMeanSrc = valid_pixels[:, 0].mean()
        aMeanSrc = valid_pixels[:, 1].mean()
        bMeanSrc = valid_pixels[:, 2].mean()
        lStdSrc = valid_pixels[:, 0].std()
        aStdSrc = valid_pixels[:, 1].std()
        bStdSrc = valid_pixels[:, 2].std()
        
        # Estatisticas da imagem de referencia (usando todos os pixels)
        lMeanRef, aMeanRef, bMeanRef = cv2.mean(reference_lab)[:3]
        lStdRef = reference_lab[:, :, 0].std()
        aStdRef = reference_lab[:, :, 1].std()
        bStdRef = reference_lab[:, :, 2].std()
        
        params = {
            "lMeanSrc": lMeanSrc, "aMeanSrc": aMeanSrc, "bMeanSrc": bMeanSrc,
            "lStdSrc": lStdSrc, "aStdSrc": aStdSrc, "bStdSrc": bStdSrc,
            "lMeanRef": lMeanRef, "aMeanRef": aMeanRef, "bMeanRef": bMeanRef,
            "lStdRef": lStdRef, "aStdRef": aStdRef, "bStdRef": bStdRef,
        }
        return params

    def apply_color_transfer_params(self, source_rgb, params):
        """
        Aplica a transferencia de cor a imagem source_rgb utilizando os parametros LAB calculados,
        alterando TODOS os pixels da imagem.
        """
        # Converte a imagem fonte para LAB (float32)
        source_lab = cv2.cvtColor(source_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype("float32")
        (l, a, b) = cv2.split(source_lab)
        
        eps = 1e-8  # Para evitar divisao por zero
        
        # Aplica a transformacao em TODOS os pixels
        l_trans = (l - params["lMeanSrc"]) * (params["lStdRef"] / (params["lStdSrc"] + eps)) + params["lMeanRef"]
        a_trans = (a - params["aMeanSrc"]) * (params["aStdRef"] / (params["aStdSrc"] + eps)) + params["aMeanRef"]
        b_trans = (b - params["bMeanSrc"]) * (params["bStdRef"] / (params["bStdSrc"] + eps)) + params["bMeanRef"]
        
        l_trans = np.clip(l_trans, 0, 255)
        a_trans = np.clip(a_trans, 0, 255)
        b_trans = np.clip(b_trans, 0, 255)
        
        transfer_lab = cv2.merge([l_trans, a_trans, b_trans])
        transfer_rgb = cv2.cvtColor(transfer_lab.astype("uint8"), cv2.COLOR_LAB2RGB)
        return transfer_rgb

    # ==================================================
    # 2) PROCESSAMENTO EM LOTE
    # ==================================================
    def process_images(self):
        if not self.input_folder or not self.output_folder:
            print("Please select both input and output folders.")
            return
    
        file_list = [f for f in os.listdir(self.input_folder)
                     if os.path.isfile(os.path.join(self.input_folder, f))]
        total_files = len(file_list)
    
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(total_files)
        self.progress_bar.setValue(0)
    
        print(f"Processing images from {self.input_folder} to {self.output_folder}...")
        
        if not hasattr(self, 'overwrite_existing'):
            self.overwrite_existing = False
    
        for i, filename in enumerate(file_list, start=1):
            input_path = os.path.join(self.input_folder, filename)
            if os.path.isfile(input_path):
                base_name, _ = os.path.splitext(filename)
                output_path = os.path.join(self.output_folder, base_name + ".tif")
                if os.path.exists(output_path) and not self.overwrite_existing:
                    print(f"File {output_path} already exists. Skipping processing for {filename}.")
                    self.progress_bar.setValue(i)
                    continue
    
                original_dataset = gdal.Open(input_path)
                if original_dataset is None:
                    print(f"Failed to open: {input_path}")
                    self.progress_bar.setValue(i)
                    continue
    
                geotransform = original_dataset.GetGeoTransform()
                projection = original_dataset.GetProjection()
                original_dataset = None
    
                image_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                if image_bgr is not None:
                    if image_bgr.ndim == 2:
                        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
                    elif image_bgr.shape[2] == 4:
                        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

                    input_dtype = image_bgr.dtype
                    if np.issubdtype(input_dtype, np.integer):
                        scale_max = np.iinfo(input_dtype).max
                    else:
                        scale_max = float(np.max(image_bgr)) if np.max(image_bgr) > 0 else 1.0

                    image_bgr_f = image_bgr.astype(np.float32)
                    if scale_max != 255:
                        image_bgr_f *= (255.0 / scale_max)

                    image_rgb = cv2.cvtColor(image_bgr_f, cv2.COLOR_BGR2RGB)
    
                    # Aplica a transferencia de cor, se os parametros ja tiverem sido calculados;
                    # caso contrario, calcula-os e aplica.
                    if self.color_params is not None and self.reference_image is not None:
                        image_rgb = self.apply_color_transfer_params(image_rgb, self.color_params)
                    else:
                        if self.reference_image is not None:
                            self.color_params = self.compute_color_transfer_params(image_rgb, self.reference_image)
                            image_rgb = self.apply_color_transfer_params(image_rgb, self.color_params)
                    
# Remove haze effect using a simple luminance boost.
                    edited_image = self.apply_current_parameters(image_rgb)

                    if scale_max != 255:
                        edited_image = edited_image.astype(np.float32) * (scale_max / 255.0)
                        edited_image = np.clip(edited_image, 0, scale_max).astype(input_dtype)
    
                    self.progress_bar.setValue(i)
                    QApplication.processEvents()                   
    
                    try:
                        save_image_with_epsg(
                            edited_image,
                            output_path,
                            geotransform=geotransform,
                            projection=projection
                        )
                        print(f"Processed and saved: {output_path}")
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
    
            self.progress_bar.setValue(i)
            QApplication.processEvents()
    
        print("Processamento completo.")
    
    def apply_lut_rgb(self, image_rgb, lut_list):
        out = np.zeros_like(image_rgb)
        for ch in range(3):
            out[..., ch] = cv2.LUT(image_rgb[..., ch], lut_list[ch])
        return out
    
    # ==================================================
    # 3) DEMAIS METODOS (BOTOES, SELECAO DE PASTAS, ETC.)
    # ==================================================
    def add_batch_processing_buttons(self):
        batch_layout = QVBoxLayout()
    
        select_input_folder_button = QPushButton("Selecione a pasta de origem")
        select_input_folder_button.clicked.connect(self.select_input_folder)
        batch_layout.addWidget(select_input_folder_button)
    
        select_output_folder_button = QPushButton("Selecione a pasta de saida")
        select_output_folder_button.clicked.connect(self.select_output_folder)
        batch_layout.addWidget(select_output_folder_button)
    
        process_images_button = QPushButton("Processar Imagens")
        process_images_button.clicked.connect(self.process_images)
        batch_layout.addWidget(process_images_button)
    
        self.param_window.layout().addLayout(batch_layout)
    
    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self.param_window, "Selecione a pasta de origem")
        if folder:
            self.input_folder = folder
            print(f"Input folder selected: {self.input_folder}")
    
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self.param_window, "Selecione a pasta de saida")
        if folder:
            self.output_folder = folder
            print(f"Output folder selected: {self.output_folder}")
    
    # ==================================================
# Remove haze effect using a simple luminance boost.
    # ==================================================
    def apply_current_parameters(self, image):
        """
        Aplica os ajustes dos sliders (brilho, contraste, gamma, hue, satur, value,
        temperature, tint, ganhos R/G/B) e, por fim, filtro Gaussiano e Dehaze se selecionados.
        """
        img = image.astype(np.float32)
    
        # 1) Brilho & Contraste
        brightness_factor = 1 + (self.param_window.brightness_slider[1].value() / 100.0)
        contrast_adjust = self.param_window.contrast_slider[1].value() / 100.0
        mean_val = np.mean(img)
        img = (1 + contrast_adjust) * (img - mean_val) + mean_val
        img *= brightness_factor
        img = np.clip(img, 0, 255)
    
        # 2) Temperature / Tint + R/G/B Gains
        temp_value = self.param_window.temperature_slider[1].value()
        tint_value = self.param_window.tint_slider[1].value()
        r_gain = self.param_window.r_gain_slider[1].value() / 100.0
        g_gain = self.param_window.g_gain_slider[1].value() / 100.0
        b_gain = self.param_window.b_gain_slider[1].value() / 100.0
    
        if temp_value != 6500:
            factor = (temp_value - 6500) / 4500.0
            if factor > 0:
                r_gain *= (1 + factor)
            else:
                b_gain *= (1 - factor)
    
        if tint_value != 0:
            ft = tint_value / 100.0
            if ft > 0:
                r_gain *= (1 + 0.5 * ft)
                b_gain *= (1 + 0.5 * ft)
            else:
                g_gain *= (1 - 0.5 * ft)
    
        img[..., 0] *= r_gain
        img[..., 1] *= g_gain
        img[..., 2] *= b_gain
        img = np.clip(img, 0, 255)
    
        # 3) Hue, Satur, Value (HSV)
        hue_adjust = self.param_window.hue_slider[1].value()
        saturation_factor = 1 + (self.param_window.saturation_slider[1].value() / 100.0)
        value_factor = 1 + (self.param_window.value_slider[1].value() / 100.0)
    
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_adjust) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_factor, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    
        # 4) Gamma
        gamma_val = self.param_window.gamma_slider[1].value() / 100.0
        if gamma_val != 1.0:
            img = np.power(img / 255.0, gamma_val) * 255
        img = np.clip(img, 0, 255)
    
        # 5) Filtro Gaussiano, se ativado
        if self.param_window.gaussian_checkbox.isChecked():
            sigma_val = self.param_window.gaussian_sigma_slider[1].value() / 10.0
            if sigma_val > 0:
                # Converte de RGB para BGR para aplicar e depois volta para RGB
                bgr_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                blurred_bgr = apply_gaussian_blur(bgr_img, sigma_val)
                img = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            
        # 6) Dehaze, se ativado
        if self.param_window.dehaze_checkbox.isChecked():
            strength = self.param_window.dehaze_strength_slider[1].value()
            if strength > 0:
                alpha = 1.0 + (strength / 100.0)
                bgr_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                dehazed_bgr = remove_haze(bgr_img, alpha=alpha)
                img = cv2.cvtColor(dehazed_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        
        return img.astype(np.uint8)
    
    # ==================================================
    # 7) CARREGAR / SALVAR PARAMETROS
    # ==================================================
    def load_reference_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.param_window,
            "Open Reference Image",
            "",
            "Images (*.png *.jpg *.tif *.bmp *.jp2)"
        )
        if file_path:
            ref_bgr = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if ref_bgr is None:
                print(f"Nao foi possivel carregar {file_path}")
                return
            if ref_bgr.ndim == 2:
                ref_bgr = cv2.cvtColor(ref_bgr, cv2.COLOR_GRAY2BGR)
            elif ref_bgr.shape[2] == 4:
                ref_bgr = cv2.cvtColor(ref_bgr, cv2.COLOR_BGRA2BGR)
            if ref_bgr.dtype != np.uint8:
                scale_max = np.iinfo(ref_bgr.dtype).max
                ref_bgr = (ref_bgr.astype(np.float32) * (255.0 / scale_max)).astype(np.uint8)
            self.reference_image = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
            self.image_window.update_reference_image(self.reference_image)
    
    def load_image_to_edit(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.param_window,
            "Open Image to Edit",
            "",
            "Images (*.png *.jpg *.tif *.bmp *.jp2)"
        )
        if file_path:
            image_bgr = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image_bgr is None:
                print(f"Nao foi possivel carregar {file_path}")
                return

            if image_bgr.ndim == 2:
                image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
            elif image_bgr.shape[2] == 4:
                image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
            if image_bgr.dtype != np.uint8:
                scale_max = np.iinfo(image_bgr.dtype).max
                image_bgr = (image_bgr.astype(np.float32) * (255.0 / scale_max)).astype(np.uint8)

            full_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Guarda a imagem para processamento final
            self.edited_image = full_rgb
            self.original_edited_image = full_rgb.copy()

            # Exibe a imagem em alta resolucao
            self.image_window.update_edited_image(full_rgb)
            print("Imagem carregada em alta resolucao.")

    def reset_image(self):
        if self.original_edited_image is not None:
            self.edited_image = self.original_edited_image.copy()
            self.image_window.update_edited_image(self.edited_image)
            sliders = [
                self.param_window.saturation_slider,
                self.param_window.value_slider,
                self.param_window.contrast_slider,
                self.param_window.brightness_slider,
                self.param_window.gamma_slider,
                self.param_window.hue_slider,
                self.param_window.temperature_slider,
                self.param_window.tint_slider,
                self.param_window.r_gain_slider,
                self.param_window.g_gain_slider,
                self.param_window.b_gain_slider
            ]
            for slider in sliders:
                if slider == self.param_window.gamma_slider:
                    slider[1].setValue(100)  # gamma = 1
                elif slider == self.param_window.temperature_slider:
                    slider[1].setValue(6500)
                elif slider == self.param_window.tint_slider:
                    slider[1].setValue(0)
                elif slider in (self.param_window.r_gain_slider,
                                self.param_window.g_gain_slider,
                                self.param_window.b_gain_slider):
                    slider[1].setValue(100)
                else:
                    slider[1].setValue(0)
            self.param_window.gaussian_checkbox.setChecked(False)
            self.param_window.gaussian_sigma_slider[1].setValue(0)
            self.param_window.dehaze_checkbox.setChecked(False)
            self.param_window.dehaze_strength_slider[1].setValue(0)

    def update_image(self):
        if self.edited_image is not None and self.original_edited_image is not None:
            updated_image = self.apply_current_parameters(self.original_edited_image.copy())
            self.image_window.update_edited_image(updated_image)
            
    def save_parameters(self):
        save_path, _ = QFileDialog.getSaveFileName(
            self.param_window,
            "Save Parameters",
            "",
            "JSON Files (*.json)"
        )
        if save_path:
            parameters = {
                "brightness": self.param_window.brightness_slider[1].value(),
                "contrast": self.param_window.contrast_slider[1].value(),
                "gamma": self.param_window.gamma_slider[1].value(),
                "hue": self.param_window.hue_slider[1].value(),
                "saturation": self.param_window.saturation_slider[1].value(),
                "value": self.param_window.value_slider[1].value(),
                "temperature": self.param_window.temperature_slider[1].value(),
                "tint": self.param_window.tint_slider[1].value(),
                "r_gain": self.param_window.r_gain_slider[1].value(),
                "g_gain": self.param_window.g_gain_slider[1].value(),
                "b_gain": self.param_window.b_gain_slider[1].value(),
                "gaussian_enabled": self.param_window.gaussian_checkbox.isChecked(),
                "gaussian_sigma": self.param_window.gaussian_sigma_slider[1].value(),
                "dehaze_enabled": self.param_window.dehaze_checkbox.isChecked(),
                "dehaze_strength": self.param_window.dehaze_strength_slider[1].value()
            }
            with open(save_path, 'w') as f:
                json.dump(parameters, f, indent=4)
            print(f"Parameters saved to {save_path}")
    
    def load_parameters(self):
        load_path, _ = QFileDialog.getOpenFileName(
            self.param_window,
            "Load Parameters",
            "",
            "JSON Files (*.json)"
        )
        if load_path:
            with open(load_path, 'r') as f:
                parameters = json.load(f)
        
            self.param_window.brightness_slider[1].setValue(parameters.get("brightness", 0))
            self.param_window.contrast_slider[1].setValue(parameters.get("contrast", 0))
            self.param_window.gamma_slider[1].setValue(parameters.get("gamma", 100))
            self.param_window.hue_slider[1].setValue(parameters.get("hue", 0))
            self.param_window.saturation_slider[1].setValue(parameters.get("saturation", 0))
            self.param_window.value_slider[1].setValue(parameters.get("value", 0))
            self.param_window.temperature_slider[1].setValue(parameters.get("temperature", 6500))
            self.param_window.tint_slider[1].setValue(parameters.get("tint", 0))
            self.param_window.r_gain_slider[1].setValue(parameters.get("r_gain", 100))
            self.param_window.g_gain_slider[1].setValue(parameters.get("g_gain", 100))
            self.param_window.b_gain_slider[1].setValue(parameters.get("b_gain", 100))
        
            self.param_window.gaussian_checkbox.setChecked(parameters.get("gaussian_enabled", False))
            self.param_window.gaussian_sigma_slider[1].setValue(parameters.get("gaussian_sigma", 0))
            self.param_window.dehaze_checkbox.setChecked(parameters.get("dehaze_enabled", False))
            self.param_window.dehaze_strength_slider[1].setValue(parameters.get("dehaze_strength", 0))
        
            self.update_image()
            print(f"Parameters loaded from {load_path}")
