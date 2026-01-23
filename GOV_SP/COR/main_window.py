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

# Função para remover o efeito de neblina (dehaze) – sem slider de strength
def remove_haze(image, alpha=1.2):
    # Converte de BGR para YUV, ajusta o canal Y e volta para BGR
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.convertScaleAbs(yuv[:, :, 0], alpha=alpha, beta=0)
    haze_removed = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return haze_removed

# Função para aplicar o filtro gaussiano – sem slider de sigma
def apply_gaussian_blur(image, kernel_size=(3, 3)):
    return cv2.GaussianBlur(image, kernel_size, 0)

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

        self.color_params = None  # Parâmetros LAB calculados para a transferência de cor

        self.input_folder = None
        self.output_folder = None

        self.image_window.show()
        self.param_window.show()
        
        # Botão para reabrir a janela de visualização
        self.btn_reopen_image = QPushButton("Abrir Image Viewer")
        self.param_window.layout().addWidget(self.btn_reopen_image)
        self.btn_reopen_image.clicked.connect(self.reopen_image_view)
        
        # Botões para processamento em lote
        self.add_batch_processing_buttons()
        
        # Atributos para thread (se houver processamento assíncrono)
        self.future = None
        self.watcher = None
        
    def reopen_image_view(self):
        # Exibe a janela de visualização se estiver oculta
        self.image_window.show()
        
    # ==================================================
    # 1) TRANSFERÊNCIA DE COR (USANDO MASK THRESHOLD)
    # ==================================================
    def color_transfer_callback(self):
        if self.edited_image is None or self.reference_image is None:
            print("Precisa de imagem a editar e imagem de referência.")
            return
    
        # Calcula os par?metros de transfer?ncia de cor usando a base atual
        base_image = self.original_edited_image if self.original_edited_image is not None else self.edited_image
        self.color_params = self.compute_color_transfer_params(base_image, self.reference_image)
        self.update_image()
        print("Par?metros de transfer?ncia de cor calculados.")

    def compute_color_transfer_params(self, source_rgb, reference_rgb, l_percentiles=(5, 95)):
            """
            Calcula os parâmetros de transferência de cor da imagem source_rgb para reference_rgb,
            usando estatísticas robustas em LAB:

            - Ignora extremos de luminância (L) com base em percentis (default: 5–95%)
            - Calcula médias e desvios nessa faixa "saudável"
            - Cria fatores de escala LIMITADOS para evitar efeito metálico / lente verde

            Retorna um dicionário com:
            - médias e desvios (L,a,b) da fonte e da referência
            - fatores de escala lScale, aScale, bScale
            """
            # Converte para LAB em float32
            source_lab = cv2.cvtColor(source_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype("float32")
            reference_lab = cv2.cvtColor(reference_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype("float32")

            # Achata para [N,3]
            src_flat = source_lab.reshape(-1, 3)
            ref_flat = reference_lab.reshape(-1, 3)

            L_src = src_flat[:, 0]
            L_ref = ref_flat[:, 0]

            # Percentis de L para mascarar sombras profundas / highlights extremos
            l_low_src, l_high_src = np.percentile(L_src, l_percentiles)
            l_low_ref, l_high_ref = np.percentile(L_ref, l_percentiles)

            mask_src = (L_src >= l_low_src) & (L_src <= l_high_src)
            mask_ref = (L_ref >= l_low_ref) & (L_ref <= l_high_ref)

            # Em caso de máscara muito pequena, usa tudo
            if mask_src.sum() < 100:
                mask_src = np.ones_like(L_src, dtype=bool)
            if mask_ref.sum() < 100:
                mask_ref = np.ones_like(L_ref, dtype=bool)

            src_valid = src_flat[mask_src]
            ref_valid = ref_flat[mask_ref]

            # Estatísticas na faixa válida
            lMeanSrc, aMeanSrc, bMeanSrc = src_valid.mean(axis=0)
            lStdSrc, aStdSrc, bStdSrc = src_valid.std(axis=0)

            lMeanRef, aMeanRef, bMeanRef = ref_valid.mean(axis=0)
            lStdRef, aStdRef, bStdRef = ref_valid.std(axis=0)

            eps = 1e-6

            def safe_ratio(std_ref, std_src, min_scale, max_scale):
                if std_src < eps:
                    return 1.0
                r = std_ref / std_src
                return float(np.clip(r, min_scale, max_scale))

            # L pode variar mais; a/b menos para segurar saturação esquisita
            lScale = safe_ratio(lStdRef, lStdSrc, 0.5, 2.0)
            aScale = safe_ratio(aStdRef, aStdSrc, 0.5, 1.5)
            bScale = safe_ratio(bStdRef, bStdSrc, 0.5, 1.5)

            params = {
                # estatísticas da fonte
                "lMeanSrc": lMeanSrc,
                "aMeanSrc": aMeanSrc,
                "bMeanSrc": bMeanSrc,
                "lStdSrc": lStdSrc,
                "aStdSrc": aStdSrc,
                "bStdSrc": bStdSrc,
                # estatísticas da referência
                "lMeanRef": lMeanRef,
                "aMeanRef": aMeanRef,
                "bMeanRef": bMeanRef,
                "lStdRef": lStdRef,
                "aStdRef": aStdRef,
                "bStdRef": bStdRef,
                # fatores de escala já limitados
                "lScale": lScale,
                "aScale": aScale,
                "bScale": bScale,
            }
            return params


    def apply_color_transfer_params(self, source_rgb, params, strength=0.7, match_l_only=False):
        """
        Aplica a transfer?ncia de cor ? imagem source_rgb utilizando os par?metros LAB calculados.

        - Usa os fatores de escala limitados (lScale, aScale, bScale) se dispon?veis
        - 'strength' controla o quanto da imagem corrigida ? misturado (0 = nada, 1 = 100%)
        - Se 'match_l_only=True', ajusta apenas lumin?ncia (L), mantendo a cor (a/b) original
        """
        # Converte a imagem fonte para LAB (float32)
        source_lab = cv2.cvtColor(source_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype("float32")
        L, A, B = cv2.split(source_lab)

        eps = 1e-6

        # Recupera escalas (ou cai pro comportamento antigo, se n?o houver)
        lScale = params.get("lScale", params["lStdRef"] / (params["lStdSrc"] + eps))
        aScale = params.get("aScale", params["aStdRef"] / (params["aStdSrc"] + eps))
        bScale = params.get("bScale", params["bStdRef"] / (params["bStdSrc"] + eps))

        # L sempre ? ajustado
        L_trans = (L - params["lMeanSrc"]) * lScale + params["lMeanRef"]

        if match_l_only:
            A_trans = A
            B_trans = B
        else:
            A_trans = (A - params["aMeanSrc"]) * aScale + params["aMeanRef"]
            B_trans = (B - params["bMeanSrc"]) * bScale + params["bMeanRef"]

        # Clamping de seguran?a
        L_trans = np.clip(L_trans, 0, 255)
        A_trans = np.clip(A_trans, 0, 255)
        B_trans = np.clip(B_trans, 0, 255)

        transfer_lab = cv2.merge([L_trans, A_trans, B_trans])
        transfer_rgb = cv2.cvtColor(transfer_lab.astype("uint8"), cv2.COLOR_LAB2RGB)

        # Mistura com a imagem original para suavizar (strength < 1)
        if strength < 1.0:
            rgb_out = (strength * transfer_rgb.astype(np.float32) +
                    (1.0 - strength) * source_rgb.astype(np.float32))
            rgb_out = np.clip(rgb_out, 0, 255).astype(np.uint8)
        else:
            rgb_out = transfer_rgb

        return rgb_out

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

                image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
                if image_bgr is not None:
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    if self.reference_image is not None and self.color_params is None:
                        self.color_params = self.compute_color_transfer_params(image_rgb, self.reference_image)

                    # Aplica demais ajustes (blur, dehaze, etc.)
                    edited_image = self.apply_current_parameters(image_rgb)

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
    # 3) DEMAIS MÉTODOS (BOTÕES, SELEÇÃO DE PASTAS, ETC.)
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
    # 4) APPLY_CURRENT_PARAMETERS (com blur, dehaze, etc.)
    # ==================================================
    def apply_current_parameters(self, image):
        """
        Aplica os ajustes dos sliders (brilho, contraste, gamma, hue, satur, value,
        temperature, tint, ganhos R/G/B) e, por fim, filtro Gaussiano e Dehaze se selecionados.
        """
        img = image.astype(np.float32)

        # Color match (se par?metros j? calculados)
        if self.color_params is not None:
            strength = self.param_window.strength_slider[1].value() / 100.0
            img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
            img_uint8 = self.apply_color_transfer_params(
                img_uint8, self.color_params, strength=strength
            )
            img = img_uint8.astype(np.float32)
    

        if not hasattr(self.param_window, "brightness_slider"):
            return img.astype(np.uint8)

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
            # Converte de RGB para BGR para aplicar a função e depois volta para RGB
            bgr_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            blurred_bgr = apply_gaussian_blur(bgr_img, (3, 3))
            img = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            
        # 6) Dehaze, se ativado
        if self.param_window.dehaze_checkbox.isChecked():
            bgr_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            dehazed_bgr = remove_haze(bgr_img, alpha=1.2)
            img = cv2.cvtColor(dehazed_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    
        return img.astype(np.uint8)
    
    # ==================================================
    # 7) CARREGAR / SALVAR PARÂMETROS
    # ==================================================
    def load_reference_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.param_window,
            "Open Reference Image",
            "",
            "Images (*.png *.jpg *.tif *.bmp *.jp2)"
        )
        if file_path:
            self.reference_image = cv2.imread(file_path)
            self.reference_image = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
            self.image_window.update_reference_image(self.reference_image)
     
    def load_image_to_edit(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.param_window,
            "Open Image to Edit",
            "",
            "Images (*.png *.jpg *.tif *.bmp *.jp2)"
        )
        if file_path:
            image_bgr = cv2.imread(file_path)
            if image_bgr is None:
                print(f"Não foi possível carregar {file_path}")
                return
        
            full_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
            # Guarda a imagem para processamento final
            self.edited_image = full_rgb
            self.original_edited_image = full_rgb.copy()
        
            # Exibe a imagem em alta resolução
            self.image_window.update_edited_image(full_rgb)
            print("Imagem carregada em alta resolução.")
     
    def reset_image(self):
        if self.original_edited_image is not None:
            self.edited_image = self.original_edited_image.copy()
            self.image_window.update_edited_image(self.edited_image)
            self.color_params = None
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
