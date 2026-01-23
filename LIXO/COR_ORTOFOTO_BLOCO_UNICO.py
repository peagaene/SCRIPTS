import sys
import cv2
import os
from osgeo import gdal, osr
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import json
from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QWidget, QSizePolicy, QTabWidget, QProgressBar
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QSize

def save_image_with_epsg(image, output_path, epsg=31983, geotransform=None, projection=None):
    """Salva uma imagem com georreferenciamento EPSG especificado."""
    height, width, channels = image.shape
    
    # Crie um arquivo GeoTIFF usando GDAL
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, width, height, channels, gdal.GDT_Byte)
    
    # Defina o sistema de coordenadas (projeção)
    if projection is None:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        projection = srs.ExportToWkt()
    dataset.SetProjection(projection)
    
    # Escreva os dados da imagem no arquivo
    for i in range(channels):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(image[:, :, i])
    
    # Aplique o GeoTransform se fornecido
    if geotransform:
        dataset.SetGeoTransform(geotransform)
    
    # Salve e feche o dataset
    dataset.FlushCache()
    dataset = None

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
        )
        
        # Adicionar barra de progresso
        self.progress_bar = QProgressBar()
        self.param_window.layout().addWidget(self.progress_bar)

        self.reference_image = None
        self.edited_image = None
        self.input_folder = None
        self.output_folder = None

        self.image_window.show()
        self.param_window.show()
        
        # Adicionar botões de processamento em lote
        self.add_batch_processing_buttons()
        
    def add_batch_processing_buttons(self):
        batch_layout = QVBoxLayout()

        # Botão para selecionar pasta de entrada
        select_input_folder_button = QPushButton("Selecione a pasta de origem")
        select_input_folder_button.clicked.connect(self.select_input_folder)
        batch_layout.addWidget(select_input_folder_button)

        # Botão para selecionar pasta de saída
        select_output_folder_button = QPushButton("Selecione a pasta de saida")
        select_output_folder_button.clicked.connect(self.select_output_folder)
        batch_layout.addWidget(select_output_folder_button)

        # Botão para processar imagens
        process_images_button = QPushButton("Processar Imagens")
        process_images_button.clicked.connect(self.process_images)
        batch_layout.addWidget(process_images_button)

        # Adicionar ao layout principal da janela de parâmetros
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
              
    def process_images(self):
        if not self.input_folder or not self.output_folder:
            print("Please select both input and output folders.")
            return
    
        # Obter lista de arquivos
        file_list = [f for f in os.listdir(self.input_folder) if os.path.isfile(os.path.join(self.input_folder, f))]
        total_files = len(file_list)
        
        # Configurar barra de progresso
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(total_files)
        self.progress_bar.setValue(0)
    
        print(f"Processing images from {self.input_folder} to {self.output_folder}...")
    
        for i, filename in enumerate(file_list, start=1):
            input_path = os.path.join(self.input_folder, filename)
            if os.path.isfile(input_path):
                # Carregar imagem original com GDAL para preservar metadados
                original_dataset = gdal.Open(input_path)
                if original_dataset is None:
                    print(f"Failed to open: {input_path}")
                    continue
    
                # Obter GeoTransform e projeção
                geotransform = original_dataset.GetGeoTransform()
                projection = original_dataset.GetProjection()
    
                # Carregar imagem com OpenCV
                image = cv2.imread(input_path)
                if image is not None:
                    # Converter para RGB para consistência
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Aplicar os parâmetros atuais
                    edited_image = self.apply_current_parameters(image_rgb)
                    
                    # Salvar imagem editada com georreferenciamento
                    output_path = os.path.join(self.output_folder, filename).replace(".jpg", ".tif").replace(".png", ".tif")
                    try:
                        save_image_with_epsg(edited_image, output_path, geotransform=geotransform, projection=projection)
                        print(f"Processed and saved: {output_path}")
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

            # Atualizar valor da barra de progresso
            self.progress_bar.setValue(i)

        print("Processing complete.")
                   
    def apply_current_parameters(self, image):
        """Aplica os parâmetros atuais a uma imagem."""
        # Converta a imagem de BGR para RGB
        rgb = image.astype(np.float32)
        
        # Obter os valores dos sliders
        R_adjust = self.param_window.R_slider[1].value()  # Ajuste no canal R
        G_adjust = self.param_window.G_slider[1].value()  # Ajuste no canal G
        B_adjust = self.param_window.B_slider[1].value()  # Ajuste no canal B
        saturation_adjust = 1 + (self.param_window.saturation_slider[1].value() / 100)  # Saturação
        value_adjust = self.param_window.value_slider[1].value() / 100  # Luminosidade
        intensity_adjust = self.param_window.intensity_slider[1].value() / 100  # Intensidade
        contrast_adjust = self.param_window.contrast_slider[1].value() / 100  # Contraste
        brightness_adjust = self.param_window.brightness_slider[1].value()/100  # Brilho
        gamma_adjust = self.param_window.gamma_slider[1].value() / 100  # Gamma entre 0.00 e 2.00
    
        # Ajustes nos canais R, G, B
        rgb[:, :, 0] += R_adjust  # Canal R
        rgb[:, :, 1] += G_adjust  # Canal G
        rgb[:, :, 2] += B_adjust  # Canal B
        rgb = np.clip(rgb, 0, 255)
    
        #Converter para escala de cinza para saturação e luminosidade
        gray = np.mean(rgb, axis=2, keepdims=True)
    
        # Ajustar saturação
        rgb += saturation_adjust * (rgb - gray)
        rgb = np.clip(rgb, 0, 255)
    
        # Ajustar luminosidade
        rgb += value_adjust * gray
        rgb = np.clip(rgb, 0, 255)
    
        # Aplicar intensidade
        rgb *= (1 + intensity_adjust)
        rgb = np.clip(rgb, 0, 255)
    
        # Aplicar contraste
        mean = np.mean(rgb)
        rgb = (1 + contrast_adjust) * (rgb - mean) + mean
        rgb = np.clip(rgb, 0, 255)
    
        # Aplicar brilho
        rgb += brightness_adjust
        rgb = np.clip(rgb, 0, 255)
    
        # Aplicar gama
        if gamma_adjust != 1.0:
            rgb = np.power(rgb / 255.0, gamma_adjust) * 255
            rgb = np.clip(rgb, 0, 255)
    
        return rgb.astype(np.uint8)
    
    def load_reference_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self.param_window, "Open Reference Image", "", "Images (*.png *.jpg *.tif *.bmp *.jp2)")
        if file_path:
            self.reference_image = cv2.imread(file_path)
            self.reference_image = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
            self.image_window.update_reference_image(self.reference_image)

    def load_image_to_edit(self):
        file_path, _ = QFileDialog.getOpenFileName(self.param_window, "Open Image to Edit", "", "Images (*.png *.jpg *.tif *.bmp *.jp2)")
        if file_path:
            # Carregar a imagem e converter para RGB imediatamente
            self.edited_image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            self.original_edited_image = self.edited_image.copy()  # Armazenar a imagem original para reset
            self.image_window.update_edited_image(self.edited_image)

    def reset_image(self):
        if self.original_edited_image is not None:
            # Resetar imagem editada para o estado original
            self.edited_image = self.original_edited_image.copy()
            self.image_window.update_edited_image(self.edited_image)
    
            # Resetar sliders para seus valores padrão
            sliders = [
                self.param_window.R_slider,
                self.param_window.G_slider,
                self.param_window.B_slider,
                self.param_window.saturation_slider,
                self.param_window.value_slider,
                self.param_window.intensity_slider,
                self.param_window.contrast_slider,
                self.param_window.brightness_slider,
                self.param_window.gamma_slider,
            ]
            for slider in sliders:
                if slider == self.param_window.gamma_slider:
                    slider[1].setValue(100)  # Gamma resetado para 1.00
                else:
                    slider[1].setValue(0)
    def update_image(self):
        if self.edited_image is not None:
            # Resetar para a imagem original
            image = self.original_edited_image.copy().astype(np.float32)
    
            # Obter os valores dos sliders
            R_adjust = self.param_window.R_slider[1].value() / 100
            G_adjust = self.param_window.G_slider[1].value() / 100
            B_adjust = self.param_window.B_slider[1].value() / 100
            saturation_adjust = 1 + (self.param_window.saturation_slider[1].value() / 100)
            value_adjust = self.param_window.value_slider[1].value() / 100
            intensity_adjust = self.param_window.intensity_slider[1].value() / 100
            contrast_adjust = self.param_window.contrast_slider[1].value() / 100
            brightness_adjust = self.param_window.brightness_slider[1].value()/100
            gamma_adjust = self.param_window.gamma_slider[1].value() / 100
    
            # Ajustes nos canais R, G, B
            image[:, :, 0] += R_adjust * 255  # Canal R
            image[:, :, 1] += G_adjust * 255  # Canal G
            image[:, :, 2] += B_adjust * 255  # Canal B
            image = np.clip(image, 0, 255)
    
            # Converter para escala de cinza para saturação e luminosidade
            gray = np.mean(image, axis=2, keepdims=True)
    
            # Ajustar saturação
            image += saturation_adjust * (image - gray)
            image = np.clip(image, 0, 255)
    
            # Ajustar luminosidade (similar ao "Value")
            image += value_adjust * gray
            image = np.clip(image, 0, 255)
    
            # Aplicar intensidade
            image *= (1 + intensity_adjust)
            image = np.clip(image, 0, 255)
    
            # Aplicar contraste
            mean = np.mean(image)
            image = (1 + contrast_adjust) * (image - mean) + mean
            image = np.clip(image, 0, 255)
    
            # Aplicar brilho
            image += brightness_adjust
            image = np.clip(image, 0, 255)
    
            # Aplicar gama
            if gamma_adjust != 1.0:
                image = np.power(image / 255.0, gamma_adjust) * 255
                image = np.clip(image, 0, 255)
    
            # Atualizar a imagem editada usando o resultado calculado
            updated_image = image.astype(np.uint8)
            self.image_window.update_edited_image(updated_image)  # Atualizar a janela de imagem

    
    def save_parameters(self):
        """Salva os parâmetros atuais em um arquivo JSON."""
        save_path, _ = QFileDialog.getSaveFileName(self.param_window, "Save Parameters", "", "JSON Files (*.json)")
        if save_path:
            parameters = {
                "R": self.param_window.R_slider[1].value(),
                "G": self.param_window.G_slider[1].value(),
                "B": self.param_window.B_slider[1].value(),
                "saturation": self.param_window.saturation_slider[1].value(),
                "value": self.param_window.value_slider[1].value(),
                "intensity": self.param_window.intensity_slider[1].value(),
                "contrast": self.param_window.contrast_slider[1].value(),
                "brightness": self.param_window.brightness_slider[1].value(),
                "gamma": self.param_window.gamma_slider[1].value(),
            }
            with open(save_path, 'w') as f:
                json.dump(parameters, f, indent=4)
            print(f"Parameters saved to {save_path}")

    def load_parameters(self):
        """Carrega os parâmetros de um arquivo JSON e atualiza os sliders."""
        load_path, _ = QFileDialog.getOpenFileName(self.param_window, "Load Parameters", "", "JSON Files (*.json)")
        if load_path:
            with open(load_path, 'r') as f:
                parameters = json.load(f)
            # Atualizar sliders com os valores carregados
            self.param_window.R_slider[1].setValue(parameters.get("R", 0))
            self.param_window.G_slider[1].setValue(parameters.get("G", 0))
            self.param_window.B_slider[1].setValue(parameters.get("B", 0))
            self.param_window.saturation_slider[1].setValue(parameters.get("saturation", 0))
            self.param_window.value_slider[1].setValue(parameters.get("value", 0))
            self.param_window.intensity_slider[1].setValue(parameters.get("intensity", 0))
            self.param_window.contrast_slider[1].setValue(parameters.get("contrast", 0))
            self.param_window.brightness_slider[1].setValue(parameters.get("brightness", 0))
            self.param_window.gamma_slider[1].setValue(parameters.get("gamma", 100))
            # Atualizar a imagem após carregar os parâmetros
            self.update_image()
            print(f"Parameters loaded from {load_path}")
        

class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        self.setGeometry(600, 100, 800, 600)

        # Layout principal para as imagens
        main_layout = QHBoxLayout()

        # Painel para a imagem de referência
        self.reference_panel = QLabel()
        self.reference_panel.setAlignment(Qt.AlignCenter)
        self.reference_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Painel para a imagem editada
        self.edited_panel = QLabel()
        self.edited_panel.setAlignment(Qt.AlignCenter)
        self.edited_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Adicionar os painéis ao layout principal
        main_layout.addWidget(self.reference_panel)
        main_layout.addWidget(self.edited_panel)
        self.setLayout(main_layout)

        # Armazenar as imagens originais
        self.reference_image = None
        self.edited_image = None

        # Armazenar níveis de zoom
        self.reference_zoom_level = 1.0
        self.edited_zoom_level = 1.0

    def update_reference_image(self, image):
        self.reference_image = image
        self._update_image_panel(self.reference_panel, self.reference_image, self.reference_zoom_level)

    def update_edited_image(self, image):
        self.edited_image = image
        self._update_image_panel(self.edited_panel, self.edited_image, self.edited_zoom_level)

    def _update_image_panel(self, panel, image, zoom_level):
        if image is not None:
            panel.setPixmap(self.convert_to_pixmap(image, panel.size(), zoom_level))

    def convert_to_pixmap(self, image, size, zoom_level):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Ajustar o tamanho da imagem com base no zoom
        new_width = int(size.width() * zoom_level)
        new_height = int(size.height() * zoom_level)

        return QPixmap.fromImage(q_image).scaled(
            QSize(new_width, new_height), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    def resizeEvent(self, event):
        """Atualizar as imagens ao redimensionar a janela."""
        if self.reference_image is not None:
            self._update_image_panel(self.reference_panel, self.reference_image, self.reference_zoom_level)
        if self.edited_image is not None:
            self._update_image_panel(self.edited_panel, self.edited_image, self.edited_zoom_level)
        super().resizeEvent(event)


class ParameterWindow(QWidget):
    def __init__(self, load_reference_callback, load_edit_callback, reset_callback, update_image_callback, save_parameters_callback, load_parameters_callback):
        super().__init__()
        self.setWindowTitle("Image Parameters")
        self.setGeometry(200, 200, 600, 900)
        # Layout principal
        main_layout = QVBoxLayout()

        # Botões para carregar imagens e resetar
        button_layout = QHBoxLayout()
        load_reference_button = QPushButton("Load Reference Image")
        load_reference_button.clicked.connect(load_reference_callback)
        load_edit_button = QPushButton("Load Image to Edit")
        load_edit_button.clicked.connect(load_edit_callback)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(reset_callback)
        
        # Botões para salvar e carregar parâmetros
        save_button = QPushButton("Save Parameters")
        save_button.clicked.connect(save_parameters_callback)
        load_button = QPushButton("Load Parameters")
        load_button.clicked.connect(load_parameters_callback)
        
        #Botoes no painel
        button_layout.addWidget(load_reference_button)
        button_layout.addWidget(load_edit_button)
        button_layout.addWidget(reset_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(load_button)
        

        # TabWidget para controle de ajustes
        self.tab_widget = QTabWidget()

        # Aba global para ajustes gerais
        global_tab = QWidget()
        global_layout = QVBoxLayout()
        self.R_slider = self.create_slider(-255, 255, 0, "R", update_image_callback)
        self.G_slider = self.create_slider(-255, 255, 0, "G", update_image_callback)
        self.B_slider = self.create_slider(-255, 255, 0, "B", update_image_callback)
        self.saturation_slider = self.create_slider(-200, 200, 0, "Saturatição (%)", update_image_callback)
        self.value_slider = self.create_slider(-100, 100, 0, "Luminosidade (%)", update_image_callback)
        self.intensity_slider = self.create_slider(-100, 100, 0, "Intensidade(%)", update_image_callback)
        self.contrast_slider = self.create_slider(-100, 100, 0, "Contrastr (%)", update_image_callback)
        self.brightness_slider = self.create_slider(-100, 100, 0, "Brilho", update_image_callback)
        self.gamma_slider = self.create_slider(0, 200, 100, "Gama", update_image_callback)
        
        #Adicionar Slider
        global_layout.addLayout(self.gamma_slider[0])
        global_layout.addLayout(self.brightness_slider[0])
        global_layout.addLayout(self.R_slider[0])
        global_layout.addLayout(self.G_slider[0])
        global_layout.addLayout(self.B_slider[0])
        global_layout.addLayout(self.saturation_slider[0])
        global_layout.addLayout(self.value_slider[0])
        global_layout.addLayout(self.intensity_slider[0])
        global_layout.addLayout(self.contrast_slider[0])

        global_tab.setLayout(global_layout)

        # Adicionar aba ao TabWidget
        self.tab_widget.addTab(global_tab, "Global Adjustments")

        # Adicionar ao layout principal
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)

    def create_slider(self, min_value, max_value, default_value, label_text, callback):
        layout = QHBoxLayout()
        label = QLabel(f"{label_text}:")
        label.setFixedWidth(150)
    
        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(default_value)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
        # Label para exibir o valor atual
        value_label = QLabel(f"{default_value / 100:.2f}" if "Gama" in label_text else str(default_value))
        value_label.setFixedWidth(60)
        value_label.setAlignment(Qt.AlignCenter)
    
        def update_value_label(value):
            if "Gama" in label_text:
                value_label.setText(f"{value / 100:.2f}")  # Formato decimal para gama
            elif "%" in label_text:
                value_label.setText(f"{value}%")  # Para sliders em porcentagem
            else:
                value_label.setText(str(value))  # Outros sliders usam valores inteiros
    
        # Conectar eventos
        slider.valueChanged.connect(lambda: update_value_label(slider.value()))
        slider.sliderReleased.connect(callback)
    
        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(value_label)
    
        return layout, slider


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())
