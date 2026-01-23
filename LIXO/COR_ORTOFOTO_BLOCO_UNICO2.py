# 1. Imports
import sys
import cv2
import os
from osgeo import gdal, osr
import numpy as np
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QWidget, QFileDialog, QProgressBar, QTabWidget, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 2. Funções Auxiliares
def save_image_with_epsg(image, output_path, geotransform=None, projection=None):
    """Salva uma imagem com georreferenciamento especificado."""
    height, width, channels = image.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, width, height, channels, gdal.GDT_Byte)
    
    if projection:
        dataset.SetProjection(projection)
    if geotransform:
        dataset.SetGeoTransform(geotransform)
    
    for i in range(channels):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(image[:, :, i])
    
    dataset.FlushCache()
    dataset = None

# 3. Classes de Threads
class ImageProcessingThread(QThread):
    progress_updated = pyqtSignal(int)
    processing_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, input_folder, output_folder, apply_parameters_callback):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.apply_parameters_callback = apply_parameters_callback
        self.stop_requested = False

    def run(self):
        try:
            file_list = [f for f in os.listdir(self.input_folder) if os.path.isfile(os.path.join(self.input_folder, f))]
         
            for i, filename in enumerate(file_list, start=1):
                if self.stop_requested:
                    break

                input_path = os.path.join(self.input_folder, filename)
                original_dataset = gdal.Open(input_path)
                if original_dataset is None:
                    self.error_occurred.emit(f"Failed to open: {input_path}")
                    continue

                geotransform = original_dataset.GetGeoTransform()
                projection = original_dataset.GetProjection()

                image = cv2.imread(input_path)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    edited_image = self.apply_parameters_callback(image_rgb)
                    output_path = os.path.join(self.output_folder, filename).replace(".jpg", ".tif").replace(".png", ".tif")
                    save_image_with_epsg(edited_image, output_path, geotransform=geotransform, projection=projection)

                self.progress_updated.emit(i)

            self.processing_finished.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))

# 4. Classes Principais da Interface
class MainWindow:
    def __init__(self):
        # Inicialização da interface e componentes
        self.image_window = ImageWindow()
        self.histogram_window = HistogramWindow()
        self.param_window = ParameterWindow(
            self.load_reference_image,
            self.load_image_to_edit,
            self.reset_image,
            self.update_image,
            self.save_parameters,
            self.load_parameters,
        )
        self.progress_bar = QProgressBar()
        self.stop_button = QPushButton("Stop Processing")
        self.processing_thread = None
        self.input_folder = None
        self.output_folder = None
        self.configure_ui()

    def configure_ui(self):
        # Configuração da barra de progresso e botão de parar
        self.param_window.layout().addWidget(self.progress_bar)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_processing)
        self.param_window.layout().addWidget(self.stop_button)

        # Exibir janelas
        self.image_window.show()
        self.histogram_window.show()
        self.param_window.show()

    def process_images(self):
        # Configurar e iniciar a thread de processamento
        self.processing_thread = ImageProcessingThread(
            self.input_folder, self.output_folder, self.apply_current_parameters
        )
        self.processing_thread.progress_updated.connect(self.progress_bar.setValue)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.error_occurred.connect(self.on_processing_error)

        file_count = len([f for f in os.listdir(self.input_folder) if os.path.isfile(os.path.join(self.input_folder, f))])
        self.progress_bar.setMaximum(file_count)
        self.progress_bar.setValue(0)

        self.stop_button.setEnabled(True)
        self.processing_thread.start()

    def stop_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop_requested = True
            self.processing_thread.wait()
            self.stop_button.setEnabled(False)

    def on_processing_finished(self):
        self.stop_button.setEnabled(False)
        print("Processing complete.")

    def on_processing_error(self, error_message):
        print(f"Error: {error_message}")

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
    
        # Atualizar valor do rótulo dinamicamente
        def update_value_label(value):
            if "Gama" in label_text:
                value_label.setText(f"{value / 100:.2f}")  # Formato decimal (0.00 a 2.00)
            else:
                value_label.setText(str(value))  # Outros sliders usam valores inteiros
    
        # Conectar eventos
        slider.valueChanged.connect(lambda: update_value_label(slider.value()))
        slider.sliderReleased.connect(callback)
    
        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(value_label)
    
        return layout, slider

class HistogramWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Histograms")
        self.setGeometry(100, 100, 800, 400)

        layout = QHBoxLayout()
        self.reference_histogram = HistogramWidget()
        self.edited_histogram = HistogramWidget()
        layout.addWidget(self.reference_histogram)
        layout.addWidget(self.edited_histogram)
        self.setLayout(layout)

    def update_reference_histogram(self, image):
        self.reference_histogram.update_histogram(image)

    def update_edited_histogram(self, image):
        self.edited_histogram.update_histogram(image)

class HistogramWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_histogram(self, image):
        if image is not None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            colors = ['r', 'g', 'b']
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                ax.plot(hist, color=color)
            ax.set_xlim([0, 256])
            ax.set_title("Histogram")
            self.canvas.draw()

# 5. Execução Principal
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())
