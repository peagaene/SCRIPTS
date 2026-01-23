from PyQt5.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QWidget, QSizePolicy,
    QPushButton, QScrollArea
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage

class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        self.setGeometry(600, 100, 1000, 600)  # Janela maior para acomodar controles

        # Layout principal (horizontal) para dividir as áreas de referência e editada
        main_layout = QHBoxLayout()

        # ----- Área da imagem de referência -----
        ref_layout = QVBoxLayout()
        # Layout dos botões de zoom para a referência
        zoom_ref_layout = QHBoxLayout()
        self.btn_zoom_in_ref = QPushButton("Zoom In Ref")
        self.btn_zoom_out_ref = QPushButton("Zoom Out Ref")
        self.btn_zoom_in_ref.clicked.connect(self.zoom_in_reference)
        self.btn_zoom_out_ref.clicked.connect(self.zoom_out_reference)
        zoom_ref_layout.addWidget(self.btn_zoom_in_ref)
        zoom_ref_layout.addWidget(self.btn_zoom_out_ref)
        ref_layout.addLayout(zoom_ref_layout)

        # QLabel para a imagem de referência, encapsulada em QScrollArea
        self.reference_panel = QLabel()
        self.reference_panel.setAlignment(Qt.AlignCenter)
        # Não usamos scaledContents para que o QLabel mantenha o tamanho do pixmap
        self.reference_panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.scroll_area_reference = QScrollArea()
        # Importante: desabilitar o widget resizable para que o tamanho do QLabel (conteúdo) seja usado
        self.scroll_area_reference.setWidgetResizable(False)
        self.scroll_area_reference.setWidget(self.reference_panel)
        ref_layout.addWidget(self.scroll_area_reference)

        # ----- Área da imagem editada -----
        edit_layout = QVBoxLayout()
        # Layout dos botões de zoom para a imagem editada
        zoom_edit_layout = QHBoxLayout()
        self.btn_zoom_in_edit = QPushButton("Zoom In Edit")
        self.btn_zoom_out_edit = QPushButton("Zoom Out Edit")
        self.btn_zoom_in_edit.clicked.connect(self.zoom_in_edited)
        self.btn_zoom_out_edit.clicked.connect(self.zoom_out_edited)
        zoom_edit_layout.addWidget(self.btn_zoom_in_edit)
        zoom_edit_layout.addWidget(self.btn_zoom_out_edit)
        edit_layout.addLayout(zoom_edit_layout)

        # QLabel para a imagem editada, encapsulada em QScrollArea
        self.edited_panel = QLabel()
        self.edited_panel.setAlignment(Qt.AlignCenter)
        self.edited_panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.scroll_area_edited = QScrollArea()
        self.scroll_area_edited.setWidgetResizable(False)
        self.scroll_area_edited.setWidget(self.edited_panel)
        edit_layout.addWidget(self.scroll_area_edited)

        # Adiciona os layouts de referência e edição ao layout principal
        main_layout.addLayout(ref_layout)
        main_layout.addLayout(edit_layout)
        self.setLayout(main_layout)

        # Atributos para armazenar as imagens originais (em alta resolução)
        self.reference_image_orig = None
        self.edited_image_orig = None

        # Fatores de ajuste (fit) e zoom manual (multiplicador)
        self.reference_fit_factor = 1.0
        self.edited_fit_factor = 1.0
        self.reference_manual_zoom = 1.0
        self.edited_manual_zoom = 1.0

    def fit_zoom_factor(self, image, widget):
        """
        Calcula o fator de escala para ajustar a imagem às dimensões do widget,
        mantendo a proporção. Aqui, o widget é o viewport do QScrollArea.
        """
        panel_width = widget.width()
        panel_height = widget.height()
        h, w, _ = image.shape
        scale_w = panel_width / w
        scale_h = panel_height / h
        return min(scale_w, scale_h)

    def update_reference_image(self, image):
        """
        Atualiza a imagem de referência:
          - Armazena a imagem original.
          - Calcula o fator de "fit" com base no viewport da área de scroll.
          - Reseta o zoom manual para 1.0.
          - Atualiza o painel com a imagem escalada.
        """
        self.reference_image_orig = image.copy()
        self.reference_fit_factor = self.fit_zoom_factor(image, self.scroll_area_reference.viewport())
        self.reference_manual_zoom = 1.0
        effective = self.reference_fit_factor * self.reference_manual_zoom
        self._update_image_panel(self.reference_panel, self.reference_image_orig, effective)

    def update_edited_image(self, image):
        """
        Atualiza a imagem editada:
          - Armazena a imagem original.
          - Calcula o fator de "fit" com base no viewport da área de scroll.
          - Reseta o zoom manual para 1.0.
          - Atualiza o painel com a imagem escalada.
        """
        self.edited_image_orig = image.copy()
        self.edited_fit_factor = self.fit_zoom_factor(image, self.scroll_area_edited.viewport())
        self.edited_manual_zoom = 1.0
        effective = self.edited_fit_factor * self.edited_manual_zoom
        self._update_image_panel(self.edited_panel, self.edited_image_orig, effective)

    def _update_image_panel(self, panel, image, zoom_level):
        if image is not None:
            pixmap = self.convert_to_pixmap(image, zoom_level)
            panel.setPixmap(pixmap)
            panel.resize(pixmap.size())  # Define o tamanho do QLabel para o tamanho do pixmap

    def convert_to_pixmap(self, image, zoom_level):
        """
        Converte um array NumPy (imagem no formato RGB) em QPixmap,
        aplicando o fator de zoom sobre as dimensões originais da imagem.
        """
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        new_width = int(w * zoom_level)
        new_height = int(h * zoom_level)
        return QPixmap.fromImage(q_image).scaled(
            QSize(new_width, new_height), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    # Métodos de Zoom para a imagem de referência
    def zoom_in_reference(self):
        self.reference_manual_zoom *= 1.2
        effective = self.reference_fit_factor * self.reference_manual_zoom
        self._update_image_panel(self.reference_panel, self.reference_image_orig, effective)

    def zoom_out_reference(self):
        self.reference_manual_zoom /= 1.2
        effective = self.reference_fit_factor * self.reference_manual_zoom
        self._update_image_panel(self.reference_panel, self.reference_image_orig, effective)

    # Métodos de Zoom para a imagem editada
    def zoom_in_edited(self):
        self.edited_manual_zoom *= 1.2
        effective = self.edited_fit_factor * self.edited_manual_zoom
        self._update_image_panel(self.edited_panel, self.edited_image_orig, effective)

    def zoom_out_edited(self):
        self.edited_manual_zoom /= 1.2
        effective = self.edited_fit_factor * self.edited_manual_zoom
        self._update_image_panel(self.edited_panel, self.edited_image_orig, effective)

    def resizeEvent(self, event):
        """
        Ao redimensionar a janela, recalcula os fatores de "fit" (usando o viewport)
        e atualiza os painéis, preservando o zoom manual aplicado.
        """
        if self.reference_image_orig is not None:
            self.reference_fit_factor = self.fit_zoom_factor(self.reference_image_orig, self.scroll_area_reference.viewport())
            effective = self.reference_fit_factor * self.reference_manual_zoom
            self._update_image_panel(self.reference_panel, self.reference_image_orig, effective)
        if self.edited_image_orig is not None:
            self.edited_fit_factor = self.fit_zoom_factor(self.edited_image_orig, self.scroll_area_edited.viewport())
            effective = self.edited_fit_factor * self.edited_manual_zoom
            self._update_image_panel(self.edited_panel, self.edited_image_orig, effective)
        super().resizeEvent(event)
        
        def closeEvent(self, event):
            # Em vez de fechar a janela, apenas a oculta
            event.ignore()
            self.hide()    
