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
        self.setGeometry(600, 100, 1000, 600)  # Larger window for controls

        # Main layout (horizontal) to split reference and edited areas
        main_layout = QHBoxLayout()

        # ----- Reference image area -----
        ref_layout = QVBoxLayout()
        zoom_ref_layout = QHBoxLayout()
        self.btn_zoom_in_ref = QPushButton("Zoom In Ref")
        self.btn_zoom_out_ref = QPushButton("Zoom Out Ref")
        self.btn_zoom_in_ref.clicked.connect(self.zoom_in_reference)
        self.btn_zoom_out_ref.clicked.connect(self.zoom_out_reference)
        zoom_ref_layout.addWidget(self.btn_zoom_in_ref)
        zoom_ref_layout.addWidget(self.btn_zoom_out_ref)
        ref_layout.addLayout(zoom_ref_layout)

        self.reference_panel = QLabel()
        self.reference_panel.setAlignment(Qt.AlignCenter)
        self.reference_panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.scroll_area_reference = QScrollArea()
        self.scroll_area_reference.setWidgetResizable(False)
        self.scroll_area_reference.setWidget(self.reference_panel)
        ref_layout.addWidget(self.scroll_area_reference)

        # ----- Edited image area -----
        edit_layout = QVBoxLayout()
        zoom_edit_layout = QHBoxLayout()
        self.btn_zoom_in_edit = QPushButton("Zoom In Edit")
        self.btn_zoom_out_edit = QPushButton("Zoom Out Edit")
        self.btn_zoom_in_edit.clicked.connect(self.zoom_in_edited)
        self.btn_zoom_out_edit.clicked.connect(self.zoom_out_edited)
        zoom_edit_layout.addWidget(self.btn_zoom_in_edit)
        zoom_edit_layout.addWidget(self.btn_zoom_out_edit)
        edit_layout.addLayout(zoom_edit_layout)

        self.edited_panel = QLabel()
        self.edited_panel.setAlignment(Qt.AlignCenter)
        self.edited_panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.scroll_area_edited = QScrollArea()
        self.scroll_area_edited.setWidgetResizable(False)
        self.scroll_area_edited.setWidget(self.edited_panel)
        edit_layout.addWidget(self.scroll_area_edited)

        main_layout.addLayout(ref_layout)
        main_layout.addLayout(edit_layout)
        self.setLayout(main_layout)

        # Store original high-res images
        self.reference_image_orig = None
        self.edited_image_orig = None

        # Fit factor and manual zoom multiplier
        self.reference_fit_factor = 1.0
        self.edited_fit_factor = 1.0
        self.reference_manual_zoom = 1.0
        self.edited_manual_zoom = 1.0

    def fit_zoom_factor(self, image, widget):
        """
        Compute a scale factor to fit the image into the widget viewport while
        keeping aspect ratio.
        """
        panel_width = widget.width()
        panel_height = widget.height()
        h, w, _ = image.shape
        scale_w = panel_width / w
        scale_h = panel_height / h
        return min(scale_w, scale_h)

    def update_reference_image(self, image):
        """
        Update the reference image and reset manual zoom to 1.0.
        """
        self.reference_image_orig = image.copy()
        self.reference_fit_factor = self.fit_zoom_factor(image, self.scroll_area_reference.viewport())
        self.reference_manual_zoom = 1.0
        effective = self.reference_fit_factor * self.reference_manual_zoom
        self._update_image_panel(self.reference_panel, self.reference_image_orig, effective)

    def update_edited_image(self, image):
        """
        Update the edited image and reset manual zoom to 1.0.
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
            panel.resize(pixmap.size())

    def convert_to_pixmap(self, image, zoom_level):
        """
        Convert a NumPy RGB image into QPixmap and apply zoom.
        """
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        new_width = int(w * zoom_level)
        new_height = int(h * zoom_level)
        return QPixmap.fromImage(q_image).scaled(
            QSize(new_width, new_height), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    # Zoom methods for reference image
    def zoom_in_reference(self):
        self.reference_manual_zoom *= 1.2
        effective = self.reference_fit_factor * self.reference_manual_zoom
        self._update_image_panel(self.reference_panel, self.reference_image_orig, effective)

    def zoom_out_reference(self):
        self.reference_manual_zoom /= 1.2
        effective = self.reference_fit_factor * self.reference_manual_zoom
        self._update_image_panel(self.reference_panel, self.reference_image_orig, effective)

    # Zoom methods for edited image
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
        Recalculate fit factors on resize, preserving manual zoom.
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
        # Hide the window instead of closing
        event.ignore()
        self.hide()
