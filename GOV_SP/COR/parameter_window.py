from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QWidget, QSizePolicy, QTabWidget
)
from PyQt5.QtCore import Qt


class ParameterWindow(QWidget):
    def __init__(
        self,
        load_reference_callback,
        load_edit_callback,
        reset_callback,
        update_image_callback,
        save_parameters_callback,
        load_parameters_callback,
        color_transfer_callback,
    ):
        super().__init__()
        self.setWindowTitle("Image Parameters")
        self.setGeometry(200, 200, 600, 400)

        main_layout = QVBoxLayout()

        # ====== BOTÕES PRINCIPAIS ======
        button_layout = QHBoxLayout()

        load_reference_button = QPushButton("Load Reference Image")
        load_reference_button.clicked.connect(load_reference_callback)

        load_edit_button = QPushButton("Load Image to Edit")
        load_edit_button.clicked.connect(load_edit_callback)

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(reset_callback)

        save_button = QPushButton("Save Parameters")
        save_button.clicked.connect(save_parameters_callback)

        load_button = QPushButton("Load Parameters")
        load_button.clicked.connect(load_parameters_callback)

        color_match_button = QPushButton("Apply Color Match")
        color_match_button.clicked.connect(color_transfer_callback)

        button_layout.addWidget(load_reference_button)
        button_layout.addWidget(load_edit_button)
        button_layout.addWidget(reset_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(load_button)
        button_layout.addWidget(color_match_button)

        # ====== ABA ÚNICA: COLOR MATCH ======
        self.tab_widget = QTabWidget()
        color_tab = QWidget()
        color_layout = QVBoxLayout()

        # Slider de força do Color Match (0–100%)
        self.strength_slider = self.create_slider(
            0, 100, 70, "Color Match Strength (%)", update_image_callback
        )
        color_layout.addLayout(self.strength_slider[0])

        color_tab.setLayout(color_layout)
        self.tab_widget.addTab(color_tab, "Color Match")

        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)

    # ====== FUNÇÃO AUXILIAR PARA CRIAR SLIDER ======
    def create_slider(self, min_value, max_value, default_value, label_text, callback):
        layout = QHBoxLayout()
        label = QLabel(f"{label_text}:")
        label.setFixedWidth(180)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(default_value)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        value_label = QLabel(f"{default_value}%")
        value_label.setFixedWidth(60)
        value_label.setAlignment(Qt.AlignCenter)

        def update_value_label(value):
            value_label.setText(f"{value}%")

        slider.valueChanged.connect(lambda: update_value_label(slider.value()))
        slider.sliderReleased.connect(callback)

        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(value_label)

        return (layout, slider)
