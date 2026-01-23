from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QWidget, QSizePolicy, QTabWidget, QCheckBox
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
        self.setGeometry(200, 200, 600, 900)

        main_layout = QVBoxLayout()

        # Layout dos botoes principais
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

        color_match_button = QPushButton("Color Matching")
        color_match_button.clicked.connect(color_transfer_callback)

        # Adicionar botoes ao layout
        button_layout.addWidget(load_reference_button)
        button_layout.addWidget(load_edit_button)
        button_layout.addWidget(reset_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(load_button)
        button_layout.addWidget(color_match_button)


        self.tab_widget = QTabWidget()

        # ============== ABA 1: Ajustes Globais ==============
        global_tab = QWidget()
        global_layout = QVBoxLayout()

        self.brightness_slider = self.create_slider(-100, 100, 0, "Brilho", update_image_callback)
        self.contrast_slider   = self.create_slider(-100, 100, 0, "Contraste (%)", update_image_callback)
        self.gamma_slider      = self.create_slider(0, 200, 100, "Gama", update_image_callback)
        self.hue_slider        = self.create_slider(-180, 180, 0, "Hue (Matiz)", update_image_callback)
        self.saturation_slider = self.create_slider(-100, 100, 0, "Saturation (HSV)", update_image_callback)
        self.value_slider      = self.create_slider(-100, 100, 0, "Value (HSV)", update_image_callback)

        global_layout.addLayout(self.brightness_slider[0])
        global_layout.addLayout(self.contrast_slider[0])
        global_layout.addLayout(self.gamma_slider[0])
        global_layout.addLayout(self.hue_slider[0])
        global_layout.addLayout(self.saturation_slider[0])
        global_layout.addLayout(self.value_slider[0])

        global_tab.setLayout(global_layout)
        self.tab_widget.addTab(global_tab, "Global Adjustments")

        # ============== ABA 2: Ajustes Avancados ==============
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout()

        self.temperature_slider = self.create_slider(2000, 10000, 6500, "Temperature (K)", update_image_callback)
        self.tint_slider = self.create_slider(-100, 100, 0, "Tint (G<->M)", update_image_callback)
        self.r_gain_slider = self.create_slider(0, 200, 100, "R Gain (%)", update_image_callback)
        self.g_gain_slider = self.create_slider(0, 200, 100, "G Gain (%)", update_image_callback)
        self.b_gain_slider = self.create_slider(0, 200, 100, "B Gain (%)", update_image_callback)

        # Checkbox e slider para Gaussian
        self.gaussian_checkbox = QCheckBox("Apply Gaussian Blur")
        self.gaussian_checkbox.setChecked(False)
        self.gaussian_checkbox.stateChanged.connect(update_image_callback)
        advanced_layout.addWidget(self.gaussian_checkbox)

        self.gaussian_sigma_slider = self.create_slider(0, 50, 0, "Gaussian Sigma (0..50)", update_image_callback)
        advanced_layout.addLayout(self.gaussian_sigma_slider[0])

        # Checkbox e slider para Dehaze
        self.dehaze_checkbox = QCheckBox("Apply Dehaze")
        self.dehaze_checkbox.setChecked(False)
        self.dehaze_checkbox.stateChanged.connect(update_image_callback)
        advanced_layout.addWidget(self.dehaze_checkbox)

        self.dehaze_strength_slider = self.create_slider(0, 100, 0, "Dehaze Strength", update_image_callback)
        advanced_layout.addLayout(self.dehaze_strength_slider[0])

        advanced_layout.addLayout(self.temperature_slider[0])
        advanced_layout.addLayout(self.tint_slider[0])
        advanced_layout.addLayout(self.r_gain_slider[0])
        advanced_layout.addLayout(self.g_gain_slider[0])
        advanced_layout.addLayout(self.b_gain_slider[0])

        advanced_tab.setLayout(advanced_layout)
        self.tab_widget.addTab(advanced_tab, "Advanced Color")

        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)

    def create_slider(self, min_value, max_value, default_value, label_text, callback):
        layout = QHBoxLayout()
        label = QLabel(f"{label_text}:")
        label.setFixedWidth(150)
    
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(default_value)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
        if "Gama" in label_text:
            initial_text = f"{default_value / 100:.2f}"
        elif "%" in label_text:
            initial_text = f"{default_value}%"
        else:
            initial_text = str(default_value)

        value_label = QLabel(initial_text)
        value_label.setFixedWidth(60)
        value_label.setAlignment(Qt.AlignCenter)
    
        def update_value_label(value):
            if "Gama" in label_text:
                value_label.setText(f"{value / 100:.2f}")
            elif "%" in label_text:
                value_label.setText(f"{value}%")
            else:
                value_label.setText(str(value))

        # Conectamos o valueChanged somente para atualizar o texto
        slider.valueChanged.connect(lambda: update_value_label(slider.value()))
        # Conectamos sliderReleased para chamar a callback (evita recalcular a cada pixel)
        slider.sliderReleased.connect(callback)
    
        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(value_label)
    
        return (layout, slider)
