"""
Config: layers and constants.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

# Verbosidade
VERBOSE = True

# Simbologia fixa
USE_FIXED_SIMBOLOGIA = True
FIXED_SIMBOLOGIA_PATH = r"\\192.168.2.29\d\2304_REURB_SP\SIMBOLOGIA.dxf"
SIMBOLOGIA_DEFAULT_PATH = FIXED_SIMBOLOGIA_PATH

# ---------------- Layers de dados (DXF de entrada) ----------------
LAYER_LOTES = "LOTE"
LAYER_EDIF = "EDIF"
LAYER_EIXO_SETAS = "EIXO"  # para setas de drenagem
LAYER_PERIMETRO = "PER_INTERESSE"
LAYER_PER_LEVANTAMENTO = "PER_LEVANTAMENTO"  # recorte das curvas de nivel

# Eixos de via (alias explicito)
LAYER_EIXO_VIA = "EIXO"

# Layers com linhas que podem orientar rotacao de blocos/textos
ROTATION_LAYERS = {
    "VIA", "EIXO",
}

# Bordas de pista (meio-fio). Para medicoes, usamos a COM_GUIA.
LAYERS_PISTA_BORDA = {"VIA"}

# ---------------- Layers de saida ----------------
LAYER_SETAS_SAIDA = "TOP_SISTVIA"
LAYER_SOLEIRA_BLOCO = "EDIF_COTA_SOLEIRA"
LAYER_SOLEIRA_NUM_PAV = "TOP_EDIF_NUM_PAV"
LAYER_SOLEIRA_AREA = "TOP_AREA_LOTE"

# Curvas de nivel
LAYER_CURVA_INTER = "HM_CURVA_NIV_INTERMEDIARIA"
LAYER_CURVA_MESTRA = "HM_CURVA_NIV_MESTRA"
LAYER_CURVA_ROTULO = "TOP_CURVA_NIV"

# Vertices/Tabela/Ordinate
LAYER_VERTICE_PTO = "HM_VERTICES_PTO"
LAYER_VERTICE_TXT = "TOP_VERTICE"
LAYER_TABELA = "TOP_TABELA"
LAYER_ORDINATE = "TOP_AZIMUTE"

# Medicoes de vias
LAYER_VIA_MEDIDA = "TOP_COTAS_VIARIO"
LAYER_VIA_NOME = "TOP_SISTVIA"

# ---------------- Blocos / Estilos ----------------
STYLE_TEXTO = "Arial"
BLOCO_SETA = "SETA_VIA"
BLOCO_SOLEIRA_POS = "SOLEIRA1"
BLOCO_VERTICE = "VERTICE"
