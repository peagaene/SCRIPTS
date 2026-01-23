# config.py
from __future__ import annotations
from dataclasses import dataclass

# Verbosidade
VERBOSE = True

# Simbologia fixa
USE_FIXED_SIMBOLOGIA  = True
FIXED_SIMBOLOGIA_PATH = r"D:\2304_REURB_SP\SIMBOLOGIA.dxf"

SIMBOLOGIA_DEFAULT_PATH = FIXED_SIMBOLOGIA_PATH

# ---------------- Layers de dados (DXF de entrada) ----------------
LAYER_LOTES        = "LOTE"
LAYER_EDIF         = "EDIF"
LAYER_EIXO_SETAS   = "EIXO"      # para setas de drenagem
LAYER_PERIMETRO    = "PER_INTERESSE"
LAYER_PER_LEVANTAMENTO = "PER_LEVANTAMENTO"  # recorte das curvas de nível

# Eixos de via (alias explícito)
LAYER_EIXO_VIA     = "EIXO"

# Layers com linhas que podem orientar rotação de blocos/textos
ROTATION_LAYERS = {
    "VIA", "EIXO",
}

# Bordas de pista (meio-fio). Para medições, usamos a COM_GUIA.
LAYERS_PISTA_BORDA = {"VIA"}

# ---------------- Layers de saída ----------------
LAYER_SETAS_SAIDA   = "TOP_SISTVIA"
LAYER_SOLEIRA_BLOCO = "EDIF_COTA_SOLEIRA"
LAYER_SOLEIRA_NUM_PAV = "TOP_EDIF_NUM_PAV"
LAYER_SOLEIRA_AREA  = "TOP_AREA_LOTE"

# Curvas de nível
LAYER_CURVA_INTER   = "HM_CURVA_NIV_INTERMEDIARIA"
LAYER_CURVA_MESTRA  = "HM_CURVA_NIV_MESTRA"
LAYER_CURVA_ROTULO  = "TOP_CURVA_NIV"

# Curvas de nível REURB

# Vértices/Tabela/Ordinate
LAYER_VERTICE_PTO   = "HM_VERTICES_PTO"
LAYER_VERTICE_TXT   = "TOP_VERTICE"
LAYER_TABELA        = "TOP_TABELA"
LAYER_ORDINATE      = "TOP_AZIMUTE"

# Medições de vias
LAYER_VIA_MEDIDA = "TOP_COTAS_VIARIO"
LAYER_VIA_NOME   = "TOP_SISTVIA"

# Layers REURB específicos

# ---------------- Blocos / Estilos ----------------
STYLE_TEXTO       = "Arial"
BLOCO_SETA        = "SETA_VIA"
BLOCO_SOLEIRA_POS = "SOLEIRA1"
BLOCO_VERTICE     = "VERTICE"

# ---------------- Mapeamentos (se usados em processing.py) ----------------
# ATENÇÃO: o leitor uppercasa o 'type', então use SOMENTE chaves em MAIÚSCULAS.
TYPE_TO_LAYER = {
    # iluminação / energia (exemplos seus)
    'PA': 'ELET_POSTE_ALTA_TENSAO_LUMINARIA',
    'PI': 'ELET_POSTE_ALTA_TENSAO_LUMINARIA',
    'PFI': 'ELET_POSTE_ALTA_TENSAO',
    # TUBO DE TELEFONIA / TV (exemplos seus)
    'PVTEL': 'INFRA_PVT', 'PVT': 'INFRA_PVT', 'AEPVTEL': 'INFRA_PVT',
    # PV de água pluvial
    'PVA':'INFRA_PVAP', 'PVAP':'INFRA_PVAP', 'AEPVA':'INFRA_PVAP',
    # >>> CORREÇÃO: AEPVE deve ir para INFRA_PVE (antes estava PVAP)
    'AEPVE': 'INFRA_PVE',  'ES': 'INFRA_PVE',  'PVE':'INFRA_PVE',
    'AEPVPD':'INFRA_PVE',  'PV': 'INFRA_PVE',
    # Boca de lobo (variações)
    'AEBO': 'INFRA_BOCA_LOBO', 'AEBO1': 'INFRA_BOCA_LOBO', 'AEBO2': 'INFRA_BOCA_LOBO', 'AEBO3': 'INFRA_BOCA_LOBO',
    'AEBO.1': 'INFRA_BOCA_LOBO', 'AEBO.2': 'INFRA_BOCA_LOBO',
    'BL1': 'INFRA_BOCA_LOBO', 'BL2': 'INFRA_BOCA_LOBO', 'BL3': 'INFRA_BOCA_LOBO',
    # Boca de leão (variações)
    'AEBE': 'INFRA_BOCA_LEAO', 'AEBE1': 'INFRA_BOCA_LEAO', 'AEBE2': 'INFRA_BOCA_LEAO', 'AEBE3': 'INFRA_BOCA_LEAO',
    'AEBE.1': 'INFRA_BOCA_LEAO', 'AEBE.2': 'INFRA_BOCA_LEAO', 'AEBE.3': 'INFRA_BOCA_LEAO', 'AEB': 'INFRA_BOCA_LEAO',
    # >>> NOVO: trate AEBL genérico (sem número)
    'AEBL':  'INFRA_BOCA_LOBO', 'AEBL1': 'INFRA_BOCA_LOBO', 'AEBL2': 'INFRA_BOCA_LOBO', 'AEBL3': 'INFRA_BOCA_LOBO',
    # Árvores (normalização no wrapper para 'ARVORE')
    'ARVORE': 'VEG_ARVORE_ISOLADA',
}
LAYER_TO_BLOCK = {
    'INFRA_PVE':  'INFRA_PVE',
    'INFRA_PVAP':'INFRA_PVAP',
    'INFRA_PVT': 'INFRA_PVT',
    'INFRA_BOCA_LOBO': 'BOCA_LOBO',
    'INFRA_BOCA_LEAO': 'BOCA_DE_LEAO',
    'ELET_POSTE_ALTA_TENSAO_LUMINARIA': 'POSTE_ILUMI',
    'ELET_POSTE_ALTA_TENSAO': 'POSTE_TENSAO',
    'MOB_MOBILIARIO_URBANO': 'MOB_URBANO_PT_ONIBUS',
    'INFRA_BUEIRO': 'COD119',
    'VEG_ARVORE_ISOLADA': 'ARVORE',
}


IGNORAR_SEM_LOG = {"DIVL", "VIELA"}

# ---------------- Parâmetros ----------------
@dataclass
class Params:
    # Setas de drenagem
    min_seg_len: float      = 20.0
    offset_seta: float      = 0.4
    delta_interp: float     = 0.1
    dist_busca_rot: float   = 8.0
    setas_por_trecho: int   = 2   # <<< agora este valor é respeitado no desenho
    
    # Controle inteligente de setas de drenagem
    setas_buffer_distancia: float = 5.0      # Buffer mínimo entre setas (metros)
    setas_seg_curto_max: int = 3             # Máximo de setas para segmentos curtos
    setas_seg_medio_max: int = 4             # Máximo de setas para segmentos médios
    setas_seg_longo_max: int = 5             # Máximo de setas para segmentos longos
    setas_seg_curto_threshold: float = 30.0  # Limite para segmento curto (metros)
    setas_seg_medio_threshold: float = 60.0  # Limite para segmento médio (metros)

    # Textos / Nº / Pav / Área do lote
    altura_texto_soleira: float = 0.75
    altura_texto_area: float    = 0.75
    line_spacing_factor: float  = 1.2

    # Geometria / regras
    buffer_lote_edif: float = 0.20
    max_dist_lote: float    = 5.0

    # Perímetro & anotações
    altura_texto_P: float       = 0.75
    p_label_offset_m: float   = 1.8
    p_label_offset_step: float = 0.5
    altura_texto_tabela: float  = 2.0
    tabela_cell_w: float        = 35.0
    tabela_cell_h: float        = 6.0
    tabela_offset_x: float      = 120.0
    tabela_offset_y: float      = 0.0
    dimtxt_ordinate: float      = 0.5
    dimasz_ordinate: float      = 0.2

    # Curvas de nível
    curva_equidist: float        = 1.0     # m
    curva_mestra_cada: int       = 5       # mestra a cada N m
    altura_texto_curva: float    = 0.4     # altura do rótulo
    curva_char_w_factor: float   = 0.60    # largura ~ h * fator * n_chars
    curva_gap_margin: float      = 0.50    # margem auxiliar (compatibilidade)
    curva_label_step_m: float    = 80.0    # distância entre rótulos na mestra (m)
    curva_min_len: float         = 10.0    # descarta linhas curtas (m)
    curva_min_area: float        = 20.0    # descarta anéis muito pequenos (m²)
    curva_smooth_sigma_px: float = 1.0     # suavização (sigma em pixels; 0 desliga)
    curva_label_offset_m: float  = 0.25    # afastamento perpendicular do texto
    curva_label_gap_enabled: bool = False  # evita cortar a curva sob o texto

    # Medições de vias
    altura_texto_via: float = 0.75
    via_offset_texto: float = 0.50
    via_cross_span: float   = 80.0
    
    # Configurações gerais de soleira/numeração
    rotacionar_numero_casa: bool = True


# Objeto global de parâmetros (sem painel UI)
GLOBAL_PARAMS = Params()
