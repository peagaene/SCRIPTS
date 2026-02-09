"""
Config: Params dataclass and numeric defaults.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometricTolerances:
    """Tolerancias geometricas documentadas."""

    EPSILON: float = 1e-6
    SNAP_TOLERANCE: float = 0.05
    ANGLE_TOLERANCE_DEG: float = 5.0

    MIN_SEGMENT_LENGTH: float = 0.5
    MIN_POLYGON_AREA: float = 1.0

    TEXT_CHAR_WIDTH_FACTOR: float = 0.6
    TEXT_MIN_HEIGHT: float = 0.1

    DIM_MIN_LENGTH: float = 1e-4
    DIM_ARROW_SIZE_FACTOR: float = 0.5

    OFFSET_LABEL_DEFAULT: float = 0.60
    OFFSET_DIM_GAP: float = 0.60

    MERGE_TOLERANCE: float = 0.01
    KEY_TOLERANCE_MIN: float = 1e-3
    MAX_MERGE_ITERS: int = 5
    DEDUP_ROUND_SCALE: int = 1000


@dataclass
class Params:
    # Setas de drenagem
    min_seg_len: float = 20.0
    offset_seta: float = 0.4
    delta_interp: float = 0.1
    dist_busca_rot: float = 8.0
    setas_por_trecho: int = 2

    # Controle inteligente de setas de drenagem
    setas_buffer_distancia: float = 5.0
    setas_seg_curto_max: int = 3
    setas_seg_medio_max: int = 4
    setas_seg_longo_max: int = 5
    setas_seg_curto_threshold: float = 30.0
    setas_seg_medio_threshold: float = 60.0

    # Textos / No / Pav / Area do lote
    altura_texto_soleira: float = 0.75
    altura_texto_area: float = 0.75
    line_spacing_factor: float = 1.2

    # Geometria / regras
    buffer_lote_edif: float = 0.20
    max_dist_lote: float = 5.0

    # Perimetro & anotacoes
    altura_texto_P: float = 1.0
    p_label_offset_m: float = 1.8
    p_label_offset_step: float = 0.5
    altura_texto_tabela: float = 2.0
    tabela_cell_w: float = 35.0
    area_table_cell_w: float = 75.0
    tabela_cell_h: float = 6.0
    tabela_header_h: float = 12.0
    tabela_col_w_segmento: float = 25.0
    tabela_col_w_distancia: float = 25.0
    tabela_col_w_azimute: float = 25.0
    tabela_col_w_ponto: float = 25.0
    tabela_col_w_e: float = 25.0
    tabela_col_w_n: float = 25.0
    tabela_offset_x: float = 120.0
    tabela_offset_y: float = 0.0
    dimtxt_ordinate: float = 0.5
    dimasz_ordinate: float = 0.2
    lote_dim_offset_m: float = 0.60
    lote_dim_min_len_m: float = 0.0
    lote_dim_min_spacing_m: float = 0.0
    lote_dim_snap_tol_m: float = 0.20

    # Curvas de nivel
    curva_equidist: float = 1.0
    curva_mestra_cada: int = 5
    altura_texto_curva: float = 0.75
    curva_char_w_factor: float = 0.60
    curva_gap_margin: float = 0.50
    curva_label_step_m: float = 80.0
    curva_min_len: float = 10.0
    curva_min_area: float = 20.0
    curva_smooth_sigma_px: float = 1.0
    curva_label_offset_m: float = 0.25
    curva_label_gap_enabled: bool = False

    # Medicoes de vias
    altura_texto_via: float = 0.75
    via_offset_texto: float = 0.50
    via_cross_span: float = 80.0
    via_dim_gap_m: float = 0.60
    via_dim_min_len_m: float = 12.0
    via_dim_min_spacing_m: float = 25.0
    via_dim_max_por_trecho: int = 2
    via_dim_max_dist_m: float = 20.0
    via_dim_min_sep_area_m: float = 10.0
    via_dim_equal_tol_m: float = 0.05

    # Configuracoes gerais de soleira/numero
    rotacionar_numero_casa: bool = False


# Objeto global de parametros (sem painel UI)
GLOBAL_PARAMS = Params()

# Tolerancias globais
TOLERANCES = GeometricTolerances()
