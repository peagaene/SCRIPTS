"""
Config: mappings type->layer and layer->block.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

# ATENCAO: o leitor uppercasa o 'type', entao use SOMENTE chaves em MAIUSCULAS.
TYPE_TO_LAYER = {
    # iluminacao / energia
    "PA": "ELET_POSTE_ALTA_TENSAO_LUMINARIA",
    "PI": "ELET_POSTE_ALTA_TENSAO_LUMINARIA",
    "PFI": "ELET_POSTE_ALTA_TENSAO",
    # TUBO DE TELEFONIA / TV
    "PVTEL": "INFRA_PVT", "PVT": "INFRA_PVT", "AEPVTEL": "INFRA_PVT",
    # PV de agua pluvial
    "PVA": "INFRA_PVAP", "PVAP": "INFRA_PVAP", "AEPVA": "INFRA_PVAP",
    # CORRECAO: AEPVE deve ir para INFRA_PVE (antes estava PVAP)
    "AEPVE": "INFRA_PVE", "ES": "INFRA_PVE", "PVE": "INFRA_PVE",
    "AEPVPD": "INFRA_PVE", "PV": "INFRA_PVE",
    # Boca de lobo (variacoes)
    "AEBO": "INFRA_BOCA_LOBO", "AEBO1": "INFRA_BOCA_LOBO", "AEBO2": "INFRA_BOCA_LOBO", "AEBO3": "INFRA_BOCA_LOBO",
    "AEBO.1": "INFRA_BOCA_LOBO", "AEBO.2": "INFRA_BOCA_LOBO",
    "BL1": "INFRA_BOCA_LOBO", "BL2": "INFRA_BOCA_LOBO", "BL3": "INFRA_BOCA_LOBO",
    # Boca de leao (variacoes)
    "AEBE": "INFRA_BOCA_LEAO", "AEBE1": "INFRA_BOCA_LEAO", "AEBE2": "INFRA_BOCA_LEAO", "AEBE3": "INFRA_BOCA_LEAO",
    "AEBE.1": "INFRA_BOCA_LEAO", "AEBE.2": "INFRA_BOCA_LEAO", "AEBE.3": "INFRA_BOCA_LEAO", "AEB": "INFRA_BOCA_LEAO",
    # AEBL generico
    "AEBL": "INFRA_BOCA_LOBO", "AEBL1": "INFRA_BOCA_LOBO", "AEBL2": "INFRA_BOCA_LOBO", "AEBL3": "INFRA_BOCA_LOBO",
    # Arvores (normalizacao no wrapper para 'ARVORE')
    "ARVORE": "VEG_ARVORE_ISOLADA",
}

LAYER_TO_BLOCK = {
    "INFRA_PVE": "INFRA_PVE",
    "INFRA_PVAP": "INFRA_PVAP",
    "INFRA_PVT": "INFRA_PVT",
    "INFRA_BOCA_LOBO": "BOCA_LOBO",
    "INFRA_BOCA_LEAO": "BOCA_DE_LEAO",
    "ELET_POSTE_ALTA_TENSAO_LUMINARIA": "POSTE_ILUMI",
    "ELET_POSTE_ALTA_TENSAO": "POSTE_TENSAO",
    "MOB_MOBILIARIO_URBANO": "MOB_URBANO_PT_ONIBUS",
    "INFRA_BUEIRO": "COD119",
    "VEG_ARVORE_ISOLADA": "ARVORE",
}

IGNORAR_SEM_LOG = {"DIVL", "VIELA"}
