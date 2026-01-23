# === area_table.py ===
from __future__ import annotations
import math
from typing import Optional, Dict, Any
from shapely.geometry import Polygon
import ezdxf

def format_area_br_m2(area_m2: float, casas: int = 3) -> str:
    """Formata área com milhar '.' e decimal ',' seguido de m²."""
    try:
        s = f"{float(area_m2):,.{casas}f}"
    except Exception:
        return f"{area_m2} m²"
    s = s.replace(",", "|").replace(".", ",").replace("|", ".")
    return f"{s} m²"

def create_area_table(ms, 
                     per_interesse: Optional[Polygon] = None,
                     per_levantamento: Optional[Polygon] = None,
                     params: Any = None) -> bool:
    """
    Cria tabela com áreas de levantamento e núcleo (PER_INTERESSE).
    
    Args:
        ms: ModelSpace do DXF
        per_interesse: Polígono do perímetro de interesse (núcleo)
        per_levantamento: Polígono do perímetro de levantamento
        params: Parâmetros de configuração
        
    Returns:
        bool: True se a tabela foi criada com sucesso
    """
    try:
        # Configurações
        layer_tabela = getattr(params, "layer_tabela", "TOP_TABELA")
        altura_texto = getattr(params, "altura_texto_tabela", 2.0)
        style_texto = getattr(params, "style_texto", "SIMPLEX")
        
        # Posição da tabela
        anchor = getattr(params, "area_table_anchor", None)
        offset_x = getattr(params, "tabela_offset_x", 120.0)
        offset_y = getattr(params, "tabela_offset_y", 0.0)
        cell_width = getattr(params, "tabela_cell_w", 25.0)
        cell_height = getattr(params, "tabela_cell_h", 6.0)
        
        # Calcular áreas
        area_nucleo = None
        area_levantamento = None
        
        if per_interesse and not per_interesse.is_empty:
            try:
                area_nucleo = float(per_interesse.area)
            except Exception:
                pass
                
        if per_levantamento and not per_levantamento.is_empty:
            try:
                area_levantamento = float(per_levantamento.area)
            except Exception:
                pass
        
        # Se não há áreas válidas, não criar tabela
        if area_nucleo is None and area_levantamento is None:
            return False
        
        # Posição inicial da tabela
        if isinstance(anchor, tuple) and len(anchor) == 2:
            start_x, start_y = float(anchor[0]), float(anchor[1])
        else:
            start_x = offset_x
            start_y = offset_y
        
        # Desenho no formato duas células lado a lado, conforme exemplo
        total_w = cell_width * 2.0
        total_h = cell_height

        # Retângulo externo
        try:
            ms.add_lwpolyline([(start_x, start_y), (start_x + total_w, start_y),
                               (start_x + total_w, start_y - total_h), (start_x, start_y - total_h),
                               (start_x, start_y)], close=True, dxfattribs={"layer": layer_tabela})
            # Linha divisória central
            ms.add_line((start_x + total_w / 2.0, start_y), (start_x + total_w / 2.0, start_y - total_h),
                        dxfattribs={"layer": layer_tabela})
        except Exception:
            pass

        # Textos centralizados em cada célula
        def add_center(titulo: str, valor: str, cx: float):
            conteudo = f"{titulo}\\P{valor}"
            mt = ms.add_mtext(conteudo, dxfattribs={"style": style_texto, "layer": layer_tabela})
            try:
                mt.dxf.char_height = altura_texto
                mt.dxf.attachment_point = 5
                mt.dxf.width = max(4.0, cell_width - 2.0)
            except Exception:
                pass
            try:
                mt.set_location((cx, start_y - total_h / 2.0, 0.0))
            except Exception:
                mt.dxf.insert = (cx, start_y - total_h / 2.0)

        left_data = None
        right_data = None
        if area_nucleo is not None:
            left_data = ("ÁREA TOTAL DO NÚCLEO", format_area_br_m2(area_nucleo))
        if area_levantamento is not None:
            right_data = ("ÁREA TOTAL DE LEVANTAMENTO", format_area_br_m2(area_levantamento))

        # fallback se veio apenas um lado
        if left_data and right_data:
            add_center(left_data[0], left_data[1], start_x + total_w * 0.25)
            add_center(right_data[0], right_data[1], start_x + total_w * 0.75)
        elif left_data:
            add_center(left_data[0], left_data[1], start_x + total_w * 0.5)
        elif right_data:
            add_center(right_data[0], right_data[1], start_x + total_w * 0.5)

        return True
        
    except Exception as e:
        print(f"[WARN] Falha ao criar tabela de áreas: {e}")
        return False
