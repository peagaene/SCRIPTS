# === collision_detector.py ===
from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict, Any
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
import numpy as np

class TextElement:
    """Representa um elemento de texto com suas dimensões e posição."""
    def __init__(self, content: str, x: float, y: float, height: float, 
                 rotation: float = 0.0, layer: str = "", style: str = ""):
        self.content = content
        self.x = x
        self.y = y
        self.height = height
        self.rotation = rotation
        self.layer = layer
        self.style = style
        self.width = self._calculate_width()
        self.bounds = self._calculate_bounds()
    
    def _calculate_width(self) -> float:
        """Calcula largura aproximada do texto baseada no conteúdo."""
        # Fator de largura baseado no estilo (Arial vs SIMPLEX)
        char_width_factor = 0.6 if "arial" in self.style.lower() else 0.7
        return len(self.content) * self.height * char_width_factor
    
    def _calculate_bounds(self) -> Tuple[float, float, float, float]:
        """Calcula bounding box do texto (x_min, y_min, x_max, y_max)."""
        half_width = self.width / 2
        half_height = self.height / 2
        
        # Aplica rotação se necessário
        if abs(self.rotation) > 0.1:
            cos_r = math.cos(math.radians(self.rotation))
            sin_r = math.sin(math.radians(self.rotation))
            
            # Pontos dos cantos do retângulo
            corners = [
                (-half_width, -half_height),
                (half_width, -half_height),
                (half_width, half_height),
                (-half_width, half_height)
            ]
            
            # Rotaciona os cantos
            rotated_corners = []
            for cx, cy in corners:
                rx = cx * cos_r - cy * sin_r + self.x
                ry = cx * sin_r + cy * cos_r + self.y
                rotated_corners.append((rx, ry))
            
            # Calcula bounding box dos pontos rotacionados
            xs = [p[0] for p in rotated_corners]
            ys = [p[1] for p in rotated_corners]
            return (min(xs), min(ys), max(xs), max(ys))
        else:
            return (self.x - half_width, self.y - half_height, 
                   self.x + half_width, self.y + half_height)
    
    def get_polygon(self) -> Polygon:
        """Retorna polígono representando o texto."""
        x_min, y_min, x_max, y_max = self.bounds
        return Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

class CollisionDetector:
    """Detector e resolvedor de colisões entre elementos de texto."""
    
    def __init__(self, min_distance: float = 1.0, buffer_factor: float = 1.2):
        self.min_distance = min_distance
        self.buffer_factor = buffer_factor
        self.text_elements: List[TextElement] = []
        self.feature_polygons: List[Polygon] = []
    
    def add_text_element(self, content: str, x: float, y: float, height: float,
                        rotation: float = 0.0, layer: str = "", style: str = ""):
        """Adiciona um elemento de texto para verificação de colisão."""
        element = TextElement(content, x, y, height, rotation, layer, style)
        self.text_elements.append(element)
        return element
    
    def add_feature_polygon(self, polygon: Polygon):
        """Adiciona um polígono de feature para verificação de colisão."""
        self.feature_polygons.append(polygon)
    
    def detect_text_collisions(self) -> List[Tuple[TextElement, TextElement]]:
        """Detecta colisões entre textos."""
        collisions = []
        n = len(self.text_elements)
        
        for i in range(n):
            for j in range(i + 1, n):
                elem1, elem2 = self.text_elements[i], self.text_elements[j]
                
                # Verifica se são da mesma camada (evita conflitos entre layers diferentes)
                if elem1.layer == elem2.layer:
                    if self._check_text_collision(elem1, elem2):
                        collisions.append((elem1, elem2))
        
        return collisions
    
    def detect_text_feature_collisions(self) -> List[Tuple[TextElement, Polygon]]:
        """Detecta colisões entre textos e features."""
        collisions = []
        
        for text_elem in self.text_elements:
            for feature in self.feature_polygons:
                if self._check_text_feature_collision(text_elem, feature):
                    collisions.append((text_elem, feature))
        
        return collisions
    
    def _check_text_collision(self, elem1: TextElement, elem2: TextElement) -> bool:
        """Verifica se dois elementos de texto colidem."""
        # Distância euclidiana entre centros
        distance = math.hypot(elem1.x - elem2.x, elem1.y - elem2.y)
        
        # Distância mínima baseada no tamanho dos textos
        min_dist = (elem1.height + elem2.height) * self.buffer_factor + self.min_distance
        
        return distance < min_dist
    
    def _check_text_feature_collision(self, text_elem: TextElement, feature: Polygon) -> bool:
        """Verifica se um texto colide com uma feature."""
        text_poly = text_elem.get_polygon()
        
        # Verifica interseção direta
        if text_poly.intersects(feature):
            return True
        
        # Verifica proximidade com buffer
        buffer_dist = text_elem.height * self.buffer_factor + self.min_distance
        if text_poly.distance(feature) < buffer_dist:
            return True
        
        return False
    
    def resolve_collisions(self) -> Dict[str, Any]:
        """Resolve colisões reposicionando elementos."""
        resolved = {
            "text_moves": [],
            "text_removals": [],
            "feature_avoidance": []
        }
        
        # 1. Resolver colisões texto-texto
        text_collisions = self.detect_text_collisions()
        for elem1, elem2 in text_collisions:
            # Estratégia: mover o elemento com menor prioridade
            priority1 = self._get_text_priority(elem1)
            priority2 = self._get_text_priority(elem2)
            
            if priority1 < priority2:
                new_pos = self._find_alternative_position(elem1, elem2)
                if new_pos:
                    resolved["text_moves"].append((elem1, new_pos))
                else:
                    resolved["text_removals"].append(elem1)
            else:
                new_pos = self._find_alternative_position(elem2, elem1)
                if new_pos:
                    resolved["text_moves"].append((elem2, new_pos))
                else:
                    resolved["text_removals"].append(elem2)
        
        # 2. Resolver colisões texto-feature
        feature_collisions = self.detect_text_feature_collisions()
        for text_elem, feature in feature_collisions:
            new_pos = self._find_alternative_position_near_feature(text_elem, feature)
            if new_pos:
                resolved["text_moves"].append((text_elem, new_pos))
            else:
                resolved["text_removals"].append(text_elem)
        
        return resolved
    
    def _get_text_priority(self, element: TextElement) -> int:
        """Retorna prioridade do elemento (menor = maior prioridade)."""
        priority_map = {
            "TOP_EDIF_NUM": 1,      # Número da casa - maior prioridade
            "TOP_EDIF_PAV": 2,      # Pavimento
            "TOP_AREA_LOTE": 3,    # Área do lote
            "TOP_SISTVIA": 4,      # Nome da via
            "TOP_COTAS_VIARIO": 5, # Cotas viárias
            "TOP_CURVA_NIV": 6,    # Curvas de nível
            "TOP_VERTICE": 7,      # Vértices
        }
        return priority_map.get(element.layer, 8)
    
    def _find_alternative_position(self, moving_elem: TextElement, 
                                 reference_elem: TextElement) -> Optional[Tuple[float, float]]:
        """Encontra posição alternativa para evitar colisão."""
        # Direções para tentar reposicionamento
        directions = [
            (1.0, 0.0),   # direita
            (-1.0, 0.0),  # esquerda
            (0.0, 1.0),   # acima
            (0.0, -1.0),  # abaixo
            (1.0, 1.0),   # diagonal superior direita
            (-1.0, 1.0),  # diagonal superior esquerda
            (1.0, -1.0),  # diagonal inferior direita
            (-1.0, -1.0), # diagonal inferior esquerda
        ]
        
        step_size = (moving_elem.height + reference_elem.height) * 1.5
        
        for dx, dy in directions:
            new_x = moving_elem.x + dx * step_size
            new_y = moving_elem.y + dy * step_size
            
            # Verifica se a nova posição não colide
            temp_elem = TextElement(moving_elem.content, new_x, new_y, 
                                  moving_elem.height, moving_elem.rotation, 
                                  moving_elem.layer, moving_elem.style)
            
            if not self._check_text_collision(temp_elem, reference_elem):
                # Verifica se não colide com outros elementos
                collision_found = False
                for other_elem in self.text_elements:
                    if other_elem != moving_elem and other_elem != reference_elem:
                        if self._check_text_collision(temp_elem, other_elem):
                            collision_found = True
                            break
                
                if not collision_found:
                    return (new_x, new_y)
        
        return None
    
    def _find_alternative_position_near_feature(self, text_elem: TextElement, 
                                              feature: Polygon) -> Optional[Tuple[float, float]]:
        """Encontra posição alternativa próxima à feature mas sem colisão."""
        # Tenta posições ao redor da feature
        buffer_dist = text_elem.height * 2.0 + self.min_distance
        
        # Pontos de teste ao redor da feature
        bounds = feature.bounds
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        
        test_positions = [
            (center_x + buffer_dist, center_y),           # direita
            (center_x - buffer_dist, center_y),           # esquerda
            (center_x, center_y + buffer_dist),           # acima
            (center_x, center_y - buffer_dist),           # abaixo
            (center_x + buffer_dist, center_y + buffer_dist),  # diagonal
            (center_x - buffer_dist, center_y + buffer_dist),
            (center_x + buffer_dist, center_y - buffer_dist),
            (center_x - buffer_dist, center_y - buffer_dist),
        ]
        
        for test_x, test_y in test_positions:
            temp_elem = TextElement(text_elem.content, test_x, test_y,
                                  text_elem.height, text_elem.rotation,
                                  text_elem.layer, text_elem.style)
            
            # Verifica se não colide com a feature
            if not self._check_text_feature_collision(temp_elem, feature):
                # Verifica se não colide com outros textos
                collision_found = False
                for other_elem in self.text_elements:
                    if other_elem != text_elem:
                        if self._check_text_collision(temp_elem, other_elem):
                            collision_found = True
                            break
                
                if not collision_found:
                    return (test_x, test_y)
        
        return None

def apply_collision_resolution(ms, collision_results: Dict[str, Any]):
    """Aplica as resoluções de colisão no ModelSpace."""
    # Remove elementos que causam conflito
    for element in collision_results["text_removals"]:
        # Nota: Em uma implementação real, seria necessário rastrear
        # os objetos DXF para removê-los
        print(f"[INFO] Removendo texto conflitante: {element.content}")
    
    # Move elementos para novas posições
    for element, new_pos in collision_results["text_moves"]:
        print(f"[INFO] Movendo texto '{element.content}' de ({element.x:.2f}, {element.y:.2f}) para ({new_pos[0]:.2f}, {new_pos[1]:.2f})")
        # Nota: Em uma implementação real, seria necessário atualizar
        # a posição dos objetos DXF
