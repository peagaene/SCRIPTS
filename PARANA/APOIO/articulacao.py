import geopandas as gpd
import numpy as np

# Carregar shapefile
gdf = gpd.read_file(r"D:\2212_GOV_SAO_PAULO\ADITIVO\AREA\ARTICULACAO_DEMO.shp")

# Calcular centróides
gdf["centroid"] = gdf.geometry.centroid
gdf["x"] = gdf.centroid.x
gdf["y"] = gdf.centroid.y

# === AGRUPAMENTO POR COLUNAS (X) ===
# Ordenar colunas da esquerda para direita
valores_x = gdf["x"].round(-3)  # agrupar a cada 1000m
colunas_unicas = np.sort(np.unique(valores_x))
gdf["coluna"] = gdf["x"].apply(lambda x: np.where(colunas_unicas == round(x, -3))[0][0])

# === AGRUPAMENTO POR LINHAS (Y) ===
valores_y = gdf["y"].round(-3)
linhas_unicas = np.sort(np.unique(valores_y))[::-1]  # de cima para baixo
gdf["linha"] = gdf["y"].apply(lambda y: np.where(linhas_unicas == round(y, -3))[0][0])

# === GERAR ID FINAL ===
gdf["ID_RENOMEADO"] = 100000 + gdf["linha"] * 1000 + gdf["coluna"]

# Remover geometria auxiliar
gdf = gdf.drop(columns="centroid")

# Salvar resultado
gdf.to_file(r"D:\2212_GOV_SAO_PAULO\ADITIVO\AREA\ARTICULACAO_RENOMEADA.shp")
