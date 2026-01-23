import geopandas as gpd
import os

# === CONFIGURAÇÕES ===
CAMINHO_GPKG = r"D:\80225_PROJETO_IAT_PARANA\2 - Execução de voo\1_1_GPKG\ES_PV_LASER_L09_FAIXAS_R0.gpkg"
NOME_CAMADA = "ES_PV_LASER_L09_FAIXAS_R0"
CAMINHO_SAIDA_SHP = r"D:\80225_PROJETO_IAT_PARANA\2 - Execução de voo\1_1_GPKG\ES_PV_LASER_L09_FAIXAS_R0.shp"

# === CONVERSÃO ===
gdf = gpd.read_file(CAMINHO_GPKG, layer=NOME_CAMADA)
os.makedirs(os.path.dirname(CAMINHO_SAIDA_SHP), exist_ok=True)
gdf.to_file(CAMINHO_SAIDA_SHP, driver="ESRI Shapefile")

print("Conversão concluída:", CAMINHO_SAIDA_SHP)
