import os
import pandas as pd
import logging
from dbfread import DBF
import geopandas as gpd
from shapely.geometry import LineString, Point
from pyproj import CRS
from shapely.ops import nearest_points

DIRETORIO = r"D:\Pedro\PROJETOS\GOV_SP\SI_08"
#Obter lista de arquivos gerados
arquivos_foto = os.listdir(DIRETORIO)
    
# Lista para armazenar todos os DataFrames processados
df_list = []

# Loop através dos arquivos gerados
for arquivo_fotos in arquivos_foto:
    if arquivo_fotos.startswith("Preliminar") and arquivo_fotos.endswith(".xlsx"):
        caminho_arquivo_foto = os.path.join(DIRETORIO, arquivo_fotos)
        
        # Ler o arquivo gerado
        df_foto = pd.read_excel(caminho_arquivo_foto) 

        df_list.append(df_foto)
                    
df_combined = pd.concat(df_list, ignore_index=True)
# Remover linhas com células vazias na planilha combinada
df_combined = df_combined.dropna()

# Agrupar os dados pelo valor de "Faixa"
gdf = gpd.GeoDataFrame(df_combined, geometry=None)

# Criar um GeoDataFrame com a coluna "Split_Faixa"
gdf["Split_Faixa"] = [int(faixa.split('_')[1]) if isinstance(faixa, str) else None for faixa in gdf["Faixa"]]

# Criar geometrias de linha para cada faixa
line_geometries = []

grouped = gdf.groupby("Faixa")
for faixa, group in grouped:
    points = group.apply(lambda row: (row["E"], row["N"]), axis=1)
    line = LineString(points)
    line_geometries.append(line)


# Criar um GeoDataFrame com as geometrias de linha
line_gdf = gpd.GeoDataFrame(
    {
        "Faixa": [faixa for faixa, _ in grouped],  # Usar a faixa do grupo de pontos
        "Split_Faixa": [int(faixa.split('_')[1]) for faixa, _ in grouped],
        "geometry": line_geometries
    },
    crs=CRS.from_epsg(31983)  # Definir a projeção UTM SIRGAS 2000 fuso 23S
)

# Criar geometrias de ponto para cada ponto
point_geometries = []

for _, row in gdf.iterrows():
    point = Point(row["E"], row["N"])
    point_geometries.append(point)

# Criar um GeoDataFrame com as geometrias de ponto
point_gdf = gpd.GeoDataFrame(
    gdf[["Imagem analisada", "Faixa", "E", "N", "Split_Faixa"]],  # Manter apenas as colunas necessárias
    geometry=point_geometries,
    crs=CRS.from_epsg(31983)  # Definir a projeção UTM SIRGAS 2000 fuso 23S
)

# Loop para calcular a menor distância entre pontos e linhas
min_distances = []

for _, point_row in point_gdf.iterrows():
    if pd.notna(point_row['Split_Faixa']):
        faixa_atual = point_row['Split_Faixa']
        target_faixa = point_row['Split_Faixa'] + 1

        # Filtrar a linha de destino
        target_lines = line_gdf[line_gdf['Split_Faixa'] == target_faixa]['geometry']

        if not target_lines.empty:
            target_line = target_lines.iloc[0]

            # Encontrar os pontos mais próximos entre o ponto e a linha
            nearest_point_on_line, nearest_point_on_point = nearest_points(target_line, point_row['geometry'])

            # Calcular a distância entre os pontos mais próximos
            distance = nearest_point_on_line.distance(nearest_point_on_point)

            min_distances.append(distance)

        else:
            # Não há linha correspondente para target_faixa, então tentar com target_faixa - 1
            target_lines_prev = line_gdf[line_gdf['Split_Faixa'] == (target_faixa - 2)]['geometry']
            if not target_lines_prev.empty:
                target_line_prev = target_lines_prev.iloc[0]

                # Encontrar os pontos mais próximos entre o ponto e a linha anterior
                nearest_point_on_line_prev, nearest_point_on_point_prev = nearest_points(target_line_prev, point_row['geometry'])
                distance = nearest_point_on_line_prev.distance(nearest_point_on_point_prev)
                min_distances.append(distance)
            else:
                # Não há linha correspondente nem para target_faixa nem para target_faixa - 1
                min_distances.append(None)

# Adicionar a coluna de distâncias mínimas ao GeoDataFrame de pontos
point_gdf['Min_Dist_to_Next_Line'] = min_distances
filtered_gdf = point_gdf[(point_gdf['Min_Dist_to_Next_Line'] >= 200) & (point_gdf['Min_Dist_to_Next_Line'] < 450)]

lado = 1044.96

# Calcular sobreposição lateral correta
maior_distancia = filtered_gdf.groupby("Faixa")["Min_Dist_to_Next_Line"].max()
menor_sobreposicao = ((lado - maior_distancia) / lado) * 100

menor_distancia = filtered_gdf.groupby("Faixa")["Min_Dist_to_Next_Line"].min()
maior_sobreposicao = ((lado - menor_distancia) / lado) * 100

media_distancia = filtered_gdf.groupby("Faixa")["Min_Dist_to_Next_Line"].mean()
media_sobreposicao = ((lado - media_distancia) / lado) * 100

# Calcular sobreposição longitudinal correta
maior_longitudinal = df_combined.groupby("Faixa")["Sobreposição"].max()

menor_longitudinal = df_combined.groupby("Faixa")["Sobreposição"].min()

media_longitudinal = df_combined.groupby("Faixa")["Sobreposição"].mean()

# Criar um DataFrame com as informações de distâncias
df_distancias = pd.DataFrame({
    "Maior Sobreposição(%) Lateral": maior_sobreposicao,
    "Menor Sobreposição(%) Lateral": menor_sobreposicao,
    "Media Sobreposição(%) Lateral": media_sobreposicao,
    "Maior Sobreposição(%) Longitudinal": maior_longitudinal,
    "Menor Sobreposição(%) Longitudinal": menor_longitudinal,
    "Media Sobreposição(%) Longitudinal": media_longitudinal,
}).reset_index()

caminho_distancia = os.path.join(DIRETORIO, "sobreposicao_lateral.xlsx")
df_distancias.to_excel(caminho_distancia, index=False)

# Exportar o GeoDataFrame de pontos para shapefile
shapefile_pontos = os.path.join(DIRETORIO, "pontos.shp")
point_gdf.to_file(shapefile_pontos)

# Exportar o GeoDataFrame de linhas para shapefile
shapefile_linhas = os.path.join(DIRETORIO, "linhas.shp")
line_gdf.to_file(shapefile_linhas)