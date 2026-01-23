import geopandas as gpd

# Caminho do arquivo de entrada
shp_entrada = r"D:\80225_PROJETO_IAT_PARANA\1 - Planejamento voo\APOIO\Divisao de area\Lotes_PR.shp"

# Caminho para salvar o shapefile convertido
shp_saida = r"D:\80225_PROJETO_IAT_PARANA\1 - Planejamento voo\APOIO\Divisao de area\Lotes_PR_utm.shp"

# Lê o shapefile original (esperado em EPSG:4674 - SIRGAS 2000 geográfico)
gdf = gpd.read_file(shp_entrada)

# Verifica se o CRS atual é realmente EPSG:4674
print("CRS original:", gdf.crs)

# Reprojeta para UTM zona 22S (EPSG:31982)
gdf_utm = gdf.to_crs(epsg=31982)

# Salva o novo shapefile com CRS transformado
gdf_utm.to_file(shp_saida)

print("Shapefile convertido com sucesso para UTM fuso 22S (EPSG:31982).")
