import geopandas as gpd
import pandas as pd
from tqdm import tqdm  # Barra de progresso

# Função para calcular a distância entre dois pontos
def calcular_distancia(geom1, geom2):
    return geom1.distance(geom2)

# Carregar os dois shapefiles
shp1 = gpd.read_file(r"D:\SI_16\EO.shp")
shp2 = gpd.read_file(r"D:\SI_16\EO2.shp")

# Campo da tabela de atributos para comparar (diferente em cada shapefile)
campo_comparacao_shp1 = "Imagem"
campo_comparacao_shp2 = "ID"

# Usar merge para comparar as duas tabelas de atributos
df1 = shp1[[campo_comparacao_shp1, 'geometry']]
df2 = shp2[[campo_comparacao_shp2, 'geometry']]

# Realizar o merge entre os dois DataFrames baseado nos valores dos atributos
merged = pd.merge(df1, df2, left_on=campo_comparacao_shp1, right_on=campo_comparacao_shp2)

# Lista para armazenar os resultados
resultados = []

# Loop para calcular as distâncias apenas nos pontos que têm valores comuns
for idx, row in tqdm(merged.iterrows(), total=len(merged), desc="Calculando distâncias", unit="linha"):
    distancia = calcular_distancia(row['geometry_x'], row['geometry_y'])
    
    # Formatar a distância com 2 casas decimais e substituir ponto por vírgula
    distancia_formatada = f"{distancia:.2f}".replace('.', ',')
    
    # Adicionar o resultado à lista
    resultados.append({
        "valor_comum": row[campo_comparacao_shp1],
        "distancia": distancia_formatada  # Distância com vírgula
    })

# Criar um DataFrame com os resultados
df_resultados = pd.DataFrame(resultados)

# Salvar o resultado em um arquivo CSV com separador ponto e vírgula e distância com vírgula
df_resultados.to_csv(r"D:\SI_16\resultados_comparacao.csv", index=False, sep=";", float_format="%.2f")

# Exibir os resultados
print(df_resultados)
