import os
import pandas as pd
from dbfread import DBF
import geopandas as gpd
from shapely.geometry import LineString, Point
from pyproj import CRS
from shapely.ops import nearest_points
import numpy as np

DIRETORIO = r"D:\Pedro\PROJETOS\GOV_SP\SI_15"
#Obter lista de arquivos gerados
arquivos_foto = os.listdir(DIRETORIO)
    
# Lista para armazenar todos os DataFrames processados
df_list = []

# Lista para armazenar as informações desejadas
informacoes = []

# Loop através dos arquivos gerados
for arquivo_fotos in arquivos_foto:
    if arquivo_fotos.startswith("Preliminar") and arquivo_fotos.endswith(".xlsx"):

        basename = arquivo_fotos[-13:-5]
        caminho_arquivo_foto = os.path.join(DIRETORIO, arquivo_fotos)
        
        # Ler o arquivo gerado
        df_foto = pd.read_excel(caminho_arquivo_foto)

        # Abaixo estão suas operações com o DataFrame df_foto, como dropna, seleção de colunas etc.
        df_foto = df_foto.dropna()
        
        # Crie uma nova coluna "Ângulo Solar" com valores vazios (NaN)
        df_foto['Angulo Solar'] = None

        columns_interest = ["Imagem analisada", "Tempo GPS", "E", "N", "Z", "Omega", "Phi", "Kappa", "Latitude", "Longitude", "Sobreposição", "Hora", "Angulo Solar", "Deriva", "Faixa"]

        df_foto = df_foto[columns_interest]

        # Calcular a média absoluta de Omega, Phi e Kappa
        media_absoluta_omega = df_foto["Omega"].abs().mean()
        media_absoluta_phi = df_foto["Phi"].abs().mean()
        media_absoluta_kappa = df_foto["Deriva"].abs().mean()

        # Adicione o DataFrame à lista
        df_list.append(df_foto)

        file_path = os.path.join(DIRETORIO, basename + '.txt')

        with open(file_path, "r") as file:
            lines = file.readlines()
            header = lines[0].strip().split('\t')
            tempo_exp_index = header.index("Shutter Speed 1/#")

             # Inicialize as variáveis para armazenar os valores máximos e mínimos
            tempo_exp_maximo = None
            tempo_exp_minimo = None

            # Lista para armazenar os resultados
            results = [] 
            # Processa cada linha do arquivo (exceto o cabeçalho)
            for line in lines[1:]:
                values = line.strip().split('\t')
                if len(values) > tempo_exp_index:
                    tempo_exp = values[tempo_exp_index]

                    if tempo_exp_maximo is None or tempo_exp > tempo_exp_maximo:
                        tempo_exp_maximo = tempo_exp

                    if tempo_exp_minimo is None or tempo_exp < tempo_exp_minimo:
                        tempo_exp_minimo = tempo_exp
        
        dbf_file_path = os.path.join(DIRETORIO, basename + '.dbf')

        dbf_data = DBF(dbf_file_path)

        # Inicialize variáveis para armazenar os valores
        altitude_maxima_realizada = None
        altitude_minima_realizada = None
        maior_diferenca = None
        valores_maior_diferenca = None

        for record in dbf_data:
            altitude_de_voo_planejada = record['Planned_Al']
            altitude_de_voo_realizada = record['Altitude_[']

            # Verifique se há informações de altitude planejada e realizada
            if altitude_de_voo_planejada is not None and altitude_de_voo_realizada is not None:
                # Calcule a diferença entre a altitude planejada e realizada
                diferenca = abs(altitude_de_voo_planejada - altitude_de_voo_realizada)

                # Verifique se a diferença é maior do que a maior diferença atual 
                if maior_diferenca is None or diferenca > maior_diferenca:  
                    maior_diferenca = diferenca
                    valores_maior_diferenca = altitude_de_voo_realizada    

            # Atualize as altitudes máximas e mínimas realizadas  
            if altitude_de_voo_realizada is not None:       
                if altitude_maxima_realizada is None or altitude_de_voo_realizada > altitude_maxima_realizada:
                    altitude_maxima_realizada = altitude_de_voo_realizada
                if altitude_minima_realizada is None or altitude_de_voo_realizada < altitude_minima_realizada:
                    altitude_minima_realizada = altitude_de_voo_realizada

        # Calcular a altura de voo
        if maior_diferenca is not None and valores_maior_diferenca is not None:
            altura_de_voo = (maior_diferenca / valores_maior_diferenca) * 100
        else:
            altura_de_voo = None  # ou outro valor padrão caso necessário

        # Adicionar informações à lista
        informacoes.append({
            'Voo': basename,
            'Altitude Minima': altitude_minima_realizada,
            'Altitude Maxima': altitude_maxima_realizada,
            'Altura voo': altura_de_voo,
            'Media Omega': media_absoluta_omega,
            'Media Phi': media_absoluta_phi,
            'Media Deriva': media_absoluta_kappa,
            'Tempo de exposição Minimo': tempo_exp_minimo,
            'Tempo de exposição Maximo': tempo_exp_maximo,
        })

        # Criar um DataFrame a partir da lista de informações
        df_informacoes = pd.DataFrame(informacoes)
        df_informacoes = df_informacoes.loc[:, ['Voo', 'Altitude Minima', 'Altitude Maxima', 'Altura voo', 'Media Omega', 'Media Phi', 'Media Deriva', 'Tempo de exposição Minimo', 'Tempo de exposição Maximo']]

        # Abre o arquivo ExcelWriter para salvar em várias planilhas
        output_file_name_combined = os.path.join(DIRETORIO, "Combinado_" + basename + ".xlsx")
        with pd.ExcelWriter(output_file_name_combined, engine='xlsxwriter') as writer:

            # Salve o DataFrame df_informacoes em uma planilha separada chamada "Informacoes"
            df_informacoes.to_excel(writer, sheet_name='Infos voo', index=False)

            # Salve o DataFrame df_foto em uma planilha chamada "Fotos"
            df_foto.to_excel(writer, sheet_name='Fotos', index=False)
          
df_combined = pd.concat(df_list, ignore_index=True)

# Agrupar os dados pelo valor de "Faixa"
gdf = gpd.GeoDataFrame(df_combined, geometry=None)

# Criar um GeoDataFrame com a coluna "Split_Faixa"
gdf["Split_Faixa"] = [int(faixa.split('_')[1]) if isinstance(faixa, str) else None for faixa in gdf["Faixa"]]
gdf["Voo"] = [int(faixa.split('_')[0]) if isinstance(faixa, str) else None for faixa in gdf["Faixa"]]

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
maior_longitudinal =gdf.groupby("Faixa")["Sobreposição"].max()

menor_longitudinal = gdf.groupby("Faixa")["Sobreposição"].min()

media_longitudinal = gdf.groupby("Faixa")["Sobreposição"].mean()

# Calcule o valor máximo em módulo para "Omega," "Phi" e "Deriva" em cada grupo
maior_omega = gdf.groupby("Faixa", group_keys=False)["Omega"].apply(lambda x: x[x.abs().idxmax()])
maior_phi = gdf.groupby("Faixa", group_keys=False)["Phi"].apply(lambda x: x[x.abs().idxmax()])
maior_kappa = gdf.groupby("Faixa", group_keys=False)["Deriva"].apply(lambda x: x[x.abs().idxmax()])

data = {
    "Faixa": maior_omega.index,
    "Media Sobreposição(%) Lateral": media_sobreposicao,
    "Menor Sobreposição(%) Lateral": menor_sobreposicao,
    "Maior Sobreposição(%) Lateral": maior_sobreposicao,
    "Maior Omega": maior_omega,
    "Maior Phi": maior_phi,
    "Maior Deriva": maior_kappa,
}

df_distancias = pd.DataFrame(data)

caminho_distancia = os.path.join(DIRETORIO, "sobreposicao_lateral.xlsx")
df_distancias.to_excel(caminho_distancia, index=False)
