import os
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# === CAMINHOS ===
GPKG_LINHAS = r"D:\80225_PROJETO_IAT_PARANA\2 Planejamento voo\02 - VOO LASER\LOTE_10\1_1_GPKG\ES_PV_LASER_L10_FAIXAS_R0.gpkg"
SHP_BLOCOS = r"D:\80225_PROJETO_IAT_PARANA\2 Planejamento voo\02 - VOO LASER\APOIO\LOTE_10\\BLOCOS.shp"
SHP_MUNICIPIO = r"D:\80225_PROJETO_IAT_PARANA\2 Planejamento voo\APOIO\Limites Municipais\\Limite_PR.shp"
PDF_SAIDA = r"D:\80225_PROJETO_IAT_PARANA\2 Planejamento voo\02 - VOO LASER\LOTE_10\1_2_PDF\\ES_PV_LASER_L10_R0.pdf"

# === LEITURA DOS DADOS ===
gdf_linhas = gpd.read_file(GPKG_LINHAS)
gdf_blocos = gpd.read_file(SHP_BLOCOS)
gdf_municipio = gpd.read_file(SHP_MUNICIPIO)

# Reprojetar todos para o CRS comum do GPKG
crs_base = gdf_linhas.crs
gdf_blocos = gdf_blocos.to_crs(crs_base)
gdf_municipio = gdf_municipio.to_crs(crs_base)

# === GERAÇÃO DO PDF COM MATPLOTLIB ===
fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape

# Zoom apenas com base nos blocos
x0, y0, x1, y1 = gdf_blocos.total_bounds
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)

# Limite municipal
gdf_municipio.boundary.plot(ax=ax, color='black', linewidth=1, label='Limite Municipal')

# Blocos vazados com contorno vermelho
gdf_blocos.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=1, label='Blocos')

# Linhas de voo
gdf_linhas.plot(ax=ax, color='black', linewidth=0.2)

# Rótulo das linhas (FlightLine)
used_positions = []

for _, row in gdf_linhas.iterrows():
    x, y = row.geometry.interpolate(0.01, normalized=True).xy
    bloco = row['BLOCO'] if 'BLOCO' in row else ''

    bloco_geom = gdf_blocos[gdf_blocos['BLOCOS'] == bloco].geometry
    if not bloco_geom.empty:
        largura = bloco_geom.iloc[0].bounds[2] - bloco_geom.iloc[0].bounds[0]
        dx = largura * 0.01 if bloco in ['A', 'C', 'E', 'G', 'I', 'M'] else -largura * 0.01
        ha = 'left' if dx > 0 else 'right'
    else:
        dx = 0
        ha = 'center'

    x_pos = x[0] + dx
    y_pos = y[0]

    for used_x, used_y in used_positions:
        dist = ((x_pos - used_x)**2 + (y_pos - used_y)**2)**0.5
        if dist < 0.001:
            y_pos += 0.001

    used_positions.append((x_pos, y_pos))

    ax.text(
        x_pos, y_pos, row['FlightLine'], fontsize=1.5, ha=ha, va='center',
        bbox=dict(facecolor='white', edgecolor='none', pad=0.3)
    )

# Nome dos blocos dentro dos polígonos
coluna_nome_bloco = 'BLOCOS'
if coluna_nome_bloco:
    for _, row in gdf_blocos.iterrows():
        centroide = row.geometry.centroid
        cx, cy = centroide.x, centroide.y
        ax.text(
            cx, cy, row[coluna_nome_bloco], fontsize=10, ha='center', va='center', color='black'
        )

# Estilo
ax.axis('off')
ax.set_aspect('equal')

# Salvar PDF
os.makedirs(os.path.dirname(PDF_SAIDA), exist_ok=True)
with PdfPages(PDF_SAIDA) as pdf:
    pdf.savefig(fig, bbox_inches='tight')

plt.close()
print("PDF salvo em:", PDF_SAIDA)

