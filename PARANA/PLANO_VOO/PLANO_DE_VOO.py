import os
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import LineString, Point
import numpy as np
from pyproj import Transformer

# === CONFIGURAÇÕES ===
DIR_BLOCOS = r"D:\80225_PROJETO_IAT_PARANA\2 Planejamento voo\02 - VOO LASER\APOIO\LOTE_10\SHP_LINHA_FAIXA"
DIR_MDT = r"D:\80225_PROJETO_IAT_PARANA\2 Planejamento voo\02 - VOO LASER\APOIO\LOTE_10\MDT"
DIRETORIO_SAIDA = r"D:\80225_PROJETO_IAT_PARANA\2 Planejamento voo\02 - VOO LASER\LOTE_10\1_3_XLSX"
ARQUIVO_SAIDA_GPKG = os.path.join(DIRETORIO_SAIDA, "ES_PV_LASER_L09_FAIXAS_R0.gpkg")

ALTURA_VOO = 1400  # metros fixos
VELOCIDADE_VOO_MS = 120 * 0.514444  # knots para m/s

# === FUNÇÕES AUXILIARES ===
def extrair_perfil_agl(geometry: LineString, mdt_path: str):
    with rasterio.open(mdt_path) as src:
        num_amostras = 100
        distances = np.linspace(0, geometry.length, num_amostras)
        coords = [(geometry.interpolate(d).x, geometry.interpolate(d).y) for d in distances]

        # reprojetar os pontos para o mesmo CRS do MDT, se forem diferentes
        if src.crs and src.crs.to_string() != "EPSG:4674":
            transformer = Transformer.from_crs("EPSG:4674", src.crs, always_xy=True)
            coords = [transformer.transform(x, y) for x, y in coords]

        elevs_raw = list(src.sample(coords))
        nodata = src.nodata
        elevs = np.array([e[0] for e in elevs_raw if e[0] is not None and not np.isnan(e[0]) and (nodata is None or e[0] != nodata)])
        if elevs.size == 0:
            print("[Aviso] Nenhuma elevação válida encontrada para linha.")
            return None, None, None

        terreno_inicial = elevs[0]
        altura_voo_fixa = terreno_inicial + ALTURA_VOO

        agl = altura_voo_fixa - elevs
        alt_msl = altura_voo_fixa
        return alt_msl, agl.min(), agl.max()

def formatar_coordenada(coord_xy, crs_src):
    transformer = Transformer.from_crs(crs_src, "EPSG:4674", always_xy=True)
    lon, lat = transformer.transform(coord_xy[0], coord_xy[1])

    def dec_to_dms(dec, is_lat):
        graus = int(abs(dec))
        minutos_dec = (abs(dec) - graus) * 60
        minutos = int(minutos_dec)
        segundos = (minutos_dec - minutos) * 60
        direcao = (
            'N' if is_lat and dec >= 0 else
            'S' if is_lat else
            'E' if not is_lat and dec >= 0 else 'W'
        )
        return f"{graus:02d}º{minutos:02d}'{segundos:04.1f}\"{direcao}"

    return f"{dec_to_dms(lat, True)} {dec_to_dms(lon, False)}"

# === PROCESSAMENTO ===
linhas = []
codigo_faixa = 1

# Garantir ordem alfabética dos blocos A-J
arquivos_blocos = sorted([f for f in os.listdir(DIR_BLOCOS) if f.startswith("BLOCO_") and f.endswith(".shp")], key=lambda x: x.replace("BLOCO_", "").replace(".shp", ""))

for arquivo in arquivos_blocos:
    bloco_id = arquivo.replace("BLOCO_", "").replace(".shp", "").upper()
    caminho_shp = os.path.join(DIR_BLOCOS, arquivo)
    caminho_mdt = os.path.join(DIR_MDT, f"BLOCO_{bloco_id}.tif")

    if not os.path.exists(caminho_mdt):
        print(f"MDT do bloco {bloco_id} não encontrado.")
        continue

    gdf = gpd.read_file(caminho_shp)
    gdf = gdf[gdf.geometry.type == "LineString"]

    # manter em EPSG:4674 conforme instrução
    gdf["centroid_y"] = gdf.geometry.centroid.y
    gdf = gdf.sort_values(by="centroid_y", ascending=False)

    for i, row in gdf.iterrows():
        linha = row.geometry
        flightline = f"04L10{bloco_id}{codigo_faixa:03d}"
        linha_proj = gpd.GeoSeries([linha], crs=gdf.crs).to_crs(epsg=31978).iloc[0]
        ext_m = linha_proj.length
        tempo_seg = ext_m / VELOCIDADE_VOO_MS  # aproximação se em graus

        alt_msl, agl_min, agl_max = extrair_perfil_agl(linha, caminho_mdt)

        first_event = formatar_coordenada(linha.coords[0], gdf.crs)
        last_event = formatar_coordenada(linha.coords[-1], gdf.crs)

        linhas.append({
            "FlightLine": flightline,
            "BLOCO": bloco_id,
            "geometry": linha,
            "Length(m)": round(ext_m, 5),
            "AltMSL(m)": round(alt_msl, 2) if alt_msl else None,
            "MinAltAGL(m)": round(agl_min, 2) if agl_min else None,
            "MaxAltAGL(m)": round(agl_max, 2) if agl_max else None,
            "FirstEvent": first_event,
            "LastEventW": last_event,
            "Estimateddf": round(tempo_seg, 1)
        })

        print(f"Processado: {flightline}")
        codigo_faixa += 1

# === EXPORTAR PARA GPKG ===
gdf_linhas = gpd.GeoDataFrame(linhas, crs=gdf.crs)
gdf_linhas.to_file(ARQUIVO_SAIDA_GPKG, driver="GPKG")
print("Arquivo salvo em:", ARQUIVO_SAIDA_GPKG)
