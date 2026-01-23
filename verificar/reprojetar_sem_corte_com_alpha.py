# === PROCESSAMENTO COMPLETO COM INTERFACE (OTIMIZADO) ===
# ➤ Reprojetar EPSG:31983
# ➤ Cortar com shapefile (apenas geometrias que intersectam)
# ➤ Remover banda alpha mantendo transparência
# ➤ Converter para .JP2
# ➤ Salvar em IMAGENS_TRANSFORMADAS
# ➤ Log completo com progresso detalhado

import os
import subprocess
import threading
from osgeo import gdal, osr
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping
import geopandas as gpd
from datetime import datetime
import PySimpleGUI as sg

# === CONFIG GDAL ===
gdal.UseExceptions()
os.environ["GDAL_DRIVER_PATH"] = r"C:\\Users\\compartilhar\\anaconda3\\envs\\geo_env\\Library\\lib\\gdalplugins"
os.environ["GDAL_DATA"] = r"C:\\Users\\compartilhar\\anaconda3\\envs\\geo_env\\Library\\share\\gdal"

# === CONSTANTES ===
TARGET_EPSG = "EPSG:31983"
NUM_THREADS = "ALL_CPUS"
GDALWARP = r"C:\\Users\\compartilhar\\anaconda3\\envs\\geo_env\\Library\\bin\\gdalwarp.exe"
GDAL_TRANSLATE = r"C:\\Users\\compartilhar\\anaconda3\\envs\\geo_env\\Library\\bin\\gdal_translate.exe"
GDALEDIT = r"C:\\Users\\compartilhar\\anaconda3\\envs\\geo_env\\Scripts\\gdal_edit.py"
LOG_PATH = os.path.join(os.path.expanduser("~"), "log_processamento_completo.txt")

# === FUNÇÕES ===
def get_epsg_code(img_path):
    try:
        ds = gdal.Open(img_path)
        proj = ds.GetProjection()
        if not proj:
            return None
        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj)
        return srs.GetAuthorityCode("PROJCS") if srs.IsProjected() else srs.GetAuthorityCode("GEOGCS")
    except:
        return None

def remover_banda_alpha(img_entrada, img_saida):
    try:
        with gdal.Open(img_entrada) as ds:
            if ds.RasterCount == 4:
                cmd = [GDAL_TRANSLATE, "--config", "CHECK_DISK_FREE_SPACE", "FALSE",
                       "-b", "1", "-b", "2", "-b", "3", img_entrada, img_saida]
                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.returncode == 0
            else:
                os.rename(img_entrada, img_saida)
                return True
    except:
        return False

def salvar_log(linhas):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n=== Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for linha in linhas:
            f.write(linha + "\n")
        f.write(f"=== Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

def processar(pasta_raiz, shapefile, window):
    imagens = [os.path.join(dp, f) for dp, _, files in os.walk(pasta_raiz) for f in files if f.lower().endswith(".tif")]
    total = len(imagens)
    window["-PROG-"].update(current_count=0, max=total)
    count = 0
    log_linhas = []

    gdf = gpd.read_file(shapefile).to_crs(epsg=31983)

    for img in imagens:
        nome = os.path.basename(img)
        pasta_img = os.path.dirname(img)
        saida_dir = os.path.join(pasta_img, "IMAGENS_TRANSFORMADAS")
        os.makedirs(saida_dir, exist_ok=True)
        temp = os.path.join(saida_dir, f"temp__{nome}")
        saida_final = os.path.join(saida_dir, nome)

        window["-PROG-TEXT-"].update(f"Processando: {nome}")

        if os.path.exists(saida_final):
            log_linhas.append(f"{nome}: já existente")
            count += 1
            window["-PROG-"].update(current_count=count)
            continue

        epsg = get_epsg_code(img)
        atribuiu_epsg = False
        reprojetar = epsg != "31983"

        if not epsg:
            subprocess.run(["python", GDALEDIT, "-a_srs", TARGET_EPSG, img], check=True)
            atribuiu_epsg = True
            reprojetar = False

        if reprojetar:
            cmd_warp = [GDALWARP, "--config", "CHECK_DISK_FREE_SPACE", "FALSE", "-t_srs", TARGET_EPSG,
                        "-multi", "-wo", f"NUM_THREADS={NUM_THREADS}", "-dstalpha",
                        "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER", "-overwrite", img, temp]
            res = subprocess.run(cmd_warp, capture_output=True, text=True)
            if res.returncode != 0:
                log_linhas.append(f"{nome}: erro no gdalwarp — {res.stderr.strip()}")
                continue
        else:
            temp = img

        try:
            with rasterio.open(temp) as src:
                bounds = src.bounds
                geom = box(*bounds)
                intersecta = gdf[gdf.intersects(geom)]
                if intersecta.empty:
                    log_linhas.append(f"{nome}: sem interseção com shape")
                    if temp != img and os.path.exists(temp): os.remove(temp)
                    count += 1
                    window["-PROG-"].update(current_count=count)
                    continue

                geoms = [mapping(g) for g in intersecta.geometry]
                img_cortada, out_transform = mask(src, geoms, crop=True, nodata=255)
                meta = src.meta.copy()
                meta.update({"height": img_cortada.shape[1], "width": img_cortada.shape[2],
                             "transform": out_transform, "nodata": 255})

            cortada_temp = os.path.join(saida_dir, f"recorte__{nome}")
            with rasterio.open(cortada_temp, "w", **meta) as dest:
                dest.write(img_cortada)

            ok = remover_banda_alpha(cortada_temp, saida_final)
            if ok and os.path.exists(cortada_temp): os.remove(cortada_temp)
            if temp != img and os.path.exists(temp): os.remove(temp)

            if ok:
                jp2_saida = os.path.splitext(saida_final)[0] + ".jp2"
                cmd_jp2 = [GDAL_TRANSLATE, "-of", "JP2OpenJPEG", "-co", "QUALITY=10",
                           "-co", "REVERSIBLE=YES", "--config", "GDAL_NUM_THREADS", "ALL_CPUS",
                           "--config", "CHECK_DISK_FREE_SPACE", "FALSE", saida_final, jp2_saida]
                res_jp2 = subprocess.run(cmd_jp2, capture_output=True, text=True)
                if res_jp2.returncode == 0:
                    log_linhas.append(f"{nome}: convertido para JP2")
                else:
                    log_linhas.append(f"{nome}: erro ao converter JP2 — {res_jp2.stderr.strip()}")

            acao = []
            if reprojetar: acao.append("reprojetado")
            if atribuiu_epsg: acao.append("EPSG atribuído")
            acao.append("recortado")
            log_linhas.append(f"{nome}: {' + '.join(acao)}")

        except Exception as e:
            log_linhas.append(f"{nome}: erro inesperado — {str(e)}")

        count += 1
        window["-PROG-"].update(current_count=count)

    salvar_log(log_linhas)
    window["-OUT-"].print("\n🎉 Finalizado. Log salvo em:", LOG_PATH)

# === INTERFACE ===
layout = [
    [sg.Text("Pasta com imagens:")],
    [sg.Input(key="-FOLDER-"), sg.FolderBrowse()],
    [sg.Text("Shapefile de corte (.shp):")],
    [sg.Input(key="-SHP-"), sg.FileBrowse(file_types=(("Shapefile", "*.shp"),))],
    [sg.Text("", size=(60, 1), key="-PROG-TEXT-")],
    [sg.ProgressBar(100, orientation='h', size=(50, 20), key='-PROG-')],
    [sg.Button("Iniciar", key="-START-"), sg.Button("Sair")],
    [sg.Output(size=(90, 20), key='-OUT-')]
]
window = sg.Window("Processamento de Imagens Georreferenciadas", layout, finalize=True)

def iniciar_processamento(folder, shp, window):
    processar(folder, shp, window)

while True:
    event, values = window.read(timeout=100)
    if event == sg.WIN_CLOSED or event == "Sair":
        break
    elif event == "-START-":
        folder = values["-FOLDER-"]
        shp = values["-SHP-"]
        if not folder or not os.path.isdir(folder):
            sg.popup_error("Selecione uma pasta válida.")
        elif not shp or not os.path.isfile(shp):
            sg.popup_error("Selecione um shapefile válido.")
        else:
            window["-START-"].update(disabled=True)
            threading.Thread(target=iniciar_processamento, args=(folder, shp, window), daemon=True).start()

window.close()
