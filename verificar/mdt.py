import os
import json
import shutil
import traceback
import tempfile
from typing import Optional

# GUI
import PySimpleGUI as sg

# Núcleo numérico / raster
import numpy as np
import rasterio
from rasterio.fill import fillnodata

# Suavização opcional
try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# PDAL
try:
    import pdal
except Exception:
    sg.popup_error(
        "PDAL não encontrado.\n\nInstale com:\n  pip install pdal",
        title="Erro de dependência"
    )
    raise

# ================== PRESETS RECOMENDADOS (subúrbio montanhoso) ==================
DEFAULTS = {
    "resolution": 0.50,   # m/pixel (use 0.25 se o GSD for muito fino e a área não for enorme)
    "epsg": 31983,        # SIRGAS 2000 / UTM 23S (ajuste para 31982 se estiver no 22S)
    "idw_radius": 6.0,    # metros (aumente p/ 8 em quarteirões maiores)
    "idw_power": 1.8,     # suaviza influência local e evita degraus
    "nodata": -9999.0,
    "do_fill": True,
    "fill_max_dist_px": 12,   # pixels (preenche telhados suavemente)
    "smooth": True,
    "smooth_sigma": 1.0,      # suavização leve
    "filter_ground": True     # manter só Class=2 (terreno)
}


def build_pdal_pipeline(input_laz: str,
                        output_tif: str,
                        epsg: int,
                        resolution: float,
                        idw_radius: float,
                        idw_power: float,
                        nodata: float,
                        filter_ground: bool) -> str:
    """
    Cria um pipeline PDAL que:
      - Lê LAZ
      - (Opcional) mantém apenas Class=2 (terreno)
      - Rasteriza via writers.gdal com método IDW
    """
    stages = [{
        "type": "readers.las",
        "filename": input_laz,
        "spatialreference": f"EPSG:{epsg}"
    }]

    if filter_ground:
        stages.append({
            "type": "filters.range",
            "limits": "Classification[2:2]"
        })

    # writers.gdal sem 'creation_options' para compatibilidade ampla
    stages.append({
        "type": "writers.gdal",
        "filename": output_tif,
        "gdaldriver": "GTiff",
        "dimension": "Z",
        "output_type": "idw",
        "power": float(idw_power),
        "radius": float(idw_radius),
        "resolution": float(resolution),
        "window_size": 0,
        "nodata": float(nodata),
        "data_type": "float32"
    })

    return json.dumps({"pipeline": stages}, indent=2)


def run_pdal(pipeline_json: str, log_elem: Optional[sg.Multiline] = None):
    pip = pdal.Pipeline(pipeline_json)
    try:
        count = pip.execute()
        if log_elem:
            log_elem.print(f"[PDAL] Pipeline executado. Pontos processados: {count}")
        return pip
    except Exception:
        if log_elem:
            log_elem.print(f"[PDAL] ERRO:\n{traceback.format_exc()}", text_color='red')
        raise


def fill_nodata_and_smooth(in_tif: str,
                           out_tif: str,
                           nodata: float,
                           do_fill: bool,
                           max_dist_px: int,
                           smooth: bool,
                           sigma: float,
                           log_elem: Optional[sg.Multiline] = None):
    """
    Preenche NoData de forma contínua a partir das bordas e aplica suavização leve opcional.
    - Usa rasterio.fill.fillnodata e gaussian_filter (se disponível).
    - Grava saída final em tiles 256×256 (múltiplos de 16) usando escrita temporária segura.
    """
    with rasterio.open(in_tif) as src:
        profile = src.profile.copy()
        arr = src.read(1).astype("float32")
        nodata_val = nodata

    # Ajuste do profile para escrita final (tiles 256×256, múltiplos de 16)
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=nodata_val,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="lzw",
        predictor=3,
        BIGTIFF="IF_SAFER"
    )

    # Máscara de NoData
    mask_nodata = np.isclose(arr, nodata_val) | np.isnan(arr)

    if do_fill:
        if log_elem:
            log_elem.print("[Fill] Preenchendo NoData (contorno → centro) ...")
        # fillnodata: usa raios em pixels
        arr = fillnodata(arr, mask=mask_nodata, max_search_distance=max_dist_px)

    if smooth:
        if not _HAS_SCIPY:
            if log_elem:
                log_elem.print("[Smooth] SciPy não disponível. Pulando suavização.", text_color='yellow')
        else:
            if log_elem:
                log_elem.print(f"[Smooth] Aplicando suavização gaussiana σ={sigma} ...")
            # Aplica filtro apenas nas áreas que não eram NoData originalmente
            valid_mask = ~mask_nodata
            sm = gaussian_filter(arr, sigma=sigma, mode='nearest')
            blend = 0.8  # mistura 80% suavizado nas áreas válidas
            arr = np.where(valid_mask, (1 - blend) * arr + blend * sm, arr)

    # --- escrita segura em arquivo temporário ---
    tmp_out = out_tif + ".tmp.tif"

    # garante que não há restos de tentativas anteriores
    for p in (tmp_out, out_tif):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    with rasterio.open(tmp_out, "w", **profile) as dst:
        dst.write(arr, 1)

    # move o temporário para o destino final (atomicidade simples)
    shutil.move(tmp_out, out_tif)


def main():
    sg.theme("SystemDefault")

    layout = [
        [sg.Text("Arquivo LAZ:"), sg.Input(key="-IN-", enable_events=True, size=(60,1)),
         sg.FileBrowse(file_types=(("LAS/LAZ", "*.las;*.laz"),))],
        [sg.Text("GeoTIFF de saída:"), sg.Input(key="-OUT-", size=(60,1)),
         sg.SaveAs(file_types=(("GeoTIFF", "*.tif"),), default_extension="tif")],
        [sg.Frame("Parâmetros principais", [
            [sg.Text("Resolução (m/pixel):"), sg.Input(str(DEFAULTS["resolution"]), key="-RES-", size=(10,1))],
            [sg.Text("CRS (EPSG):"), sg.Input(str(DEFAULTS["epsg"]), key="-EPSG-", size=(10,1))],
            [sg.Text("IDW - Raio (m):"), sg.Input(str(DEFAULTS["idw_radius"]), key="-RADIUS-", size=(10,1)),
             sg.Text("Potência:"), sg.Input(str(DEFAULTS["idw_power"]), key="-POWER-", size=(6,1))],
            [sg.Checkbox("Manter apenas Class=2 (terreno)", key="-GROUND-", default=DEFAULTS["filter_ground"])]
        ])],
        [sg.Frame("Pós-processamento", [
            [sg.Checkbox("Preencher NoData", key="-DOFILL-", default=DEFAULTS["do_fill"]),
             sg.Text("Distância máx. (px):"), sg.Input(str(DEFAULTS["fill_max_dist_px"]), key="-FILLDIST-", size=(6,1))],
            [sg.Checkbox("Suavização leve (gaussiana)", key="-SMOOTH-", default=DEFAULTS["smooth"]),
             sg.Text("σ:"), sg.Input(str(DEFAULTS["smooth_sigma"]), key="-SIGMA-", size=(6,1)),
             sg.Text("(requer scipy)")]
        ])],
        [sg.Multiline(size=(110,18), key="-LOG-", autoscroll=True, disabled=True, write_only=True)],
        [sg.Push(), sg.Button("Gerar MDT"), sg.Button("Sair")]
    ]

    win = sg.Window("LAZ → MDT GeoTIFF (IDW + Fill + Smooth)", layout, finalize=True)
    log = win["-LOG-"]

    while True:
        ev, vals = win.read()
        if ev in (sg.WINDOW_CLOSED, "Sair"):
            break
        if ev == "Gerar MDT":
            try:
                in_laz = vals["-IN-"]
                out_tif = vals["-OUT-"]
                if not in_laz or not os.path.isfile(in_laz):
                    sg.popup_error("Selecione um arquivo .las/.laz válido.")
                    continue
                if not out_tif:
                    base = os.path.splitext(os.path.basename(in_laz))[0]
                    out_tif = os.path.join(os.path.dirname(in_laz), f"{base}_MDT.tif")
                    win["-OUT-"].update(out_tif)

                # Parâmetros
                res = float(vals["-RES-"])
                epsg = int(vals["-EPSG-"])
                radius = float(vals["-RADIUS-"])
                power = float(vals["-POWER-"])
                do_fill = bool(vals["-DOFILL-"])
                filldist = int(vals["-FILLDIST-"])
                smooth = bool(vals["-SMOOTH-"])
                sigma = float(vals["-SIGMA-"])
                filter_ground = bool(vals["-GROUND-"])
                nodata = DEFAULTS["nodata"]

                log.update(value="")
                log.print("== Iniciando ==")
                log.print(f"Entrada: {in_laz}")
                log.print(f"Saída:   {out_tif}")
                log.print(f"EPSG={epsg} | Res={res} m | IDW radius={radius} m | power={power}")
                if filter_ground:
                    log.print("Filtro: manter apenas Class=2 (terreno).")
                if smooth and not _HAS_SCIPY:
                    log.print("Aviso: scipy não encontrado; suavização será ignorada.", text_color='yellow')

                # 1) Gera raster inicial via PDAL (IDW)
                tmpdir = tempfile.mkdtemp(prefix="mdt_")
                tmp_tif = os.path.join(tmpdir, "mdt_idw.tif")

                pipe_json = build_pdal_pipeline(
                    input_laz=in_laz,
                    output_tif=tmp_tif,
                    epsg=epsg,
                    resolution=res,
                    idw_radius=radius,
                    idw_power=power,
                    nodata=nodata,
                    filter_ground=filter_ground
                )
                log.print("[PDAL] Executando pipeline ...")
                run_pdal(pipe_json, log_elem=log)

                # 2) Preencher NoData e suavizar (se habilitado)
                log.print("[Raster] Pós-processamento ...")
                fill_nodata_and_smooth(
                    in_tif=tmp_tif,
                    out_tif=out_tif,
                    nodata=nodata,
                    do_fill=do_fill,
                    max_dist_px=filldist,
                    smooth=smooth,
                    sigma=sigma,
                    log_elem=log
                )

                log.print("✅ Finalizado com sucesso.", text_color='green')
                sg.popup_ok(f"MDT gerado:\n{out_tif}")

            except Exception:
                log.print("❌ ERRO ao gerar MDT:", text_color='red')
                log.print(traceback.format_exc(), text_color='red')

    win.close()


if __name__ == "__main__":
    main()
