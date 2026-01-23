# -*- coding: utf-8 -*-
"""
OSM/OSM.PBF -> Shapefile (GDAL) com GUI (Windows-friendly)
- Detecta camadas via ogrinfo e usa: lines, multilinestrings, multipolygons (nessa ordem, se existirem)
- Caminhos com espaço OK (caminhos sempre entre aspas)
- Filtro padrão: vias com nome (highway IS NOT NULL AND name IS NOT NULL)
- Exporta para EPSG:31983 e/ou EPSG:31982
"""

import os
import subprocess
from pathlib import Path
from shutil import which
import PySimpleGUI as sg

# ==================== UTIL ====================

def have_cmd(cmd: str) -> bool:
    return which(cmd) is not None

def run_cmd(cmd: str) -> tuple[int, str, str]:
    """Executa comando e retorna (rc, stdout, stderr)."""
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def list_layers(osm_path: Path) -> list[str]:
    """Descobre camadas do arquivo OSM via ogrinfo."""
    # aspas em volta do caminho para suportar espaços
    cmd = f'ogrinfo -ro -so "{osm_path}"'
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"Falha no ogrinfo: {err.strip() or out.strip()}")
    layers = []
    for line in out.splitlines():
        s = line.strip()
        # linhas típicas: "1: points (Point)", "2: lines (LineString)" etc.
        if ":" in s and "(" in s and ")" in s:
            try:
                name = s.split(":")[1].split("(")[0].strip()
                if name:
                    layers.append(name)
            except Exception:
                pass
    return layers

def convert_osm(
    input_osm: Path,
    out_dir: Path,
    epsgs: list[int],
    basename: str,
    only_named_roads: bool = True,
    extra_fields: str | None = None,
    window: sg.Window | None = None
):
    if not have_cmd("ogr2ogr") or not have_cmd("ogrinfo"):
        raise EnvironmentError("GDAL não encontrado no PATH (ogr2ogr/ogrinfo). Abra no ambiente com GDAL.")

    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        (window["-LOG-"].print(msg) if window else print(msg))

    # 1) Descobrir camadas e selecionar candidatas
    layers_avail = list_layers(input_osm)
    log(f"Camadas detectadas: {layers_avail}")
    candidate_layers = [l for l in ["lines", "multilinestrings", "multipolygons"] if l in layers_avail]
    if not candidate_layers:
        raise RuntimeError("Nenhuma das camadas esperadas encontrada (lines/multilinestrings/multipolygons).")

    # 2) Montar filtros/campos
    where = "highway IS NOT NULL" + (" AND name IS NOT NULL" if only_named_roads else "")
    base_select = ["name", "highway"]
    if extra_fields:
        for f in [s.strip() for s in extra_fields.split(",") if s.strip()]:
            if f not in base_select:
                base_select.append(f)
    select = ",".join(base_select)

    log("Iniciando conversão...")
    log(f"Arquivo OSM: {input_osm}")
    log(f"Pasta de saída: {out_dir}")
    log(f"Filtro WHERE: {where}")
    log(f"Campos SELECT: {select}")
    log(f"Camadas usadas (ordem): {candidate_layers}")

    # 3) Converter por EPSG
    for epsg in epsgs:
        shp_path = out_dir / f"{basename}_{epsg}.shp"
        created = False

        for camada in candidate_layers:
            # primeira camada cria; demais fazem append
            update_append = "-update -append" if created else ""
            mode_label = "[append]" if created else "[create]"
            log(f">> {mode_label} camada '{camada}' em EPSG:{epsg}")

            cmd = (
                f'ogr2ogr -f "ESRI Shapefile" "{shp_path}" '
                f'"{input_osm}" {camada} '
                f'-where "{where}" -select {select} '
                f'-nlt PROMOTE_TO_MULTI -t_srs EPSG:{epsg} {update_append}'
            ).strip()

            rc, out_txt, err_txt = run_cmd(cmd)
            if out_txt.strip():
                log(out_txt.strip())
            if rc != 0:
                # pode falhar por camada vazia; registra e segue para próxima
                log(f"   Aviso: camada '{camada}' não adicionada. Detalhe: {err_txt.strip()}")
            # Considera criado se o arquivo SHP existe após a primeira chamada OK
            if shp_path.exists():
                created = True

        if not created:
            raise RuntimeError(
                f"Nenhuma feição gerada para EPSG:{epsg}. "
                f"Tente desmarcar 'Somente vias com nome' ou remover o filtro."
            )

        log(f"[OK] Gerado: {shp_path}")

    log("Conversão concluída com sucesso.")

# ==================== GUI ====================

def main():
    sg.theme("SystemDefault")

    layout = [
        [sg.Text("Arquivo OSM/OSM.PBF:"), sg.Input(key="-INFILE-", expand_x=True),
         sg.FileBrowse("Procurar...", file_types=(("OSM / PBF", "*.osm;*.pbf"),), initial_folder=os.getcwd())],
        [sg.Text("Pasta de saída:"), sg.Input(key="-OUTDIR-", expand_x=True),
         sg.FolderBrowse("Selecionar...", initial_folder=os.getcwd())],
        [sg.Text("Prefixo do shapefile (basename):"), sg.Input("vias", key="-BASENAME-", expand_x=True)],

        [sg.Frame("Projeções (EPSG)", [
            [sg.Checkbox("31983 (SIRGAS 2000 / UTM 23S)", key="-EPSG31983-", default=False)],
            [sg.Checkbox("31982 (SIRGAS 2000 / UTM 22S)", key="-EPSG31982-", default=True)],
        ])],

        [sg.Frame("Filtros e campos", [
            [sg.Checkbox("Somente vias com nome (name IS NOT NULL)", key="-ONLYNAMED-", default=True)],
            [sg.Text("Campos extras (vírgula):"), sg.Input(key="-EXTRA-", expand_x=True, tooltip="Ex.: maxspeed,oneway,surface")],
        ])],

        [sg.Button("Converter", key="-RUN-", size=(12,1), button_color=("white","green")),
         sg.Button("Sair", key="-EXIT-", size=(10,1))],

        [sg.Multiline("", key="-LOG-", size=(100,18), autoscroll=True, expand_x=True, expand_y=True, disabled=True)]
    ]

    window = sg.Window("OSM -> Shapefile (GDAL)", layout, resizable=True)

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "-EXIT-"):
            break

        if event == "-RUN-":
            window["-LOG-"].update("")  # limpa log

            infile = values["-INFILE-"]
            outdir = values["-OUTDIR-"]
            basename = (values["-BASENAME-"].strip() or "vias")
            epsgs = []
            if values["-EPSG31983-"]:
                epsgs.append(31983)
            if values["-EPSG31982-"]:
                epsgs.append(31982)

            only_named = values["-ONLYNAMED-"]
            extras = values["-EXTRA-"].strip() or None

            # validações
            if not infile or not Path(infile).exists():
                sg.popup_error("Selecione um arquivo .osm ou .pbf válido.")
                continue
            if not outdir:
                sg.popup_error("Selecione a pasta de saída.")
                continue
            if not epsgs:
                sg.popup_error("Selecione ao menos um EPSG.")
                continue

            try:
                convert_osm(
                    input_osm=Path(infile),
                    out_dir=Path(outdir),
                    epsgs=epsgs,
                    basename=basename,
                    only_named_roads=only_named,
                    extra_fields=extras,
                    window=window
                )
                sg.popup_ok("Conversão concluída com sucesso!")
            except Exception as e:
                sg.popup_error(f"Erro na conversão:\n{e}")

    window.close()

if __name__ == "__main__":
    main()
