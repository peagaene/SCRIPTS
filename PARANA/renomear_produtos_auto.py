#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Renomeia e organiza produtos (NP/MDS/MDT/HC/INT) a partir de um caminho base.
Regras:
- NP/NPc/MDS/MDT: corrige nomes conforme pasta (5_NUVEM_PONTOS / 6_MDS / 7_MDT)
- HC/INT: renomeia IMG_HC / IMG_INTENS e move para subpastas corretas
- GeoTIFF: se nao tiver EPSG, define 31982
- Metadados HC/INT: gera TXT apos renomear e ajustar EPSG
"""

from __future__ import annotations
import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple, Iterable

try:
    from osgeo import gdal, osr
    _HAS_GDAL = True
except Exception:
    gdal = None
    osr = None
    _HAS_GDAL = False

try:
    import rasterio
    from rasterio.crs import CRS
    _HAS_RASTERIO = True
except Exception:
    rasterio = None
    CRS = None
    _HAS_RASTERIO = False

try:
    from metadados import build_metadata_text
    _HAS_METADATA = True
except Exception:
    build_metadata_text = None
    _HAS_METADATA = False

try:
    import PySimpleGUI as sg
    _HAS_SG = True
except Exception:
    sg = None
    _HAS_SG = False


IGNORED_DIRS = {
    "System Volume Information",
    "$RECYCLE.BIN",
    "Recycle.Bin",
    "FOUND.000",
    "Config.Msi",
}


def remover_revisoes(base: str) -> str:
    partes = base.split("_")
    while partes and re.fullmatch(r"R\d+", partes[-1]):
        partes.pop()
    return "_".join(partes)


def aplicar_regra(base: str, novo_trecho: str) -> str:
    base_lower = base.lower()
    if not any(x in base_lower for x in ("_np_", "_npc_c_", "_npc_t_", "_mdt_", "_mds_")):
        return ""
    base = remover_revisoes(base)
    base = base.replace("-", "_")
    base = re.sub(r"_(NPc?_C|NPc?_T|NP|MDT|MDS)_", novo_trecho, base, flags=re.IGNORECASE)
    if not base.endswith("_R0"):
        base = f"{base}_R0"
    return base


def ensure_tif_epsg(tif_path: Path, default_epsg: int = 31982) -> Optional[int]:
    if not tif_path.is_file():
        return None
    if _HAS_GDAL:
        ds = gdal.Open(str(tif_path), gdal.GA_Update)
        if ds is None:
            raise RuntimeError("GDAL nao conseguiu abrir o arquivo.")
        proj = ds.GetProjection()
        epsg = None
        if proj:
            srs = osr.SpatialReference()
            if srs.ImportFromWkt(proj) == 0:
                try:
                    srs.AutoIdentifyEPSG()
                except Exception:
                    pass
                epsg = srs.GetAuthorityCode(None)
                if epsg:
                    try:
                        epsg = int(epsg)
                    except Exception:
                        epsg = None
        if not epsg:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(int(default_epsg))
            ds.SetProjection(srs.ExportToWkt())
            ds.FlushCache()
            epsg = int(default_epsg)
        ds = None
        return epsg
    if _HAS_RASTERIO:
        with rasterio.open(str(tif_path), "r+") as ds:
            epsg = ds.crs.to_epsg() if ds.crs else None
            if not epsg:
                ds.crs = CRS.from_epsg(int(default_epsg))
                epsg = int(default_epsg)
        return epsg
    raise RuntimeError("GDAL/rasterio nao disponivel para consultar/definir EPSG.")


def iter_files_safely(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for name in filenames:
            yield Path(dirpath) / name


def detect_prefix(parts: list[str]) -> str:
    return ("_".join(parts[:3]) + "_") if len(parts) >= 3 else ("_".join(parts) + ("_" if parts else ""))


def normalize_code(token: str) -> str:
    return token.replace("-", "_").replace(" ", "_")


def categorize_suffix(raw_suffix: str) -> Optional[str]:
    s = raw_suffix.strip("_").lower()
    if s in {"hc", "int"}:
        return s
    return None


def build_new_name_img(prefix: str, category: str, code: str) -> str:
    tag = "IMG_HC" if category == "hc" else "IMG_INTENS"
    return f"{prefix}{tag}_{code}_R0"


def parse_imagem_name(filename: str) -> Tuple[Optional[str], Optional[str]]:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 5:
        return (None, None)
    prefix = detect_prefix(parts)
    raw_suffix = parts[-1]
    category = categorize_suffix(raw_suffix)
    if category is None:
        return (None, None)
    mid_tokens = parts[3:-1]
    if not mid_tokens:
        return (None, None)
    code = normalize_code(mid_tokens[-1])
    return build_new_name_img(prefix, category, code), category


def find_lote_bloco(parts: list[str]) -> Tuple[Optional[str], Optional[str]]:
    lote = None
    bloco = None
    for p in parts:
        up = p.upper()
        if up.startswith("LOTE_"):
            lote = p
        elif up.startswith("BLOCO_"):
            bloco = p
    return lote, bloco


def move_or_rename(src: Path, dest: Path, log, dry_run: bool) -> None:
    if src.resolve() == dest.resolve():
        return
    log(f"[MOVE] {src} -> {dest}")
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))


def process_np_mds_mdt(base_dir: Path, log, dry_run: bool) -> None:
    roots = [
        base_dir / "5_NUVEM_PONTOS",
        base_dir / "6_MDS",
        base_dir / "7_MDT",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for p in iter_files_safely(root):
            if not p.is_file():
                continue
            rel_parts = [part for part in p.parts if part.upper() in {"1_NP", "2_NPC_COMPLETO", "3_NPC_TERRENO"}]
            novo_trecho = None
            if "5_NUVEM_PONTOS" in (s.upper() for s in p.parts):
                if "1_NP" in (s.upper() for s in rel_parts):
                    novo_trecho = "_NP_"
                elif "2_NPC_COMPLETO" in (s.upper() for s in rel_parts):
                    novo_trecho = "_NPc_C_"
                elif "3_NPC_TERRENO" in (s.upper() for s in rel_parts):
                    novo_trecho = "_NPc_T_"
            if "6_MDS" in (s.upper() for s in p.parts):
                novo_trecho = "_MDS_"
            if "7_MDT" in (s.upper() for s in p.parts):
                novo_trecho = "_MDT_"
            if not novo_trecho:
                continue
            if p.suffix.lower() in {".tif", ".tiff"}:
                if dry_run:
                    log(f"[EPSG] DRY-RUN {p}")
                else:
                    ensure_tif_epsg(p, 31982)
            new_base = aplicar_regra(p.stem, novo_trecho)
            if not new_base:
                continue
            dest = p.with_name(f"{new_base}{p.suffix.lower()}")
            if dest != p:
                log(f"[RENAME] {p.name} -> {dest.name}")
                if not dry_run:
                    p.rename(dest)


def process_hc_int(base_dir: Path, log, dry_run: bool) -> None:
    roots = [
        base_dir / "8_IMG_HIPSOMETRICA_COMPOSTA",
        base_dir / "9_IMG_INTENSIDADE",
    ]
    tifs_for_metadata: list[Tuple[Path, str]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in iter_files_safely(root):
            if not p.is_file():
                continue
            new_base, category = parse_imagem_name(p.name)
            if not new_base or not category:
                continue
            ext = p.suffix.lower()
            lote, bloco = find_lote_bloco(list(p.parts))
            if not lote or not bloco:
                # sem lote/bloco, renomeia no lugar
                dest = p.with_name(f"{new_base}{ext}")
                if dest != p:
                    log(f"[RENAME] {p.name} -> {dest.name}")
                    if not dry_run:
                        p.rename(dest)
                continue

            if category == "hc":
                base_root = base_dir / "8_IMG_HIPSOMETRICA_COMPOSTA" / lote / bloco
                if ext in {".tif", ".tiff"}:
                    sub = "2_GEOTIFF"
                elif ext == ".ecw":
                    sub = "1_ECW"
                else:
                    sub = None
                if ext in {".tif", ".tiff"}:
                    if dry_run:
                        log(f"[EPSG] DRY-RUN {p}")
                    else:
                        ensure_tif_epsg(p, 31982)
            else:
                base_root = base_dir / "9_IMG_INTENSIDADE" / lote / bloco
                if ext in {".tif", ".tiff"}:
                    sub = "3_GEOTIFF"
                elif ext == ".ecw":
                    sub = "2_ECW"
                elif ext in {".txt", ".asc", ".xyz"}:
                    sub = "1_ASCII"
                else:
                    sub = None
                if ext in {".tif", ".tiff"}:
                    if dry_run:
                        log(f"[EPSG] DRY-RUN {p}")
                    else:
                        ensure_tif_epsg(p, 31982)

            if sub:
                dest = base_root / sub / f"{new_base}{ext}"
                move_or_rename(p, dest, log, dry_run)
                if ext in {".tif", ".tiff"}:
                    tifs_for_metadata.append((dest, category))
            else:
                dest = p.with_name(f"{new_base}{ext}")
                if dest != p:
                    log(f"[RENAME] {p.name} -> {dest.name}")
                    if not dry_run:
                        p.rename(dest)
                if ext in {".tif", ".tiff"}:
                    tifs_for_metadata.append((dest, category))

    if not _HAS_METADATA:
        log("[WARN] metadados.py nao importado. Pulo de metadados.")
        return

    for tif_path, category in tifs_for_metadata:
        try:
            if category == "hc":
                meta_dir = tif_path.parent.parent / "3_METADADOS"
            else:
                meta_dir = tif_path.parent.parent / "4_METADADOS"
            out_txt = meta_dir / (tif_path.stem + ".txt")
            log(f"[META] {tif_path.name} -> {out_txt}")
            if not dry_run:
                meta_dir.mkdir(parents=True, exist_ok=True)
                meta_text = build_metadata_text(str(tif_path))
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(meta_text)
        except Exception as e:
            log(f"[META-ERRO] {tif_path} -> {e}")


def run_process(base_path: Path, dry_run: bool, do_np: bool, do_hc: bool, log) -> None:
    if do_np:
        log("[INI] Processando NP/MDS/MDT...")
        process_np_mds_mdt(base_path, log, dry_run)
    if do_hc:
        log("[INI] Processando HC/INT + metadados...")
        process_hc_int(base_path, log, dry_run)
    log("[OK] Finalizado.")


def main_gui():
    sg.theme("SystemDefaultForReal")
    layout = [
        [sg.Text("Pasta base (contendo 5_NUVEM_PONTOS/6_MDS/7_MDT/8_IMG/9_IMG):")],
        [sg.Input(key="BASE", size=(70, 1)), sg.FolderBrowse("Procurar...")],
        [sg.Checkbox("Processar NP/MDS/MDT", key="DO_NP", default=True),
         sg.Checkbox("Processar HC/INT + metadados", key="DO_HC", default=True),
         sg.Checkbox("Dry-run (nao altera)", key="DRY", default=False)],
        [sg.Button("Executar", bind_return_key=True), sg.Button("Sair")],
        [sg.Multiline("", key="LOG", size=(110, 22), autoscroll=True, disabled=True)],
    ]
    window = sg.Window("Renomear Produtos (Auto)", layout, finalize=True)

    def log(msg: str):
        window["LOG"].update(msg + "\n", append=True)

    while True:
        ev, v = window.read()
        if ev in (sg.WIN_CLOSED, "Sair"):
            break
        if ev == "Executar":
            window["LOG"].update("")
            base_dir = (v.get("BASE") or "").strip()
            if not base_dir or not Path(base_dir).is_dir():
                sg.popup_error("Informe uma pasta base valida.")
                continue
            run_process(Path(base_dir), bool(v.get("DRY")), bool(v.get("DO_NP")), bool(v.get("DO_HC")), log)

    window.close()


def main():
    parser = argparse.ArgumentParser(description="Renomear e organizar produtos por padrao de pastas.")
    parser.add_argument("--base", dest="base_dir", help="Pasta base que contem 5_NUVEM_PONTOS/6_MDS/7_MDT/8_IMG.../9_IMG...")
    parser.add_argument("--gui", action="store_true", help="Abrir interface grafica (se disponivel).")
    args = parser.parse_args()

    if args.gui or (not args.base_dir and _HAS_SG):
        if not _HAS_SG:
            print("PySimpleGUI nao disponivel. Use --base no modo CLI.")
            return
        main_gui()
        return

    base_dir = args.base_dir or input("Informe a pasta base: ").strip()
    if not base_dir:
        print("Pasta base nao informada.")
        return
    base_path = Path(base_dir)
    if not base_path.is_dir():
        print("Pasta base invalida.")
        return

    def log(msg: str):
        print(msg)

    run_process(base_path, dry_run=False, do_np=True, do_hc=True, log=log)


if __name__ == "__main__":
    main()
