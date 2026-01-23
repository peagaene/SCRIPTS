#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI para renomear/mover apenas IMAGENS (HC/INT/xyz).
Saida esperada: ES_L.._K_IMG_HC/IMG_INTENS_<COD>_R0
Pode mover por extensao ou renomear no lugar.

Requer: pip install PySimpleGUI
"""

from __future__ import annotations
import PySimpleGUI as sg
from pathlib import Path
import os, shutil, json
from typing import Optional, Tuple, Iterable

# ------------------------ Pastas a ignorar ------------------------
IGNORED_DIRS = {
    "System Volume Information",
    "$RECYCLE.BIN",
    "Recycle.Bin",
    "FOUND.000",
    "Config.Msi",
}

CONFIG_FILE = "organizar_arquivos_gui_config.json"

# ------------------------ Utilidades de nome ------------------------
def detect_prefix(parts: list[str]) -> str:
    return ("_".join(parts[:3]) + "_") if len(parts) >= 3 else ("_".join(parts) + ("_" if parts else ""))

def normalize_code(token: str) -> str:
    return token.replace("-", "_").replace(" ", "_")

def categorize_suffix(raw_suffix: str) -> Optional[str]:
    s = raw_suffix.strip("_").lower()
    if s in {"hc", "int", "xyz"}:
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

# ------------------------ Iteracao segura ------------------------
def iter_files_safely(root: Path, recursive: bool, log) -> Iterable[Path]:
    if not root:
        return
    if not recursive:
        try:
            with os.scandir(root) as it:
                for entry in it:
                    try:
                        if entry.is_file():
                            yield Path(entry.path)
                    except PermissionError:
                        log(f"[AVISO] Sem permissao: {entry.path}")
                        continue
        except PermissionError:
            log(f"[AVISO] Sem permissao: {root}")
        return
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for name in filenames:
            yield Path(dirpath) / name

# ------------------------ IMAGENS (mover opcional) ------------------------
def route_dest_img(mapping: dict, category: str, ext: str) -> Optional[Path]:
    ext_l = ext.lower().lstrip(".")
    key = None
    if category == "hc":
        if ext_l in {"tif", "tiff"}:
            key = "HC_TIF"
    elif category in {"int", "xyz"}:
        if ext_l in {"tif", "tiff"}:
            key = "INT_TIF"
        elif ext_l == "xyz":
            key = "INT_XYZ"
    return Path(mapping[key]) if key and mapping.get(key) else None

def process_imagens(input_dir: Path, mapping: dict, recursive: bool, dry_run: bool, log) -> None:
    moved = renamed_in_place = 0
    for p in iter_files_safely(input_dir, recursive, log):
        if not p.is_file():
            continue
        new_base, category = parse_imagem_name(p.name)
        if not new_base or not category:
            continue
        ext = p.suffix
        dest_dir = route_dest_img(mapping, category, ext)
        if dest_dir:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{new_base}{ext.lower()}"
            log(f"[IMG->MOVE] {p.name} -> {dest_path}")
            if not dry_run:
                try:
                    shutil.move(str(p), str(dest_path))
                except PermissionError:
                    log(f"[AVISO] Sem permissao p/ mover: {p}")
                except Exception as e:
                    log(f"[ERRO] {p} -> {e}")
                else:
                    moved += 1
        else:
            dest_path = p.with_name(f"{new_base}{ext.lower()}")
            if dest_path == p:
                continue
            log(f"[IMG->RENAME] {p.name} -> {dest_path.name}")
            if not dry_run:
                try:
                    p.rename(dest_path)
                except PermissionError:
                    log(f"[AVISO] Sem permissao p/ renomear: {p}")
                except Exception as e:
                    log(f"[ERRO] {p} -> {e}")
                else:
                    renamed_in_place += 1
    log(f"[IMAGENS] Movidos: {moved} | Renomeados no lugar: {renamed_in_place}")

# ------------------------ Config ------------------------
def load_config() -> dict:
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(cfg: dict):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ------------------------ GUI ------------------------
def main():
    sg.theme("SystemDefaultForReal")
    cfg = load_config()

    frame_imagens = [
        [sg.Checkbox("Ativar modulo IMAGENS (HC/INT/xyz)", key="MOD_IMG", default=cfg.get("MOD_IMG", True))],
        [sg.Text("Entrada (IMAGENS):"), sg.Input(cfg.get("IMG_IN", ""), key="IMG_IN", size=(60, 1)), sg.FolderBrowse("Procurar...")],
        [sg.Text("Destinos (opcionais - se vazio, renomeia no lugar)")],
        [sg.Text("HC -> TIF:"), sg.Input(cfg.get("HC_TIF", ""), key="HC_TIF", size=(55, 1)), sg.FolderBrowse()],
        [sg.Text("INT -> TIF:"), sg.Input(cfg.get("INT_TIF", ""), key="INT_TIF", size=(55, 1)), sg.FolderBrowse()],
        [sg.Text("INT -> XYZ:"), sg.Input(cfg.get("INT_XYZ", ""), key="INT_XYZ", size=(55, 1)), sg.FolderBrowse()],
    ]

    layout = [
        [sg.Frame("Imagens (HC/INT/xyz)", frame_imagens)],
        [sg.Checkbox("Recursivo", key="RECURSIVE", default=cfg.get("RECURSIVE", True)),
         sg.Checkbox("Dry-run (nao move, so simula)", key="DRYRUN", default=cfg.get("DRYRUN", False))],
        [sg.Button("Executar", bind_return_key=True), sg.Button("Salvar config"), sg.Button("Sair")],
        [sg.Multiline("", key="LOG", size=(110, 22), autoscroll=True, disabled=True)],
    ]

    window = sg.Window("Organizar Arquivos (HC/INT/xyz)", layout, finalize=True)

    def log(msg: str):
        window["LOG"].update(msg + "\n", append=True)

    while True:
        ev, v = window.read()
        if ev in (sg.WIN_CLOSED, "Sair"):
            break

        if ev == "Salvar config":
            new_cfg = {
                "MOD_IMG": bool(v.get("MOD_IMG", True)),
                "IMG_IN": v.get("IMG_IN", ""),
                "HC_TIF": v.get("HC_TIF", ""),
                "INT_TIF": v.get("INT_TIF", ""),
                "INT_XYZ": v.get("INT_XYZ", ""),
                "RECURSIVE": bool(v.get("RECURSIVE", True)),
                "DRYRUN": bool(v.get("DRYRUN", False)),
            }
            save_config(new_cfg)
            log("[INFO] Configuracao salva.")
            continue

        if ev == "Executar":
            window["LOG"].update("")
            recursive = bool(v.get("RECURSIVE", True))
            dryrun = bool(v.get("DRYRUN", False))
            try:
                if v.get("MOD_IMG", False):
                    img_in = v.get("IMG_IN", "")
                    if img_in:
                        mapping_img = {
                            "HC_TIF": v.get("HC_TIF", ""),
                            "INT_TIF": v.get("INT_TIF", ""),
                            "INT_XYZ": v.get("INT_XYZ", ""),
                        }
                        log("[IMAGENS] Iniciando...")
                        process_imagens(Path(img_in), mapping_img, recursive, dryrun, log)
                    else:
                        log("[IMAGENS] PULO: pasta de entrada nao definida.")

                log("\n[OK] Execucao finalizada.")

            except Exception as e:
                log(f"[ERRO] Execucao interrompida: {e}")

    window.close()


if __name__ == "__main__":
    main()
