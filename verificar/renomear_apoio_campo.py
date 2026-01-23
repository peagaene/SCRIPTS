#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import argparse
import unicodedata
from pathlib import Path
from typing import Optional, Tuple

# ====== Tokens/regex p/ Classe I/II ======
AJUSTE_TOKENS = {"ajuste", "rede"}                         # "Relatório de Ajuste de Rede"
LINHAS_BASE_TOKENS = {"processamento", "linhas", "base"}   # "Relatório de processamento das linhas de base"
FECHAMENTO_TOKENS = {"fechamento", "transporte", "coordenadas"}  # "Fechamento do Transporte de Coordenadas" (Classe I)

RE_LOTE = re.compile(r"(?:^|[/\\])LOTE[_\-\s]?(\d{1,3})(?:[/\\]|$)", re.IGNORECASE)
RE_BLOCO = re.compile(r"(?:^|[/\\])BLOCO[_\-\s]?([A-Z])(?:[/\\]|$)", re.IGNORECASE)
RE_CLASSE = re.compile(r"(?:^|[/\\])([12])_CLASSE_([IiVv]+)(?:[/\\]|$)")
RE_DIA = re.compile(
    r"(?:^|[/\\])DIA[_\-\s]?("
    r"(?:\d{3}(?:[_\-\s]?\d{3})*)"       # '231' OU '231_232' (ou mais)
    r")[_\-\s]?(\d{4})(?:[/\\]|$)",
    re.IGNORECASE
)

def strip_accents_and_simplify(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[_\-\.\(\)\[\]]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens_in(name: str) -> set:
    return set(strip_accents_and_simplify(name).split())

def classify_pdf(filename: str) -> Optional[str]:
    toks = tokens_in(filename)
    if AJUSTE_TOKENS.issubset(toks):
        return "AJUSTE"
    if FECHAMENTO_TOKENS.issubset(toks):
        return "FECHAMENTO"
    if LINHAS_BASE_TOKENS.issubset(toks):
        return "LINHAS_BASE"
    return None

def normalize_dias_str(dias_raw: str) -> str:
    ddds = re.findall(r"\d{3}", dias_raw)
    if not ddds:
        return dias_raw
    ddds = [f"{int(d):03d}" for d in ddds]
    return "_".join(ddds)

def extract_from_path(path: Path) -> Optional[Tuple[str, str, str, str, Optional[str]]]:
    p = str(path)
    m_lote = RE_LOTE.search(p)
    m_bloco = RE_BLOCO.search(p)
    m_dia = RE_DIA.search(p)
    m_classe = RE_CLASSE.search(p)

    if not (m_lote and m_bloco and m_dia):
        return None

    lote = f"{int(m_lote.group(1)):02d}"
    bloco = m_bloco.group(1).upper()
    dias_str = normalize_dias_str(m_dia.group(1))
    ano = m_dia.group(2)

    classe = None
    if m_classe:
        numeral = m_classe.group(2).upper()
        classe = "II" if "II" in numeral else "I"

    return lote, bloco, dias_str, ano, classe

def target_name(kind: str, lote: str, bloco: str, dias_str: str, ano: str, classe: Optional[str]) -> str:
    if kind == "AJUSTE":
        base = f"ES_L{lote}_{bloco}_AJUSTE_DE_REDE_DIA_{dias_str}_{ano}"
    elif kind == "FECHAMENTO":
        base = f"ES_L{lote}_{bloco}_PROCESSAMENTO_DAS_LINHAS_DE_BASE_{dias_str}_{ano}"
    else:  # LINHAS_BASE
        base = f"ES_L{lote}_{bloco}_PROCESSAMENTO_DAS_LINHAS_DE_BASE_DIA_{dias_str}_{ano}"
    return base + ".pdf"

def looks_already_renamed(name: str) -> bool:
    pat = re.compile(r"^ES_L\d{2}_[A-Z]_.+_(?:DIA_)?\d{3}(?:_\d{3})*_\d{4}\.pdf$", re.IGNORECASE)
    return bool(pat.match(name))

# ====== Monografia (BV → MONOGRAFIA_BV_XX.pdf) ======
RE_BV_NUMBER = re.compile(r"(?<![A-Za-z0-9])B\s*V\s*[-_ ]?\s*(\d{1,4})(?![A-Za-z])", re.IGNORECASE)

def extract_bv_number_from_name(name: str) -> Optional[str]:
    m = RE_BV_NUMBER.search(name)
    if not m:
        return None
    num = m.group(1)
    n_int = int(num)
    width = max(2, len(num))  # mínimo 2 dígitos; preserva 3–4 se vier assim
    return f"{n_int:0{width}d}"

def looks_already_monografia(name: str) -> bool:
    return bool(re.match(r"^MONOGRAFIA_BV_\d{2,}\.pdf$", name, re.IGNORECASE))

def monografia_target_name(bv_num: str) -> str:
    return f"MONOGRAFIA_BV_{bv_num}.pdf"

def rename_monografia_in_dir(dirpath: Path, commit: bool) -> None:
    for f in dirpath.iterdir():
        if not f.is_file() or f.suffix.lower() != ".pdf":
            continue
        if looks_already_monografia(f.name):
            print(f"[SKIP] (Monografia) Já está no padrão: {f}")
            continue

        bv = extract_bv_number_from_name(f.stem) or extract_bv_number_from_name(f.name)
        if not bv:
            continue  # não é BVxx

        new_name = monografia_target_name(bv)
        dst = f.with_name(new_name)
        if dst.exists():
            stem, ext = dst.stem, dst.suffix
            k = 2
            while True:
                alt = f.with_name(f"{stem} ({k}){ext}")
                if not alt.exists():
                    dst = alt
                    break
                k += 1

        if commit:
            try:
                f.rename(dst)
                print(f"[OK] (Monografia) {f.name}  ->  {dst.name}")
            except Exception as e:
                print(f"[ERR] (Monografia) Falha ao renomear '{f.name}': {e}")
        else:
            print(f"[DRY-RUN] (Monografia) {f.name}  ->  {dst.name}")

# ====== Classe I/II ======
def rename_classe_in_dir(dirpath: Path, commit: bool) -> None:
    meta = extract_from_path(dirpath)
    if not meta:
        return
    lote, bloco, dias_str, ano, classe = meta
    dias_list = re.findall(r"\d{3}", dias_str)

    for f in dirpath.iterdir():
        if not f.is_file() or f.suffix.lower() != ".pdf":
            continue

        if looks_already_renamed(f.name):
            print(f"[SKIP] Já está no padrão: {f}")
            continue

        kind = classify_pdf(f.name) or classify_pdf(dirpath.name)
        if not kind:
            continue

        if kind == "FECHAMENTO":
            if classe and classe != "I":
                print(f"[WARN] '{f.name}' parece FECHAMENTO mas classe detectada não é I (classe={classe}). Pulando.")
                continue
            if len(dias_list) < 2:
                print(f"[WARN] '{f.name}' parece FECHAMENTO mas caminho não tem dois dias (tem: {dias_str}). Pulando.")
                continue

        new_name = target_name(kind, lote, bloco, dias_str, ano, classe)
        dst = f.with_name(new_name)
        if dst.exists():
            stem, ext = dst.stem, dst.suffix
            k = 2
            while True:
                alt = f.with_name(f"{stem} ({k}){ext}")
                if not alt.exists():
                    dst = alt
                    break
                k += 1

        if commit:
            try:
                f.rename(dst)
                print(f"[OK] {f.name}  ->  {dst.name}")
            except Exception as e:
                print(f"[ERR] Falha ao renomear '{f.name}': {e}")
        else:
            print(f"[DRY-RUN] {f.name}  ->  {dst.name}")

def walk_and_rename(root: Path, commit: bool) -> None:
    """
    Varre recursivamente a partir de 'root':
      • Sempre tenta renomear monografias (BVxx → MONOGRAFIA_BV_XX.pdf) em **qualquer** pasta;
      • Para Classe I/II, só renomeia em pastas que contenham 'DIA_...' no caminho.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)

        # 1) Renomear BVs em qualquer subpasta:
        rename_monografia_in_dir(p, commit)

        # 2) Renomear Classe I/II (Ajuste/Linhas/Fechamento) quando houver DIA_...:
        if RE_DIA.search(str(p)):
            rename_classe_in_dir(p, commit)

def main():
    ap = argparse.ArgumentParser(
        description=("Renomeia PDFs:\n"
                     "  • Monografias: BVxx → MONOGRAFIA_BV_XX.pdf (varre todas as subpastas)\n"
                     "  • Classe I/II: ES_LXX_Y_..._DIA_DDD[_DDD]_YYYY.pdf (ou sem 'DIA_' p/ FECHAMENTO Classe I)")
    )
    ap.add_argument("root", help="Pasta raiz (ex.: \\\\192.168.2.28\\i\\80225_PROJETO_IAT_PARANA\\3 Execução de voo\\ENTREGA\\0_ESTEIO)")
    ap.add_argument("--commit", action="store_true", help="Aplicar as mudanças (sem isso é DRY-RUN)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Raiz não encontrada: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"==> ROOT: {root}")
    print(f"==> Modo: {'COMMIT (renomeando)' if args.commit else 'DRY-RUN (sem alterações)'}")
    walk_and_rename(root, commit=args.commit)
    print("Concluído.")

if __name__ == "__main__":
    main()
