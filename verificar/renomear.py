#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import argparse
import unicodedata
from pathlib import Path
from typing import Optional, Tuple

# ====== Configurações de "semelhança" nos nomes dos PDFs ======
# Vamos normalizar (lowercase, sem acentos, sem pontuação) e procurar esses tokens.
AJUSTE_TOKENS = {"ajuste", "rede"}  # "Relatório de Ajuste de Rede"
LINHAS_BASE_TOKENS = {"processamento", "linhas", "base"}  # "Relatório de processamento das linhas de base"

# ====== Regex para extrair informações do caminho ======
RE_LOTE = re.compile(r"(?:^|[/\\])LOTE[_\-\s]?(\d{1,3})(?:[/\\]|$)", re.IGNORECASE)
RE_BLOCO = re.compile(r"(?:^|[/\\])BLOCO[_\-\s]?([A-Z])(?:[/\\]|$)", re.IGNORECASE)
RE_DIA = re.compile(r"(?:^|[/\\])DIA[_\-\s]?(\d{3})[_\-\s]?(\d{4})(?:[/\\]|$)", re.IGNORECASE)

def strip_accents_and_simplify(s: str) -> str:
    """lower, remove acentos e caracteres não alfanum/espaco; colapsa espaços."""
    s = s.lower()
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )
    # troca separadores por espaço
    s = re.sub(r"[_\-\.]+", " ", s)
    # remove qualquer caractere que não seja letra/número/espaço
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    # colapsa múltiplos espaços
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens_in(name: str) -> set:
    return set(strip_accents_and_simplify(name).split())

def classify_pdf(filename: str) -> Optional[str]:
    """Classifica o PDF como 'AJUSTE' ou 'LINHAS_BASE' pelos tokens; None se não bater."""
    toks = tokens_in(filename)
    if AJUSTE_TOKENS.issubset(toks):
        return "AJUSTE"
    if LINHAS_BASE_TOKENS.issubset(toks):
        return "LINHAS_BASE"
    return None

def extract_from_path(path: Path) -> Optional[Tuple[str, str, str, str]]:
    """
    Extrai (lote_2d, bloco_1c, dia_3d, ano_4d) do caminho completo.
    Ex.: .../LOTE_09/.../BLOCO_G/.../DIA_231_2025 -> ("09","G","231","2025")
    """
    p = str(path)
    m_lote = RE_LOTE.search(p)
    m_bloco = RE_BLOCO.search(p)
    m_dia = RE_DIA.search(p)

    if not (m_lote and m_bloco and m_dia):
        return None

    lote = f"{int(m_lote.group(1)):02d}"  # zero-pad para 2 dígitos
    bloco = m_bloco.group(1).upper()
    dia = m_dia.group(1)
    ano = m_dia.group(2)
    return lote, bloco, dia, ano

def target_name(kind: str, lote: str, bloco: str, dia: str, ano: str) -> str:
    """
    kind: 'AJUSTE' | 'LINHAS_BASE'
    """
    if kind == "AJUSTE":
        base = f"ES_L{lote}_{bloco}_AJUSTE_DE_REDE_DIA_{dia}_{ano}"
    else:
        base = f"ES_L{lote}_{bloco}_PROCESSAMENTO_DAS_LINHAS_DE_BASE_DIA_{dia}_{ano}"
    return base + ".pdf"

def looks_already_renamed(name: str) -> bool:
    # Checa se já segue o padrão ES_Lxx_Y_..._DIA_ddd_yyyy.pdf
    pat = re.compile(r"^ES_L\d{2}_[A-Z]_.+_DIA_\d{3}_\d{4}\.pdf$", re.IGNORECASE)
    return bool(pat.match(name))

def rename_in_dir(dirpath: Path, commit: bool) -> None:
    meta = extract_from_path(dirpath)
    if not meta:
        return
    lote, bloco, dia, ano = meta

    # Procura PDFs no diretório atual
    for f in dirpath.iterdir():
        if not f.is_file() or f.suffix.lower() != ".pdf":
            continue

        if looks_already_renamed(f.name):
            print(f"[SKIP] Já está no padrão: {f}")
            continue

        kind = classify_pdf(f.name)
        if not kind:
            # Tenta classificar também pelo pai (às vezes o próprio diretório dá pista)
            kind = classify_pdf(dirpath.name)

        if not kind:
            print(f"[WARN] Não reconhecido (não contém tokens esperados): {f.name}")
            continue

        new_name = target_name(kind, lote, bloco, dia, ano)
        dst = f.with_name(new_name)

        # Evita overwrite: se já existir, acrescenta sufixo incremental
        if dst.exists():
            stem = Path(new_name).stem
            ext = Path(new_name).suffix
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
    Varre recursivamente a partir de root e tenta renomear PDFs em pastas que
    contenham LOTE_xx, BLOCO_Y e DIA_ddd_yyyy em seu caminho.
    """
    # Heurística: só tenta em diretórios que aparentam ser a pasta do DIA
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        if RE_DIA.search(str(p)):
            rename_in_dir(p, commit)

def main():
    ap = argparse.ArgumentParser(
        description="Renomeia PDFs de apoio de campo para o padrão ES_LXX_Y_..._DIA_DDD_YYYY.pdf"
    )
    ap.add_argument(
        "root",
        help="Pasta raiz para varrer (ex.: \\\\192.168.2.28\\i\\80225_PROJETO_IAT_PARANA\\3 Execução de voo\\ENTREGA)"
    )
    ap.add_argument(
        "--commit",
        action="store_true",
        help="Aplicar as mudanças (por padrão é DRY-RUN apenas mostrando o que faria)"
    )
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
