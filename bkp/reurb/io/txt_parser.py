"""
TXT parser utilities.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

import io
import re
import csv
from typing import Optional, Tuple

import pandas as pd


def _read_text(path: str) -> Tuple[str, str]:
    """Le o arquivo como texto (tenta UTF-8 e cai para Latin-1)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), "utf-8"
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read(), "latin-1"


def _first_nonempty_line(txt: str) -> str:
    for ln in txt.splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def _detect_sep(txt: str) -> str:
    """
    Detecta o separador via csv.Sniffer + heuristicas.
    Preferencia: ';', '\t', ',', '|', espaco.
    Fallback: ';'
    """
    sample = txt[:20000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t| ")
        if getattr(dialect, "delimiter", None):
            return dialect.delimiter
    except Exception:
        pass

    first = _first_nonempty_line(sample).lower().replace(" ", "")
    if first in ("type;e;n;z", "tipo;e;n;z"):
        return ";"
    if first in ("type,e,n,z", "tipo,e,n,z"):
        return ","

    cands = [";", "\t", ",", "|", " "]
    lines = [ln for ln in sample.splitlines() if ln.strip()][:40]
    for cand in cands:
        counts = []
        for ln in lines[:12]:
            parts = re.split(r"\s+", ln.strip()) if cand == " " else ln.split(cand)
            parts = [p for p in parts if p != ""]
            counts.append(len(parts))
        if counts and len(set(counts)) == 1 and counts[0] >= 2:
            return cand

    return ";"


def ler_bruto(path_txt: str, sep: Optional[str] = None) -> pd.DataFrame:
    """
    Le o TXT em DataFrame com colunas padronizadas:
      - **MAIUSCULAS**: TYPE, E, N, Z (compativel com processing_wrapper)
      - cria **aliases minusculos**: type, e, n, z
    Aceita com/sem cabecalho; converte virgula decimal; ignora linhas vazias.
    """
    txt, _enc = _read_text(path_txt)
    sep_eff = sep or _detect_sep(txt)
    sio = io.StringIO(txt)

    header_line = _first_nonempty_line(txt).strip().lower()

    def _norm_colname(s: str) -> str:
        return (
            s.strip()
            .lower()
            .replace(" ", "")
            .replace("\t", "")
            .replace("-", "")
            .replace("_", "")
            .replace(".", "")
        )

    synonyms = {
        "type": {"type", "tipo", "tp", "classe", "feature", "feat"},
        "e": {"e", "x", "east", "este", "utme", "easting", "coorde", "coordx"},
        "n": {"n", "y", "north", "norte", "utmn", "northing", "coordn", "coordy"},
        "z": {"z", "alt", "altitude", "cota", "elev", "elevacao", "elevation", "h"},
    }

    tokens = [
        t
        for t in re.split(r"\s+" if (sep or sep_eff) == " " else (sep or sep_eff), header_line)
        if t
    ]
    tokens_norm = [_norm_colname(t) for t in tokens]
    recognized = 0
    for t in tokens_norm:
        if any(t in syns for syns in synonyms.values()):
            recognized += 1
    has_header = recognized >= 2

    try:
        if has_header:
            df = pd.read_csv(sio, sep=sep_eff, engine="python", comment="#", skip_blank_lines=True)
            cols_in = [str(c) for c in df.columns]
            mapped: dict[str, str] = {}
            for c in cols_in:
                cn = _norm_colname(c)
                target = None
                for base, syns in synonyms.items():
                    if cn in syns:
                        target = base
                        break
                mapped[c] = target or c
            df.rename(columns=mapped, inplace=True)
        else:
            first_line = _first_nonempty_line(txt)
            parts = [
                p
                for p in (re.split(r"\s+", first_line) if sep_eff == " " else first_line.split(sep_eff))
                if p != ""
            ]
            if len(parts) >= 5:
                names = ["idx", "type", "e", "n", "z"]
            else:
                names = ["type", "e", "n", "z"]
            df = pd.read_csv(
                sio,
                sep=sep_eff,
                engine="python",
                header=None,
                names=names,
                comment="#",
                skip_blank_lines=True,
            )
    except Exception:
        sio2 = io.StringIO(txt)
        df = pd.read_csv(
            sio2,
            sep=";",
            engine="python",
            header=0 if has_header else None,
            names=None if has_header else ["type", "e", "n", "z"],
            comment="#",
            skip_blank_lines=True,
        )

    if "idx" in df.columns:
        try:
            idx_numeric_ratio = pd.to_numeric(df["idx"], errors="coerce").notna().mean()
            if idx_numeric_ratio > 0.8:
                df = df.drop(columns=["idx"])
        except Exception:
            df = df.drop(columns=["idx"], errors="ignore")

    base_cols = ["type", "e", "n", "z"]
    for c in base_cols:
        if c not in df.columns:
            df[c] = None
    df = df[base_cols]

    df["type"] = df["type"].astype(str).str.strip().str.upper()

    for col in ("e", "n", "z"):
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    try:
        med_e = pd.to_numeric(df["e"], errors="coerce").median(skipna=True)
        med_n = pd.to_numeric(df["n"], errors="coerce").median(skipna=True)
        e_is_million = (med_e is not None) and pd.notna(med_e) and (float(med_e) >= 1_000_000)
        n_is_million = (med_n is not None) and pd.notna(med_n) and (float(med_n) >= 1_000_000)
        if e_is_million and not n_is_million:
            df[["e", "n"]] = df[["n", "e"]].values
    except Exception:
        pass

    df = df.dropna(subset=["e", "n"]).reset_index(drop=True)

    df.columns = [c.upper() for c in df.columns]

    for up, low in zip(["TYPE", "E", "N", "Z"], ["type", "e", "n", "z"]):
        if low not in df.columns:
            df[low] = df[up]

    return df


def ler_txt(path_txt: str) -> pd.DataFrame:
    """Funcao publica usada pelo run."""
    return ler_bruto(path_txt)


__all__ = [
    "ler_txt",
    "ler_bruto",
]
