"""
Benchmark runner for REURB pipeline.

Usage (PowerShell):
  $env:REURB_TXT='D:\path\file.txt'
  $env:REURB_DADOS='D:\path\dados.dxf'
  $env:REURB_MDT='D:\path\mdt.tif'
  $env:REURB_OUT='D:\path\out'
  $env:REURB_NOME='PROJETO'
  python tools\benchmark.py
"""
from __future__ import annotations

import os
import time

from reurb.main import _executar


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    return val if val else default


def main() -> None:
    nome = _env("REURB_NOME", "BENCH")
    txt = _env("REURB_TXT")
    dados = _env("REURB_DADOS")
    mdt = _env("REURB_MDT")
    out = _env("REURB_OUT")

    if not out:
        raise SystemExit("REURB_OUT nao definido")

    settings = {
        "nome_area": nome,
        "paths": {
            "txt": txt,
            "simb_tipo": "SIMBOLOGIA",
            "dados": dados,
            "mdt": mdt,
            "saida": out,
        },
        "exports": {
            "txt": bool(txt),
            "perimetros": True,
            "curvas": bool(mdt),
            "drenagem": bool(mdt),
            "vias": True,
            "lotes_dim": True,
            "per_source": None,
        },
        "textos": {},
        "vias": {},
        "curvas": {},
        "setas": {},
    }

    t0 = time.perf_counter()
    _executar(settings)
    dt = time.perf_counter() - t0
    print(f"[BENCH] tempo_total_s={dt:.2f}")


if __name__ == "__main__":
    main()
