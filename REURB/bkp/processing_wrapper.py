# processing_wrapper.py
# Mantido apenas para compatibilidade retroativa.
from __future__ import annotations

from processing import processar_registros


def processar_registros_com_filtro(
    *, df, ms, doc, params,
    lotes, edificacoes,
    via_lines_setas, via_lines_geral,
    get_elevation,
    perimetros=None,
):
    """Encaminha diretamente para processing.processar_registros."""
    return processar_registros(
        df=df, ms=ms, doc=doc, params=params,
        lotes=lotes, edificacoes=edificacoes,
        via_lines_setas=via_lines_setas,
        via_lines_geral=via_lines_geral,
        get_elevation=get_elevation,
    )
