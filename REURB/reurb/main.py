"""
REURB main entrypoint (modular).
"""
from __future__ import annotations

import copy
import time
from contextlib import nullcontext

from shapely.ops import unary_union
from shapely.geometry import MultiLineString
from osgeo import gdal, osr

gdal.UseExceptions()

from reurb.config.layers import (
    FIXED_SIMBOLOGIA_PATH,
    STYLE_TEXTO,
    LAYER_SOLEIRA_NUM_PAV,
    LAYER_LOTES,
    LAYER_EDIF,
    LAYER_PERIMETRO,
    LAYER_PER_LEVANTAMENTO,
    LAYER_EIXO_VIA,
    LAYERS_PISTA_BORDA,
    LAYER_SOLEIRA_AREA,
)
from reurb.config.mappings import TYPE_TO_LAYER, LAYER_TO_BLOCK
from reurb.config.dimensions import GLOBAL_PARAMS

from reurb.io.dxf_io import (
    abrir_dxf_simbologia,
    salvar_dxf,
    garantir_estilos_blocos,
    carregar_poligonos_por_layer,
    carregar_linhas_por_layers,
)
from reurb.io.txt_parser import ler_txt
from reurb.io.mdt_handler import make_get_elevation, make_get_elevation_from_src
from reurb.io.shp_reader import build_shp_name_provider
from reurb.symbology.profiles import build_layer_profile

from reurb.processors.txt_blocks import processar_registros
from reurb.processors.drainage import inserir_setas_drenagem
from reurb.processors.perimeter import processar_perimetros
from reurb.processors.lot_dimensions import processar_lotes_dimensoes, _rotular_areas_lote
from reurb.processors.roads import medir_e_rotular_vias
from reurb.processors.contours import gerar_curvas_nivel
from reurb.renderers.table_renderer import create_area_table

from reurb.ui.app import abrir_ui
from reurb.utils.resource_manager import ResourceManager
from reurb.utils.logging_utils import REURBLogger


# layers de ENTRADA (agora vindo do config)
LYR_EIXO = LAYER_EIXO_VIA
LYR_GUIA_COM = "SISTVIA_PAV_COM_GUIA"
LYR_GUIA_SEM = "SISTVIA_PAV_SEM_GUIA"
LYR_DIV_FIS_LOTE = LAYER_LOTES
LYR_EDIF = LAYER_EDIF
LYR_PER_INTERESSE = LAYER_PERIMETRO
LYR_PER_LEVANT = LAYER_PER_LEVANTAMENTO


def _epsg_from_gdal_dataset(ds):
    if ds is None:
        return None, "sem dataset"
    try:
        wkt = ds.GetProjection()
        if not wkt:
            return None, "sem WKT"
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        try:
            srs.AutoIdentifyEPSG()
        except Exception as e:
            # Nao e critico; EPSG pode permanecer indeterminado.
            logger = REURBLogger(__name__)
            logger.debug(f"Falha ao auto-identificar EPSG: {e}")
        code = srs.GetAuthorityCode("PROJCS") or srs.GetAuthorityCode("GEOGCS")
        return (int(code) if code else None), ("authority" if code else "indeterminado")
    except Exception as e:
        return None, f"erro:{e}"


def _executar(settings):
    """Executa o pipeline REURB com as configuracoes da UI."""
    if not settings:
        return

    nome_area = settings["nome_area"]
    P = settings["paths"]
    X = settings["exports"]
    T = settings["textos"]
    V = settings["vias"]
    C = settings["curvas"]
    S = settings["setas"]

    logger = REURBLogger(verbose=True)

    # === 2) Simbologia fixa ===
    simb_path = FIXED_SIMBOLOGIA_PATH
    print(f"[INFO] DXF de simbologia (fixo): {simb_path}")

    # === 3) Abrir doc de saida e perfil de camadas ===
    doc, ms = abrir_dxf_simbologia(simb_path)
    garantir_estilos_blocos(doc, GLOBAL_PARAMS)
    perfil = build_layer_profile(doc, simb_path)
    print("[INFO] Perfil de camadas:", perfil)

    # === 4) Params runtime ===
    params = copy.deepcopy(GLOBAL_PARAMS)
    for k, v in T.items():
        setattr(params, k, v)
    for k, v in C.items():
        setattr(params, k, v)
    for k, v in S.items():
        setattr(params, k, v)
    setattr(params, "style_texto", STYLE_TEXTO)
    setattr(params, "via_dim_style", "Cota_Rua")
    setattr(params, "layer_curvas", perfil["curva_i"])
    setattr(params, "layer_curvas_mestra", perfil["curva_m"])
    setattr(params, "layer_curvas_txt", perfil["curva_txt"])
    setattr(params, "layer_per_out_tab", perfil["per_tab"])
    setattr(params, "layer_per_out_vert", perfil["per_vert"])
    setattr(params, "layer_txt_grande", perfil.get("txt_grande"))
    setattr(params, "layer_txt_soleira", perfil.get("txt_soleira"))
    setattr(params, "layer_soleira_num_pav", LAYER_SOLEIRA_NUM_PAV)
    setattr(params, "type_to_layer", dict(TYPE_TO_LAYER))
    setattr(params, "layer_to_block", dict(LAYER_TO_BLOCK))
    setattr(params, "escrever_area_lote", True)
    setattr(params, "soleira_num_pav_mtext", True)
    setattr(params, "rotacionar_numero_casa", False)

    # === 5) Entradas ===
    print("[INFO] Arquivos selecionados:")
    print(f"   TXT: {P['txt'] or ''}")
    print(f"   DXF SIMB: {simb_path}")
    print(f"   DXF DADOS: {P['dados'] or ''}")
    print(f"   MDT: {P['mdt'] or ''}")
    print(f"   SAIDA: {P['saida']}")

    need_mdt_reasons = []
    if X.get("txt"):
        need_mdt_reasons.append("TXT (blocos)")
    if X.get("curvas"):
        need_mdt_reasons.append("curvas de nivel")
    if X.get("drenagem"):
        need_mdt_reasons.append("setas de drenagem")
    need_mdt = bool(need_mdt_reasons)

    if need_mdt and not P.get("mdt"):
        msg = "[ERROR] MDT obrigatorio para: " + ", ".join(need_mdt_reasons)
        print(msg)
        logger.error(msg)
        return

    df = None
    if X.get("txt") and P.get("txt"):
        df = ler_txt(P["txt"])
        print(f"[INFO] TXT lido: {len(df)} linhas.")
    elif X.get("txt"):
        print("[WARN] TXT marcado sem arquivo  pulado.")

    dados_path = P.get("dados")
    dados_doc = dados_msp = None
    dados_kwargs: dict = {}
    if dados_path:
        try:
            dados_doc, dados_msp = abrir_dxf_simbologia(dados_path)
            dados_kwargs = {"doc": dados_doc, "msp": dados_msp}
        except Exception as e:
            print(f"[WARN] DXF dados: falha ao abrir {dados_path}: {e}")
            dados_path = None
            dados_doc = dados_msp = None
            dados_kwargs = {}

    lotes = carregar_poligonos_por_layer(dados_path, LYR_DIV_FIS_LOTE, **dados_kwargs) if dados_path else []
    edificacoes = carregar_poligonos_por_layer(dados_path, LYR_EDIF, **dados_kwargs) if dados_path else []
    eixos_via = carregar_linhas_por_layers(dados_path, {LYR_EIXO}, **dados_kwargs) if dados_path else []
    # Bordas de guia: use conjunto do config (ex.: {"VIA"}); fallback aos nomes legados
    guias_layers = set(LAYERS_PISTA_BORDA) if LAYERS_PISTA_BORDA else {LYR_GUIA_COM, LYR_GUIA_SEM}
    guias_via = carregar_linhas_por_layers(dados_path, guias_layers, **dados_kwargs) if dados_path else []
    testadas_abertas = carregar_linhas_por_layers(dados_path, {LYR_DIV_FIS_LOTE}, **dados_kwargs) if dados_path else []
    per_interesse = carregar_poligonos_por_layer(dados_path, LYR_PER_INTERESSE, **dados_kwargs) if dados_path else []
    per_levantamento = carregar_poligonos_por_layer(dados_path, LYR_PER_LEVANT, **dados_kwargs) if dados_path else []

    if dados_path:
        print(
            f"[INFO] DXF dados  lotes={len(lotes)}, edifs={len(edificacoes)}, "
            f"eixos_via={len(eixos_via)}, guias_via={len(guias_via)}, "
            f"per_interesse={len(per_interesse)}, per_levantamento={len(per_levantamento)}"
        )

    per_interesse_union = None
    if per_interesse:
        try:
            per_interesse_union = unary_union(per_interesse)
        except Exception:
            per_interesse_union = per_interesse[0]

    per_levantamento_union = None
    if per_levantamento:
        try:
            per_levantamento_union = unary_union(per_levantamento)
        except Exception:
            per_levantamento_union = per_levantamento[0]

    clip_poly = per_levantamento_union
    if clip_poly is None and per_levantamento:
        clip_poly = per_levantamento[0]

    area_info = {}
    if per_interesse_union is not None:
        try:
            area_info["nucleo"] = float(per_interesse_union.area)
        except Exception as e:
            logger.debug(f"Falha ao calcular area do perimetro de interesse: {e}")
    if per_levantamento_union is not None:
        try:
            area_info["levantamento"] = float(per_levantamento_union.area)
        except Exception as e:
            logger.debug(f"Falha ao calcular area do perimetro de levantamento: {e}")
    setattr(params, "perimetro_area_info", area_info)
    setattr(params, "perimetro_levantamento_geom", per_levantamento_union)
    setattr(params, "perimetro_interesse_geom", per_interesse_union)

    # MDT (quando necessario): rasterio + GDAL (com fechamento garantido)
    mdt_cm = ResourceManager().managed_mdt(P["mdt"]) if P.get("mdt") else nullcontext()
    with mdt_cm as mdt_src:
        get_elevation = None
        mdt_gdal = None
        epsg_auto, epsg_note = None, ""
        if P.get("mdt"):
            try:
                if mdt_src is not None:
                    get_elevation = make_get_elevation_from_src(mdt_src)
                else:
                    mdt_src, get_elevation = make_get_elevation(P["mdt"])
            except Exception as e:
                msg = f"[ERROR] MDT: falha ao abrir {P['mdt']}: {e}"
                print(msg)
                logger.error(msg, exc_info=True)
                return
            try:
                mdt_gdal = mdt_src if hasattr(mdt_src, "GetRasterBand") else gdal.Open(P["mdt"])
            except Exception as e_gdal:
                mdt_gdal = None
                print(f"[WARN] MDT: falha ao carregar via GDAL {P['mdt']}: {e_gdal}")
                logger.warning(f"MDT GDAL falhou: {e_gdal}")
            if mdt_gdal is not None:
                try:
                    epsg_auto, epsg_note = _epsg_from_gdal_dataset(mdt_gdal)
                except Exception:
                    epsg_auto, epsg_note = None, ""

            if get_elevation is None and need_mdt:
                msg = "[ERROR] MDT: nao foi possivel obter funcao de cota. Processo interrompido."
                print(msg)
                logger.error(msg)
                return

        # EPSG local usado apenas para SHP de nomes (quando existir)
        epsg_choice = int(epsg_auto or 31983)
        print(f"[INFO] EPSG local (auto/default): {epsg_choice}")

        # Provider de nomes: tenta SHP local
        name_provider = None
        if X.get("vias") and eixos_via:
            if per_levantamento_union is not None:
                bbox_local = per_levantamento_union.bounds
            elif per_interesse_union is not None:
                bbox_local = per_interesse_union.bounds
            else:
                bbox_local = MultiLineString(eixos_via).bounds

            # 1) tenta SHP local (se existir)
            shp_path = r"D:\2304_REURB_SP\DOCUMENTOS\SHP\SIRGAS_SHP_logradouronbl_line.shp"
            try:
                import os
                if os.path.isfile(shp_path):
                    prov_shp = build_shp_name_provider(shp_path, epsg_choice)
                    if prov_shp:
                        print("[INFO] Nomes de via: usando SHP local")
                        name_provider_shp = lambda x, y: prov_shp.get(x, y, max_dist_m=15.0)
                    else:
                        name_provider_shp = None
                else:
                    name_provider_shp = None
            except Exception as e:
                print(f"[WARN] SHP nomes: falha ao abrir {shp_path}: {e}")
                logger.warning(f"SHP nomes falha: {e}")
                name_provider_shp = None

            name_provider = name_provider_shp

        # === 8) Execucao por blocos ===
        # 7.5) So AREA do lote (quando marcado e TXT desmarcado)
        if X.get("area_lote") and not X.get("txt"):
            try:
                _rotular_areas_lote(ms, lotes, edificacoes, eixos_via, params)
                print(f"[INFO] Areas de lote rotuladas: {len(lotes)} candidatos")
            except Exception as e:
                print(f"[WARN] Falhou ao rotular areas de lote: {e}")

        dimtxt_backup = getattr(params, "dimtxt_ordinate", None)

        setas_inseridas_via_processing = False
        # Linhas candidatas para setas: preferir eixo; se vazio, usar guias
        linhas_setas = eixos_via if (eixos_via and len(eixos_via) > 0) else guias_via
        if X.get("txt") and df is not None:
            t0 = time.perf_counter()
            processar_registros(
                df=df,
                ms=ms,
                doc=doc,
                params=params,
                lotes=lotes,
                edificacoes=edificacoes,
                via_lines_setas=(linhas_setas if X.get("drenagem") else []),
                via_lines_geral=(eixos_via + guias_via),
                get_elevation=get_elevation,
            )
            print(f"[TIME] TXT/processamento: {time.perf_counter() - t0:.2f}s")
            # As setas de drenagem sao inseridas dentro de processing
            setas_inseridas_via_processing = bool(X.get("drenagem"))

        # Caso o usuario tenha marcado drenagem sem TXT (ou TXT nao processado),
        # garantimos que as setas sejam inseridas aqui tambem.
        if X.get("drenagem") and (linhas_setas and len(linhas_setas) > 0) and not setas_inseridas_via_processing:
            try:
                if get_elevation is None:
                    print("[WARN] Drenagem: MDT nao carregado  setas nao geradas.")
                else:
                    t0 = time.perf_counter()
                    inserir_setas_drenagem(ms, doc, linhas_setas, get_elevation, params)
                    print(f"[TIME] Drenagem: {time.perf_counter() - t0:.2f}s")
            except Exception as e:
                print(f"[WARN] Drenagem: falha ao inserir setas: {e}")

        if X.get("perimetros") and (per_levantamento or per_interesse):
            per_src = (settings["exports"].get("per_source") or "").lower()
            polys = None
            if per_src == "levantamento":
                polys = per_levantamento
                print("[INFO] Perimetro: usando PER_LEVANTAMENTO")
            elif per_src == "interesse":
                polys = per_interesse
                print("[INFO] Perimetro: usando PER_INTERESSE")
            elif per_src == "auto":
                if per_levantamento:
                    polys = per_levantamento
                    print("[INFO] Perimetro: usando PER_LEVANTAMENTO (auto)")
                elif per_interesse:
                    polys = per_interesse
                    print("[INFO] Perimetro: usando PER_INTERESSE (auto)")
            else:
                # Fallback automatico se nao especificado ou desconhecido
                if per_interesse:
                    polys = per_interesse
                    print("[INFO] Perimetro: usando PER_INTERESSE (fallback)")
                elif per_levantamento:
                    polys = per_levantamento
                    print("[INFO] Perimetro: usando PER_LEVANTAMENTO (fallback)")
            if polys:
                t0 = time.perf_counter()
                setattr(params, "dimtxt_ordinate", float(getattr(params, "dimtxt_ordinate_perim", 0.5)))
                # desloca rotulos para fora do perimetro selecionado
                per_ref = polys[0] if polys else None
                processar_perimetros(ms, doc, params, polys, per_ref)
                print(f"[TIME] Perimetro: {time.perf_counter() - t0:.2f}s")
            else:
                print("[WARN] Nenhum poligono de perimetro encontrado.")
        elif X.get("perimetros"):
            print("[WARN] Perimetro marcado mas sem DXF de dados  pulado.")

        if X.get("lotes_dim") and lotes:
            try:
                t0 = time.perf_counter()
                processar_lotes_dimensoes(ms, doc, params, lotes)
                print(f"[TIME] Lotes dimensoes: {time.perf_counter() - t0:.2f}s")
            except Exception as e:
                print(f"[WARN] Lotes dimensoes: falha ao gerar: {e}")
        elif X.get("lotes_dim"):
            print("[WARN] Lotes dimensoes marcadas mas sem DXF de dados  pulado.")

        if X.get("vias") and eixos_via and guias_via:
            print(f"[INFO] Vias: iniciando medidas (eixos={len(eixos_via)}, guias={len(guias_via)})")
            setattr(params, "dimtxt_ordinate", float(getattr(params, "dimtxt_ordinate_via", 0.5)))
            try:
                t0 = time.perf_counter()
                medir_e_rotular_vias(
                    ms=ms,
                    eixos=eixos_via,
                    bordas_guia=guias_via,
                    lotes_polygons=lotes,
                    testada_extra_lines=testadas_abertas,
                    texto_altura=params.altura_texto_via,
                    offset_texto_m=params.via_offset_texto,
                    cross_span_m=params.via_cross_span,
                    amostras_fracs=(1 / 3, 2 / 3),
                    name_provider=name_provider,
                    ativar_testada_testada=True,
                    nome_offset_m=float(getattr(params, "via_nome_offset_m", 0.20)),
                    nome_offset_side="auto",
                    nome_sufixo=" (Asfalto)",
                    style_texto=STYLE_TEXTO,
                    dim_text_style=getattr(params, "via_dim_style", "Cota_Rua"),
                    layer_via_medida=perfil["via_med"],
                    layer_via_nome=perfil["via_nome"],
                    sample_mode="entre_intersecoes",
                    offset_lote_lote_extra_m=float(getattr(params, "via_offset_lote_lote_extra_m", 0.60)),
                    dim_gap_m=float(getattr(params, "via_dim_gap_m", 0.60)),
                    dim_min_len_m=float(getattr(params, "via_dim_min_len_m", 12.0)),
                    dim_min_spacing_m=float(getattr(params, "via_dim_min_spacing_m", 25.0)),
                    dim_max_por_trecho=int(getattr(params, "via_dim_max_por_trecho", 2)),
                    dim_max_dist_m=float(getattr(params, "via_dim_max_dist_m", 20.0)),
                    dim_min_sep_area_m=float(getattr(params, "via_dim_min_sep_area_m", 10.0)),
                    dim_equal_tol_m=float(getattr(params, "via_dim_equal_tol_m", 0.05)),
                    nome_offset_add_dim_m=None,
                    nome_side_mode="oposto_dim",
                    nome_shift_along_m=float(getattr(params, "via_nome_shift_along_m", 6.0)),
                    nome_case="upper" if V.get("via_nome_maiusculas", False) else "as_is",
                )
                print(f"[TIME] Vias: {time.perf_counter() - t0:.2f}s")
                print("[INFO] Vias: medidas concluidas")
            except Exception as e:
                print(f"[WARN] Vias: falha ao medir/rotular vias: {e}")
        elif X.get("vias"):
            faltando = []
            if not eixos_via:
                faltando.append(f"eixo de via (layer {LYR_EIXO})")
            if not guias_via:
                try:
                    layers_guias = ", ".join(sorted(guias_layers))
                except Exception:
                    layers_guias = "bordas de guia"
                faltando.append(f"bordas de guia ({layers_guias})")
            faltando_txt = ", ".join(faltando) if faltando else "dados de via"
            print(f"[WARN] Vias marcadas mas faltam dados no DXF: {faltando_txt}. Medidas nao geradas.")

        if X.get("curvas") and P.get("mdt"):
            if mdt_gdal is None:
                print("[WARN] Curvas: MDT nao aberto via GDAL  pulado.")
            else:
                t0 = time.perf_counter()
                gerar_curvas_nivel(ms, mdt_gdal, params, clip_poly=clip_poly)
                print(f"[TIME] Curvas: {time.perf_counter() - t0:.2f}s")
        elif X.get("curvas"):
            print("[WARN] Curvas marcadas mas sem MDT  pulado.")

        if dimtxt_backup is not None:
            setattr(params, "dimtxt_ordinate", dimtxt_backup)

        # Gera tabela de areas sempre que houver perimetros (independente da simbologia)
        try:
            _area_done = getattr(params, "_area_table_done", False)
        except Exception as e:
            logger.debug(f"Falha ao ler flag de tabela de areas: {e}")
            _area_done = False
        if (per_interesse_union is not None or per_levantamento_union is not None) and not _area_done:
            try:
                t0 = time.perf_counter()
                create_area_table(ms, per_interesse_union, per_levantamento_union, params)
                print(f"[TIME] Tabela areas: {time.perf_counter() - t0:.2f}s")
                setattr(params, "_area_table_done", True)
                print("[INFO] Tabela de areas criada")
            except Exception as e:
                print(f"[WARN] Falha ao criar tabela de areas: {e}")

        try:
            doc.purge()
        except Exception as e:
            logger.debug(f"Falha ao executar doc.purge(): {e}")
        path_final = salvar_dxf(doc, P["saida"], f"{nome_area}_BLOCOS.dxf")
        print(f" Arquivo salvo em: {path_final}")

        try:
            if P.get("mdt") and mdt_gdal:
                try:
                    mdt_gdal.FlushCache()
                except Exception as e:
                    logger.debug(f"Falha ao flush do MDT GDAL: {e}")
            mdt_gdal = None
        except Exception as e:
            logger.debug(f"Falha no fechamento do MDT GDAL: {e}")


def main():
    def _run(settings):
        _executar(settings)

    abrir_ui(GLOBAL_PARAMS, on_execute=_run)


if __name__ == "__main__":
    main()
