# run_bloco_reurb.py
from __future__ import annotations

import math
import copy
from shapely.ops import unary_union
from shapely.geometry import MultiLineString

from osgeo import gdal, osr

from config import (
    GLOBAL_PARAMS, STYLE_TEXTO,
    SIMBOLOGIA_DEFAULT_PATH,
    TYPE_TO_LAYER, LAYER_TO_BLOCK,
    LAYER_SOLEIRA_NUM_PAV, LAYER_CURVA_ROTULO,
    LAYER_LOTES, LAYER_EDIF, LAYER_EIXO_VIA, LAYER_PERIMETRO, LAYER_PER_LEVANTAMENTO,
    ROTATION_LAYERS, LAYERS_PISTA_BORDA,
)
try:
    from config import LAYER_SOLEIRA_AREA
except Exception:
    # fallback so para nao quebrar em tempo de execucao se o linter nao achar
    LAYER_SOLEIRA_AREA = "TOP_AREA_LOTE"
from ui import abrir_ui
from txt_utils import ler_txt
from geom_utils import encontrar_rotacao_por_via, encontrar_rotacao_por_lote

from dxf_utils import (
    abrir_dxf_simbologia,
    garantir_estilos_blocos,
    salvar_dxf,
    carregar_poligonos_por_layer,
    carregar_linhas_por_layers,
)

from mdt_utils import make_get_elevation
from processing import processar_registros, inserir_setas_drenagem
from perimetro import processar_perimetros
from curvas_nivel import gerar_curvas_nivel
from vias_medidas import medir_e_rotular_vias
from osm_names import OSMNameProvider
from shp_names import build_shp_name_provider
from symbology_profiles import build_layer_profile
from area_table import create_area_table

# caminhos fixos de simbologia (ver config.py)

# layers de ENTRADA (agora vindo do config)
LYR_EIXO = LAYER_EIXO_VIA
# Se não houver separação entre pistas com/sem guia no seu DXF, estas listas podem ficar vazias.
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
        srs = osr.SpatialReference(); srs.ImportFromWkt(wkt)
        try:
            srs.AutoIdentifyEPSG()
        except Exception:
            pass
        code = srs.GetAuthorityCode("PROJCS") or srs.GetAuthorityCode("GEOGCS")
        return (int(code) if code else None), ("authority" if code else "indeterminado")
    except Exception as e:
        return None, f"erro:{e}"

def _rotular_areas_lote(ms, lotes, edificacoes, vias, params):
    """
    Escreve somente a AREA do lote em LAYER_SOLEIRA_AREA para todos os poligonos de 'lotes'.
    Mesma logica de POSICIONAMENTO do fluxo TXT: base no ponto representativo da maior edificacao (fallback: lote),
    rotacao pela testada do lote (fallback: via mais proxima), texto "em cima" (offset +1*line_gap ao longo da normal).
    """
    def _upright(deg):
        if deg is None:
            return None
        a = deg % 180.0
        return a if a <= 90.0 else a - 180.0
    def _normal_from_rotation(deg):
        if deg is None:
            return (0.0, 1.0)
        rad = math.radians(deg)
        return (-math.sin(rad), math.cos(rad))

    line_gap = max(float(getattr(params, "altura_texto_soleira", 0.5)), 0.50)

    for lote_poly in (lotes or []):
        # texto da area
        try:
            area_txt = f"{float(lote_poly.area):.2f} m2"
        except Exception:
            continue

        # base: maior edificacao (centro representativo)  lote (fallback)
        try:
            e_sel = None
            if edificacoes:
                max_a = -1.0
                for e in edificacoes:
                    try:
                        c = e.representative_point()
                        if not lote_poly.contains(c):
                            continue
                        a = float(e.area)
                        if a > max_a:
                            max_a = a; e_sel = e
                    except Exception:
                        continue
            if e_sel is not None:
                base_x, base_y = e_sel.representative_point().coords[0]
            else:
                base_x, base_y = lote_poly.representative_point().coords[0]
        except Exception:
            continue

        # rotacao: testada do lote  via (fallback)
        rot_txt = None
        try:
            rot_txt = encontrar_rotacao_por_lote((base_x, base_y), [lote_poly], delta=getattr(params,"delta_interp",2.0), raio=12.0)
        except Exception:
            rot_txt = None
        if rot_txt is None:
            try:
                rot_txt = encontrar_rotacao_por_via((base_x, base_y), vias or [], getattr(params, "dist_busca_rot", 50.0), getattr(params, "delta_interp", 2.0))
            except Exception:
                rot_txt = None
        rot_txt = _upright(rot_txt) if rot_txt is not None else None
        nux, nuy = _normal_from_rotation(rot_txt)

        # posicao final: "em cima" da base
        tx_x = float(base_x) + nux * (1.0 * line_gap)
        tx_y = float(base_y) + nuy * (1.0 * line_gap)

        try:
            t = ms.add_text(area_txt, dxfattribs={"height": float(getattr(params, "altura_texto_area", 0.6)),
                                                  "style": STYLE_TEXTO, "layer": LAYER_SOLEIRA_AREA})
            t.dxf.insert = (tx_x, tx_y)
            try:
                t.dxf.halign = 1  # center
                t.dxf.valign = 2  # middle
                t.dxf.align_point = (tx_x, tx_y)
            except Exception:
                pass
            if rot_txt is not None:
                try:
                    t.dxf.rotation = float(rot_txt)
                except Exception:
                    pass
        except Exception:
            continue

def _build_osm_provider(bbox_local, epsg, osm_cfg) -> OSMNameProvider:
    # busca enxuta: buffer 15 m, sem expansao, fallback 15 m
    prov = OSMNameProvider(
        bbox_local=bbox_local,
        epsg_local=int(epsg),
        overpass_url=None,
        user_agent=osm_cfg.get("user_agent", "itesp-reurb"),
        timeout=90,
        inflate_bbox_m=15.0,
        expansion_steps_m=(0,),    # sem expansao
        fallback_around_m=15,      # raio local caso bbox seja vazio
        verbose=False,
    )
    prov.build()
    return prov

def _executar(settings):
    if not settings:
        return

    nome_area = settings["nome_area"]
    P = settings["paths"]
    X = settings["exports"]
    O = settings["osm"]
    T = settings["textos"]
    V = settings["vias"]
    C = settings["curvas"]
    S = settings["setas"]

    # === 2) Simbologia fixa ===
    simb_path = SIMBOLOGIA_DEFAULT_PATH
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
    setattr(params, "via_dim_style", STYLE_TEXTO)
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

    if not P.get("mdt"):
        print("[ERROR] MDT obrigatorio: selecione um arquivo MDT valido para gerar cotas.")
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
        print(f"[INFO] DXF dados  lotes={len(lotes)}, edifs={len(edificacoes)}, "
              f"eixos_via={len(eixos_via)}, guias_via={len(guias_via)}, "
              f"per_interesse={len(per_interesse)}, per_levantamento={len(per_levantamento)}")

    per_interesse_union = None
    if per_interesse:
        try: per_interesse_union = unary_union(per_interesse)
        except Exception: per_interesse_union = per_interesse[0]

    per_levantamento_union = None
    if per_levantamento:
        try: per_levantamento_union = unary_union(per_levantamento)
        except Exception: per_levantamento_union = per_levantamento[0]

    clip_poly = per_levantamento_union
    if clip_poly is None and per_levantamento:
        clip_poly = per_levantamento[0]

    area_info = {}
    if per_interesse_union is not None:
        try: area_info["nucleo"] = float(per_interesse_union.area)
        except Exception:
            pass
    if per_levantamento_union is not None:
        try: area_info["levantamento"] = float(per_levantamento_union.area)
        except Exception:
            pass
    setattr(params, "perimetro_area_info", area_info)
    setattr(params, "perimetro_levantamento_geom", per_levantamento_union)
    setattr(params, "perimetro_interesse_geom", per_interesse_union)
    # MDT (obrigatorio): rasterio + GDAL
    mdt_src, get_elevation = None, None
    mdt_gdal = None
    epsg_auto, epsg_note = None, ""
    try:
        mdt_src, get_elevation = make_get_elevation(P["mdt"])
    except Exception as e:
        print(f"[ERROR] MDT: falha ao abrir {P['mdt']}: {e}")
        return
    try:
        mdt_gdal = mdt_src if hasattr(mdt_src, "GetRasterBand") else gdal.Open(P["mdt"])
    except Exception as e_gdal:
        mdt_gdal = None
        print(f"[WARN] MDT: falha ao carregar via GDAL {P['mdt']}: {e_gdal}")
    if mdt_gdal is not None:
        try:
            epsg_auto, epsg_note = _epsg_from_gdal_dataset(mdt_gdal)
        except Exception:
            epsg_auto, epsg_note = None, ""

    if get_elevation is None:
        print("[ERROR] MDT: nao foi possivel obter funcao de cota. Processo interrompido.")
        return

    # EPSG OSM conforme UI (padrao 31983, sem auto)
    epsg_mode = str(O.get("epsg_mode") or "").lower()
    if epsg_mode in {"31982", "31983"}:
        epsg_choice = int(epsg_mode)
        print(f"[INFO] EPSG local (fixo): {epsg_choice}")
    elif epsg_mode == "custom":
        epsg_choice = int(O.get("epsg_custom") or 31983)
        print(f"[INFO] EPSG local (custom): {epsg_choice}")
    elif epsg_mode == "auto" and epsg_auto:
        epsg_choice = int(epsg_auto)
        print(f"[INFO] EPSG local (auto legado): {epsg_choice} ({epsg_note or 'fallback 31983'})")
    else:
        epsg_choice = 31983
        print(f"[INFO] EPSG local (padrao): {epsg_choice}")

    # Provider de nomes: tenta SHP primeiro; fallback para OSM
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
                    name_provider_shp = lambda x, y: prov_shp.get(x, y, max_dist_m=float(O.get("via_nome_raio", 15.0)))
                else:
                    name_provider_shp = None
            else:
                name_provider_shp = None
        except Exception as e:
            print(f"[WARN] SHP nomes: falha ao abrir {shp_path}: {e}")
            name_provider_shp = None

        # 2) fallback OSM
        name_provider_osm = None
        try:
            prov = _build_osm_provider(bbox_local, epsg_choice, O)
            name_provider_osm = lambda x, y: prov.get(x, y, max_dist_m=float(O.get("via_nome_raio", 15.0)))
        except Exception as e:
            print(f"[WARN] OSM: falhou para EPSG {epsg_choice}: {e}")

        if name_provider_shp and name_provider_osm:
            name_provider = lambda x, y: (name_provider_shp(x, y) or name_provider_osm(x, y))
        else:
            name_provider = name_provider_shp or name_provider_osm

    # === 8) Execucao por blocos ===
    # 7.5) So AREA do lote (quando marcado e TXT desmarcado)
    if X.get('area_lote') and not X.get('txt'):
        try:
            _rotular_areas_lote(ms, lotes, edificacoes, eixos_via, params)
            print(f"[INFO] Areas de lote rotuladas: {len(lotes)} candidatos")
        except Exception as e:
            print(f"[WARN] Falhou ao rotular areas de lote: {e}")

    try:
        dimtxt_backup = getattr(params, "dimtxt_ordinate", None)

        setas_inseridas_via_processing = False
        # Linhas candidatas para setas: preferir eixo; se vazio, usar guias
        linhas_setas = eixos_via if (eixos_via and len(eixos_via) > 0) else guias_via
        if X.get("txt") and df is not None:
            processar_registros(
                df=df, ms=ms, doc=doc, params=params,
                lotes=lotes, edificacoes=edificacoes,
                via_lines_setas=(linhas_setas if X.get("drenagem") else []),
                via_lines_geral=(eixos_via + guias_via),
                get_elevation=get_elevation,
            )
            # As setas de drenagem são inseridas dentro de processing
            setas_inseridas_via_processing = bool(X.get("drenagem"))

        # Caso o usuário tenha marcado drenagem sem TXT (ou TXT não processado),
        # garantimos que as setas sejam inseridas aqui também.
        if X.get("drenagem") and (linhas_setas and len(linhas_setas) > 0) and not setas_inseridas_via_processing:
            try:
                inserir_setas_drenagem(ms, doc, linhas_setas, get_elevation, params)
            except Exception as e:
                print(f"[WARN] Drenagem: falha ao inserir setas: {e}")

        if X.get("perimetros") and (per_levantamento or per_interesse):
            per_src = (settings["exports"].get("per_source") or "").lower()
            polys = None
            if per_src == "levantamento":
                polys = per_levantamento; print("[INFO] Perimetro: usando PER_LEVANTAMENTO")
            elif per_src == "interesse":
                polys = per_interesse; print("[INFO] Perimetro: usando PER_INTERESSE")
            elif per_src == "auto":
                if per_levantamento:
                    polys = per_levantamento; print("[INFO] Perimetro: usando PER_LEVANTAMENTO (auto)")
                elif per_interesse:
                    polys = per_interesse; print("[INFO] Perimetro: usando PER_INTERESSE (auto)")
            else:
                # Fallback automático se não especificado ou desconhecido
                if per_interesse:
                    polys = per_interesse; print("[INFO] Perimetro: usando PER_INTERESSE (fallback)")
                elif per_levantamento:
                    polys = per_levantamento; print("[INFO] Perimetro: usando PER_LEVANTAMENTO (fallback)")
            if polys:
                setattr(params, "dimtxt_ordinate", float(getattr(params, "dimtxt_ordinate_perim", 0.5)))
                processar_perimetros(ms, doc, params, polys, per_levantamento_union)
            else:
                print("[WARN] Nenhum poligono de perimetro encontrado.")
        elif X.get("perimetros"):
            print("[WARN] Perimetro marcado mas sem DXF de dados  pulado.")

        if X.get("vias") and eixos_via and guias_via:
            setattr(params, "dimtxt_ordinate", float(getattr(params, "dimtxt_ordinate_via", 0.5)))
            medir_e_rotular_vias(
                ms=ms,
                eixos=eixos_via,
                bordas_guia=guias_via,
                lotes_polygons=lotes,
                testada_extra_lines=testadas_abertas,
                texto_altura=params.altura_texto_via,
                offset_texto_m=params.via_offset_texto,
                cross_span_m=params.via_cross_span,
                amostras_fracs=(1/3, 2/3),
                name_provider=name_provider,
                ativar_testada_testada=True,
                nome_offset_m=float(getattr(params, "via_nome_offset_m", 0.60)),
                nome_offset_side=settings["vias"].get("via_nome_offset_side", "auto"),
                nome_sufixo=settings["vias"].get("via_nome_sufixo", " (Asfalto)"),
                style_texto=STYLE_TEXTO,
                layer_via_medida=perfil["via_med"],
                layer_via_nome=perfil["via_nome"],
                sample_mode="entre_intersecoes",
                offset_lote_lote_extra_m=float(getattr(params, "via_offset_lote_lote_extra_m", 0.60)),
                nome_offset_add_dim_m=None,
                nome_side_mode="oposto_dim",
                nome_shift_along_m=float(getattr(params, "via_nome_shift_along_m", 6.0)),
                nome_case="upper" if V.get("via_nome_maiusculas", False) else "as_is",
            )
        elif X.get("vias"):
            print("[WARN] Vias marcadas mas faltam eixos/guias  pulado.")

        # Drenagem: agora e feita dentro do processing (nao duplicar aqui)

        if X.get("curvas") and P.get("mdt"):
            if mdt_gdal is None:
                print("[WARN] Curvas: MDT nao aberto via GDAL  pulado.")
            else:
                gerar_curvas_nivel(ms, mdt_gdal, params, clip_poly=clip_poly)
        elif X.get("curvas"):
            print("[WARN] Curvas marcadas mas sem MDT  pulado.")

        if dimtxt_backup is not None:
            setattr(params, "dimtxt_ordinate", dimtxt_backup)

        # Gera tabela de áreas sempre que houver perímetros (independente da simbologia)
        try:
            _area_done = getattr(params, "_area_table_done", False)
        except Exception:
            _area_done = False
        if (per_interesse_union is not None or per_levantamento_union is not None) and not _area_done:
            try:
                create_area_table(ms, per_interesse_union, per_levantamento_union, params)
                setattr(params, "_area_table_done", True)
                print("[INFO] Tabela de áreas criada")
            except Exception as e:
                print(f"[WARN] Falha ao criar tabela de áreas: {e}")

        # Tabela de áreas (modo REURB)
            try:
                create_area_table(ms, per_interesse_union, per_levantamento_union, params)
                print("[INFO] Tabela de áreas criada")
            except Exception as e:
                print(f"[WARN] Falha ao criar tabela de áreas: {e}")

        try: doc.purge()
        except Exception: pass
        path_final = salvar_dxf(doc, P["saida"], f"{nome_area}_BLOCOS.dxf")
        print(f" Arquivo salvo em: {path_final}")

    finally:
        try:
            if P.get("mdt") and mdt_gdal:
                try: mdt_gdal.FlushCache()
                except Exception: pass
                mdt_gdal = None
        except Exception:
            pass
        try:
            if P.get("mdt") and mdt_src and hasattr(mdt_src, "close"):
                mdt_src.close()
        except Exception:
            pass

def main():
    def _run(settings):
        _executar(settings)
    abrir_ui(GLOBAL_PARAMS, on_execute=_run)

if __name__ == "__main__":
    main()





