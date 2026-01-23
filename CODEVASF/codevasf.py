import os
import geopandas as gpd
import pandas as pd

try:
    from shapely import make_valid, set_precision
except Exception:
    make_valid = None
    set_precision = None

# ========= AJUSTE AQUI =========
shp_propr = r"G:\4 - PROCESSAMENTO\11 VETORIZAÇÃO\apoio\VETOR\CODEVASF\PROPRIEDADES.shp"
shp_uso   = r"G:\4 - PROCESSAMENTO\11 VETORIZAÇÃO\apoio\VETOR\CODEVASF\uso_solo.shp"
saida_xlsx = r"G:\4 - PROCESSAMENTO\11 VETORIZAÇÃO\apoio\VETOR\CODEVASF\relatorio_uso_por_propriedade.xlsx"

crs_area = "EPSG:31984"
TOL_M2 = 0.5  # ignora diferenças menores que 0,5 m² (gap/overlap)
TOL_PCT = 0.0001  # 0,01% da area da propriedade (tolerancia proporcional)
GRID_SNAP_M = 0.01  # snap de coordenadas (metros); use 0 para desativar
SLIVER_M2 = 0.5  # remove slivers (poligonos muito pequenos) apos overlay
# ===============================

def log(msg):
    print(msg, flush=True)

def _set_precision_safe(geom):
    if set_precision is None or GRID_SNAP_M <= 0:
        return geom
    try:
        return set_precision(geom, GRID_SNAP_M)
    except Exception:
        return geom

def _make_valid_safe(geom):
    if make_valid is None:
        return geom
    try:
        return make_valid(geom)
    except Exception:
        return geom

def garantir_poligonos(gdf, nome):
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]

    # tenta corrigir geometrias inválidas
    try:
        gdf["geometry"] = gdf.geometry.buffer(0)
    except Exception:
        pass
    # make_valid quando disponivel
    gdf["geometry"] = gdf.geometry.apply(_make_valid_safe)
    # snap/precisao quando disponivel
    gdf["geometry"] = gdf.geometry.apply(_set_precision_safe)

    gdf = gdf[gdf.geometry.notna()]
    if gdf.empty:
        raise ValueError(f"{nome}: não há geometrias poligonais válidas.")
    return gdf

def fechar_por_propriedade(propr_diss, uso):
    registros = []
    checks = []
    overlaps = []
    m2_to_ha = 1.0 / 10000.0

    uso_sidx = uso.sindex  # acelera

    for _, row in propr_diss.iterrows():
        nm = row["nm_propr"]
        geom_prop = row.geometry
        area_total = float(geom_prop.area)

        # candidatos por bbox
        cand_idx = list(uso_sidx.intersection(geom_prop.bounds))
        uso_cand = uso.iloc[cand_idx]
        uso_clip = uso_cand[uso_cand.intersects(geom_prop)].copy()

        if uso_clip.empty:
            # sem uso -> tudo SEM_USO (fecha)
            registros.append([nm, area_total, "SEM_USO", "SEM_USO", "SEM_USO", area_total])
            checks.append([nm, area_total, 0.0, area_total, 0.0])
            continue

        # overlay (intersection)
        prop_gdf = gpd.GeoDataFrame({"nm_propr": [nm]}, geometry=[geom_prop], crs=propr_diss.crs)
        try:
            uso_int = gpd.overlay(
                prop_gdf,
                uso_clip[["USO", "TIPO", "CLASSE", "geometry"]],
                how="intersection",
                keep_geom_type=False
            )
        except Exception:
            # fallback manual
            uso_int = uso_clip.copy()
            uso_int["geometry"] = uso_int.geometry.intersection(geom_prop)
            uso_int = uso_int[uso_int.geometry.notna() & ~uso_int.geometry.is_empty]

        if uso_int.empty:
            registros.append([nm, area_total, "SEM_USO", "SEM_USO", "SEM_USO", area_total])
            checks.append([nm, area_total, 0.0, area_total, 0.0])
            continue

        # remove slivers apos overlay
        uso_int = uso_int[uso_int.geometry.area >= SLIVER_M2].copy()
        if uso_int.empty:
            registros.append([nm, area_total, "SEM_USO", "SEM_USO", "SEM_USO", area_total])
            checks.append([nm, area_total, 0.0, area_total, 0.0])
            continue

        uso_int["area_uso_m2"] = uso_int.geometry.area

        # agrupar por classe
        agg = (uso_int.groupby(["USO", "TIPO", "CLASSE"], dropna=False)["area_uso_m2"]
                     .sum()
                     .reset_index())

        soma_usos = float(agg["area_uso_m2"].sum())

        # gap: propriedade - união dos usos
        try:
            uniao = uso_int.unary_union
            resto = geom_prop.difference(uniao)
            area_resto = float(resto.area) if (resto and not resto.is_empty) else 0.0
            area_uniao = float(uniao.area) if (uniao and not uniao.is_empty) else 0.0
        except Exception:
            area_resto = max(area_total - soma_usos, 0.0)
            area_uniao = max(soma_usos - area_resto, 0.0)

        excesso = soma_usos - area_total
        area_overlap = max(soma_usos - area_uniao, 0.0)

        # normaliza diferenças pequenas (<= 0,5 m²) pra "zero"
        tol_dyn = max(TOL_M2, area_total * TOL_PCT)
        if abs(area_resto) <= tol_dyn:
            area_resto = 0.0
        if abs(excesso) <= tol_dyn:
            excesso = 0.0

        checks.append([nm, area_total, soma_usos, area_resto, excesso, area_overlap])

        # linhas do detalhe com area_total ao lado
        for _, r in agg.iterrows():
            registros.append([nm, area_total, r["USO"], r["TIPO"], r["CLASSE"], float(r["area_uso_m2"])])

        # adiciona SEM_USO apenas se faltar > 0,5 m²
        if area_resto > tol_dyn:
            registros.append([nm, area_total, "SEM_USO", "SEM_USO", "SEM_USO", area_resto])

        # overlap apenas se passar > 0,5 m²
        if area_overlap > tol_dyn:
            overlaps.append([nm, area_total, soma_usos, area_overlap])

    detalhe = pd.DataFrame(
        registros,
        columns=["nm_propr", "area_total_m2", "USO", "TIPO", "CLASSE", "area_uso_m2"]
    )
    check = pd.DataFrame(
        checks,
        columns=["nm_propr", "area_total_m2", "soma_usos_m2", "area_sem_uso_m2", "excesso_m2", "overlap_m2"]
    )
    overlap = pd.DataFrame(
        overlaps,
        columns=["nm_propr", "area_total_m2", "soma_usos_m2", "overlap_m2"]
    )

    # converter para hectares na saÃ­da
    detalhe["area_total_m2"] = detalhe["area_total_m2"] * m2_to_ha
    detalhe["area_uso_m2"] = detalhe["area_uso_m2"] * m2_to_ha
    detalhe = detalhe.rename(columns={"area_total_m2": "area_total_ha", "area_uso_m2": "area_uso_ha"})

    check["area_total_m2"] = check["area_total_m2"] * m2_to_ha
    check["soma_usos_m2"] = check["soma_usos_m2"] * m2_to_ha
    check["area_sem_uso_m2"] = check["area_sem_uso_m2"] * m2_to_ha
    check["excesso_m2"] = check["excesso_m2"] * m2_to_ha
    check["overlap_m2"] = check["overlap_m2"] * m2_to_ha
    check = check.rename(columns={
        "area_total_m2": "area_total_ha",
        "soma_usos_m2": "soma_usos_ha",
        "area_sem_uso_m2": "area_sem_uso_ha",
        "excesso_m2": "excesso_ha",
        "overlap_m2": "overlap_ha",
    })

    if not overlap.empty:
        overlap["area_total_m2"] = overlap["area_total_m2"] * m2_to_ha
        overlap["soma_usos_m2"] = overlap["soma_usos_m2"] * m2_to_ha
        overlap["overlap_m2"] = overlap["overlap_m2"] * m2_to_ha
        overlap = overlap.rename(columns={
            "area_total_m2": "area_total_ha",
            "soma_usos_m2": "soma_usos_ha",
            "overlap_m2": "overlap_ha",
        })
    return detalhe, check, overlap

def main():
    log("Lendo shapefiles...")
    propr = gpd.read_file(shp_propr)
    uso = gpd.read_file(shp_uso)

    if "nm_propr" not in propr.columns:
        raise ValueError("Campo 'nm_propr' não existe em PROPRIEDADES.")
    for col in ["USO", "TIPO", "CLASSE"]:
        if col not in uso.columns:
            raise ValueError(f"Campo '{col}' não existe em uso_solo.")

    propr = garantir_poligonos(propr, "Propriedades")
    uso = garantir_poligonos(uso, "Uso do solo")

    log(f"CRS propriedades: {propr.crs}")
    log(f"CRS uso_solo:     {uso.crs}")

    log("Reprojetando para CRS de área...")
    propr = propr.to_crs(crs_area)
    uso = uso.to_crs(crs_area)
    # snap em metros (se habilitado)
    propr["geometry"] = propr.geometry.apply(_set_precision_safe)
    uso["geometry"] = uso.geometry.apply(_set_precision_safe)

    log("Dissolvendo propriedades por nm_propr...")
    propr_diss = propr.dissolve(by="nm_propr", as_index=False)

    log(f"Total de propriedades (dissolvidas): {len(propr_diss)}")

    log("Calculando tabela de uso por propriedade (pode demorar)...")
    detalhe, check, overlap = fechar_por_propriedade(propr_diss, uso)

    # remove linhas com area_uso_ha zerada antes de exportar
    detalhe = detalhe[detalhe["area_uso_ha"] > 0].copy()
    detalhe = detalhe.sort_values(["nm_propr", "USO", "TIPO", "CLASSE"])

    pasta_saida = os.path.dirname(saida_xlsx)
    os.makedirs(pasta_saida, exist_ok=True)

    log("Salvando Excel...")
    with pd.ExcelWriter(saida_xlsx, engine="openpyxl") as writer:
        detalhe.to_excel(writer, index=False, sheet_name="TABELA_USO_ha")
        check.sort_values("nm_propr").to_excel(writer, index=False, sheet_name="CHECK")
        if not overlap.empty:
            overlap.sort_values("nm_propr").to_excel(writer, index=False, sheet_name="OVERLAP")

    log("OK! Gerado:")
    log(saida_xlsx)
    if not overlap.empty:
        log(f"ATENÇÃO: {len(overlap)} propriedades com overlap (overlap_ha > {TOL_M2/10000.0}). Veja aba OVERLAP.")

if __name__ == "__main__":
    main()
