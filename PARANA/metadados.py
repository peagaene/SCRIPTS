#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lê GeoTIFFs em uma pasta e gera arquivos TXT de metadados.

- Interface gráfica com PySimpleGUI.
- Um TXT por TIFF, com mesmo nome (apenas troca a extensão).
- Mais robusto: não quebra se faltar projeção ou algum metadado.
"""

import os
import time
import datetime

import PySimpleGUI as sg
from osgeo import gdal, osr

gdal.UseExceptions()


def dec_to_dms_str(value, is_lon=True):
    """
    Converte graus decimais para string em DMS no formato:
    53° 33' 48.2908" W
    """
    hemi = ""
    if is_lon:
        hemi = "E" if value >= 0 else "W"
    else:
        hemi = "N" if value >= 0 else "S"

    v = abs(value)
    d = int(v)
    m_float = (v - d) * 60.0
    m = int(m_float)
    s = (m_float - m) * 60.0

    return f'{d}° {m:02d}\' {s:07.4f}" {hemi}'


def guess_sample_type(gdal_dtype):
    from osgeo import gdal as _gdal

    if gdal_dtype == _gdal.GDT_Byte:
        return "Unsigned 8-bit Integer"
    elif gdal_dtype == _gdal.GDT_UInt16:
        return "Unsigned 16-bit Integer"
    elif gdal_dtype == _gdal.GDT_Int16:
        return "Signed 16-bit Integer"
    elif gdal_dtype == _gdal.GDT_UInt32:
        return "Unsigned 32-bit Integer"
    elif gdal_dtype == _gdal.GDT_Int32:
        return "Signed 32-bit Integer"
    elif gdal_dtype in (_gdal.GDT_Float32, _gdal.GDT_Float64):
        return "Floating Point"
    else:
        return "Unknown Format (0)"


def get_proj_desc_safe(srs):
    """
    Monta uma string tipo 'UTM Zone -22 / SIRGAS / meters'
    mas sem explodir se algo der errado.
    """
    try:
        zone = srs.GetUTMZone()
    except Exception:
        zone = 0

    try:
        units_name = srs.GetLinearUnitsName() or "meters"
    except Exception:
        units_name = "meters"

    try:
        datum = (srs.GetAttrValue("DATUM") or "").replace("_", " ").strip()
    except Exception:
        datum = "Unknown"

    if zone != 0:
        return f"UTM Zone {zone if zone < 0 else '-' + str(zone)} / {datum.split()[0] or datum} / {units_name}"
    else:
        try:
            projcs = srs.GetAttrValue("PROJCS") or "Unknown"
        except Exception:
            projcs = "Unknown"
        return f"{projcs} / {units_name}"


def detect_photometric(info_json, band_count):
    md_image = info_json.get("metadata", {}).get("IMAGE_STRUCTURE", {})
    photometric = md_image.get("PHOTOMETRIC")

    if photometric:
        up = photometric.upper()
        if "RGB" in up:
            return "RGB Full-Color"
        if "PALETTE" in up:
            return "Palette Color"
        if "MINISBLACK" in up or "MINISWHITE" in up:
            return "Grayscale"
        return photometric

    if band_count == 3:
        return "RGB Full-Color"
    elif band_count == 1:
        return "Grayscale"

    return "Unknown"


def flatten_metadata(md, prefix="META"):
    """
    Converte um dicionario de metadata (possivelmente aninhado) em linhas chave=valor.
    Mantemos o prefixo para evitar conflito com os campos principais.
    """
    lines = []

    def _walk(obj, path):
        if isinstance(obj, dict):
            for k, v in sorted(obj.items()):
                _walk(v, path + [str(k)])
        elif isinstance(obj, list):
            for idx, v in enumerate(obj, start=1):
                _walk(v, path + [str(idx)])
        else:
            key = "_".join(path)
            lines.append(f"{prefix}_{key}={obj}")

    _walk(md, [])
    return lines


def build_metadata_text(tif_path):
    """
    Lê o GeoTIFF e retorna uma string com os metadados
    no formato desejado, com tratamento melhor de projeção
    e campos faltantes.
    """
    start_time = time.time()

    # --- ABERTURA DO ARQUIVO ---
    ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError("GDAL retornou None ao abrir o arquivo (driver não reconhece ou arquivo realmente corrompido).")

    load_time = time.time() - start_time

    # --- gdal.Info em JSON ---
    try:
        info_json = gdal.Info(tif_path, options=gdal.InfoOptions(format="json"))
        if info_json is None:
            info_json = {}
    except Exception:
        info_json = {}

    driver_short = ds.GetDriver().ShortName if ds.GetDriver() else "Unknown"
    driver_long = ds.GetDriver().LongName if ds.GetDriver() else "Unknown"

    cols = ds.RasterXSize
    rows = ds.RasterYSize
    bands = ds.RasterCount
    gt = ds.GetGeoTransform()

    origin_x = gt[0]
    pixel_w = gt[1]
    rot1 = gt[2]
    origin_y = gt[3]
    rot2 = gt[4]
    pixel_h = gt[5]

    # Cantos em coordenadas projetadas
    ulx = origin_x
    uly = origin_y
    lrx = origin_x + cols * pixel_w
    lry = origin_y + rows * pixel_h

    # Ajuste para garantir upper > lower
    if lry > uly:
        uly, lry = lry, uly

    # Área em km²
    area_m2 = abs(pixel_w * pixel_h * cols * rows)
    area_km2 = area_m2 / 1e6

    # ---------- PROJEÇÃO (sem lat/long ainda) ----------
    proj_desc = "Unknown"
    proj_datum = "Unknown"
    proj_units = "Unknown"
    epsg_str = "EPSG:Unknown"
    pcs_citation = "Unknown"
    geog_citation = "Unknown"
    srs = None

    proj_wkt = ds.GetProjection()
    if proj_wkt and proj_wkt.strip():
        try:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(proj_wkt)

            # Descrição amigável
            proj_desc = get_proj_desc_safe(srs)

            # Datum e unidades
            try:
                proj_datum = (srs.GetAttrValue("DATUM") or "Unknown").replace("_", " ")
            except Exception:
                proj_datum = "Unknown"

            try:
                proj_units = srs.GetLinearUnitsName() or "Unknown"
            except Exception:
                proj_units = "Unknown"

            # EPSG
            try:
                srs_clone = srs.Clone()
                srs_clone.AutoIdentifyEPSG()
                epsg = srs_clone.GetAuthorityCode("PROJCS") or srs_clone.GetAuthorityCode(None)
                if epsg:
                    epsg_str = f"EPSG:{epsg}"
            except Exception:
                pass

            # PCS_CITATION (nome completo da projeção)
            pcs_name = srs.GetAttrValue("PROJCS") or ""
            if pcs_name:
                pcs_citation = pcs_name

            # GEOG_CITATION (nome do sistema geográfico)
            try:
                srs_geo = srs.CloneGeogCS()
                geog_name = srs_geo.GetAttrValue("GEOGCS") or srs_geo.GetAttrValue("DATUM") or ""
                if geog_name:
                    geog_citation = geog_name.replace("_", " ")
            except Exception:
                pass

        except Exception:
            # Se algo falhar na leitura da projeção, deixa como Unknown
            srs = None

    # ---------- LAT/LONG A PARTIR DA PROJEÇÃO ----------
    # Defaults (caso não dê para transformar)
    west_lon = east_lon = north_lat = south_lat = 0.0
    ul_lon = ul_lat = ur_lon = ur_lat = lr_lon = lr_lat = ll_lon = ll_lat = 0.0

    if srs is not None:
        try:
            srs_geo = srs.CloneGeogCS()
            ct = osr.CoordinateTransformation(srs, srs_geo)

            def proj_to_geo(x, y):
                lon, lat, _ = ct.TransformPoint(x, y)
                return lon, lat

            urx = lrx
            ury = uly
            llx = ulx
            lly = lry

            ul_lon, ul_lat = proj_to_geo(ulx, uly)
            ur_lon, ur_lat = proj_to_geo(urx, ury)
            lr_lon, lr_lat = proj_to_geo(lrx, lry)
            ll_lon, ll_lat = proj_to_geo(llx, lly)

            all_lons = [ul_lon, ur_lon, lr_lon, ll_lon]
            all_lats = [ul_lat, ur_lat, lr_lat, ll_lat]

            west_lon = min(all_lons)
            east_lon = max(all_lons)
            north_lat = max(all_lats)
            south_lat = min(all_lats)

        except Exception:
            # Se a transformação der erro (problema de PROJ, etc.),
            # mantemos 0 / Unknown para lat/long, mas não perdemos EPSG, datum, etc.
            pass

    # ---------- BAND / TIPO / COMPRESSION / STRIPS ----------
    band1 = ds.GetRasterBand(1)
    gdal_dtype = band1.DataType
    bits_per_sample = gdal.GetDataTypeSize(gdal_dtype)
    bit_depth_total = bits_per_sample * bands
    sample_type = guess_sample_type(gdal_dtype)

    # SAMPLE_FORMAT (Unsigned Integer, Signed Integer, Floating Point)
    from osgeo import gdal as _gdal
    if gdal_dtype in (_gdal.GDT_Byte, _gdal.GDT_UInt16, _gdal.GDT_UInt32):
        sample_format = "Unsigned Integer"
    elif gdal_dtype in (_gdal.GDT_Int16, _gdal.GDT_Int32):
        sample_format = "Signed Integer"
    elif gdal_dtype in (_gdal.GDT_Float32, _gdal.GDT_Float64):
        sample_format = "Floating Point"
    else:
        sample_format = "Unknown"

    md_all = info_json.get("metadata", {})
    md_tiff = md_all.get("TIFF", {})
    # Alguns GDAL jogam GeoTIFF direto em metadata raiz, outros em metadata["GeoTIFF"]
    md_geotiff_block = md_all.get("GeoTIFF", {})
    md_geotiff_flat = {k: v for k, v in md_all.items() if k.startswith("GeoTIFF::")}
    md_geotiff = {}
    md_geotiff.update(md_geotiff_block)
    md_geotiff.update(md_geotiff_flat)

    # ROWS_PER_STRIP
    rows_per_strip = md_tiff.get("ROWS_PER_STRIP")
    if not rows_per_strip:
        # fallback: block size
        bx, by = band1.GetBlockSize()
        if by > 0:
            rows_per_strip = by
        else:
            rows_per_strip = "Unknown"

    # COMPRESSION
    compression = md_tiff.get("COMPRESSION")
    if not compression:
        md_img_struct = info_json.get("metadata", {}).get("IMAGE_STRUCTURE", {})
        compression = md_img_struct.get("COMPRESSION", "None")
    if not compression:
        compression = "Unknown"

    pixel_scale_str = f"( {abs(pixel_w)}, {abs(pixel_h)}, 1.0 )"
    tiepoints_str = f"( 0.00, 0.00, 0.00 ) --> ( {ulx}, {uly}, 0.000 )"

    model_type = md_geotiff.get("GeoTIFF::ModelTypeGeoKey", "Projection Coordinate System")
    raster_type = md_geotiff.get("GeoTIFF::RasterTypeGeoKey", "Pixel is Area")

    photometric = detect_photometric(info_json, bands)

    # Overviews
    overview_lines = []
    band_json_list = info_json.get("bands", [])
    if band_json_list:
        ovs = band_json_list[0].get("overviews", [])
        for idx, ov in enumerate(ovs, start=1):
            size = ov.get("size", [])
            if len(size) == 2:
                o_cols, o_rows = size
                overview_lines.append(
                    f"OVERVIEW {idx}=Pixel Size: {o_cols} x {o_rows}"
                )

    # GeoTIFF keys
    geokey_lines = []
    for key in [
        "GeoTIFF::ProjLinearUnitsGeoKey",
        "GeoTIFF::ProjectedCSTypeGeoKey",
        "GeoTIFF::GeographicTypeGeoKey",
        "GeoTIFF::GeogSemiMajorAxisGeoKey",
        "GeoTIFF::GeogSemiMinorAxisGeoKey",
        "GeoTIFF::GeogEllipsoidGeoKey",
        "GeoTIFF::GeogToWGS84GeoKey",
    ]:
        if key in md_geotiff:
            geokey_lines.append(f"{key}={md_geotiff[key]}")

    # Tempos de criação/modificação
    try:
        ctime = datetime.datetime.fromtimestamp(os.path.getctime(tif_path))
        ctime_str = ctime.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        ctime_str = "Unknown"

    try:
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(tif_path))
        mtime_str = mtime.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        mtime_str = "Unknown"

    color_bands = ",".join(str(i) for i in range(bands))
    load_time_str = f"{load_time:.2f} s"

    # ---------- MONTA TEXTO ----------
    lines = []
    lines.append(f"FILENAME={os.path.abspath(tif_path)}")
    lines.append(f"DESCRIPTION={os.path.basename(tif_path)}")
    lines.append(f"DRIVER={driver_short} ({driver_long})")
    lines.append(f"UPPER LEFT X={ulx}")
    lines.append(f"UPPER LEFT Y={uly}")
    lines.append(f"LOWER RIGHT X={lrx}")
    lines.append(f"LOWER RIGHT Y={lry}")
    lines.append(f"WEST LONGITUDE={dec_to_dms_str(west_lon, is_lon=True)}")
    lines.append(f"NORTH LATITUDE={dec_to_dms_str(north_lat, is_lon=False)}")
    lines.append(f"EAST LONGITUDE={dec_to_dms_str(east_lon, is_lon=True)}")
    lines.append(f"SOUTH LATITUDE={dec_to_dms_str(south_lat, is_lon=False)}")
    lines.append(f"UL CORNER LONGITUDE={dec_to_dms_str(ul_lon, is_lon=True)}")
    lines.append(f"UL CORNER LATITUDE={dec_to_dms_str(ul_lat, is_lon=False)}")
    lines.append(f"UR CORNER LONGITUDE={dec_to_dms_str(ur_lon, is_lon=True)}")
    lines.append(f"UR CORNER LATITUDE={dec_to_dms_str(ur_lat, is_lon=False)}")
    lines.append(f"LR CORNER LONGITUDE={dec_to_dms_str(lr_lon, is_lon=True)}")
    lines.append(f"LR CORNER LATITUDE={dec_to_dms_str(lr_lat, is_lon=False)}")
    lines.append(f"LL CORNER LONGITUDE={dec_to_dms_str(ll_lon, is_lon=True)}")
    lines.append(f"LL CORNER LATITUDE={dec_to_dms_str(ll_lat, is_lon=False)}")
    lines.append(f"PROJ_DESC={proj_desc}")
    lines.append(f"PROJ_DATUM={proj_datum}")
    lines.append(f"PROJ_UNITS={proj_units}")
    lines.append(f"EPSG_CODE={epsg_str}")
    lines.append(f"BBOX AREA={area_km2:.3f} sq km")
    lines.append(f"FILE_CREATION_TIME={ctime_str}")
    lines.append(f"FILE_MODIFIED_TIME={mtime_str}")
    lines.append(f"NUM COLUMNS={cols}")
    lines.append(f"NUM ROWS={rows}")
    lines.append(f"NUM BANDS={bands}")
    lines.append(f"COLOR BANDS={color_bands}")
    lines.append(f"PIXEL WIDTH={abs(pixel_w)} meters")
    lines.append(f"PIXEL HEIGHT={abs(pixel_h)} meters")
    lines.append(f"BIT DEPTH={bit_depth_total}")
    lines.append(f"SAMPLE TYPE={sample_type}")
    lines.append(f"GT_CITATION={pcs_citation}")       # ex.: SIRGAS 2000 / UTM zone 22S
    lines.append(f"GEOG_CITATION={geog_citation}")     # ex.: SIRGAS 2000
    lines.append(f"PHOTOMETRIC={photometric}")
    lines.append(f"SAMPLE_FORMAT={sample_format}")
    lines.append(f"ROWS_PER_STRIP={rows_per_strip}")
    lines.append(f"COMPRESSION={compression}")
    lines.append(f"PIXEL_SCALE={pixel_scale_str}")
    lines.append(f"TIEPOINTS={tiepoints_str}")
    lines.append(f"MODEL_TYPE={model_type if isinstance(model_type, str) else 'Projection Coordinate System'}")
    lines.append(f"RASTER_TYPE={raster_type if isinstance(raster_type, str) else 'Pixel is Area'}")

    # Band info detalhada (noData, color interpretation, estatisticas e blocos)
    for idx, bjson in enumerate(band_json_list, start=1):
        nodata_val = bjson.get("noDataValue")
        color_interp = bjson.get("colorInterpretation") or "Unknown"
        offset = bjson.get("offset")
        scale = bjson.get("scale")
        unit = bjson.get("unit")
        block = bjson.get("block", [])
        minimum = bjson.get("minimum")
        maximum = bjson.get("maximum")
        mean = bjson.get("mean")
        stddev = bjson.get("stdDev")

        lines.append(f"BAND{idx}_COLOR_INTERP={color_interp}")
        lines.append(f"BAND{idx}_NODATA={nodata_val if nodata_val is not None else 'None'}")
        lines.append(f"BAND{idx}_OFFSET={offset if offset is not None else 0}")
        lines.append(f"BAND{idx}_SCALE={scale if scale is not None else 1}")
        if unit:
            lines.append(f"BAND{idx}_UNIT={unit}")
        if block:
            lines.append(f"BAND{idx}_BLOCKSIZE={'x'.join(str(v) for v in block)}")
        if minimum is not None:
            lines.append(f"BAND{idx}_MIN={minimum}")
        if maximum is not None:
            lines.append(f"BAND{idx}_MAX={maximum}")
        if mean is not None:
            lines.append(f"BAND{idx}_MEAN={mean}")
        if stddev is not None:
            lines.append(f"BAND{idx}_STDDEV={stddev}")

        band_md = bjson.get("metadata", {})
        if band_md:
            lines.extend(flatten_metadata(band_md, prefix=f"BAND{idx}_META"))

    lines.extend(overview_lines)
    lines.extend(geokey_lines)

    # Metadata do dataset inteiro (todos os blocos).
    if md_all:
        lines.extend(flatten_metadata(md_all, prefix="DATASET_META"))

    return "\n".join(lines)

def process_folder(input_folder, output_folder, window=None):
    count_ok = 0
    count_err = 0

    for fname in sorted(os.listdir(input_folder)):
        if not fname.lower().endswith((".tif", ".tiff")):
            continue

        tif_path = os.path.join(input_folder, fname)
        base_name = os.path.splitext(fname)[0]
        out_txt = os.path.join(output_folder, base_name + ".txt")

        try:
            meta_text = build_metadata_text(tif_path)
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(meta_text)
            count_ok += 1
            msg = f"[OK] {fname} -> {os.path.basename(out_txt)}"
        except Exception as e:
            count_err += 1
            msg = f"[ERRO] {fname}: {e}"

        if window is not None:
            window["-LOG-"].print(msg)
        else:
            print(msg)

    resumo = f"Concluído. Sucesso: {count_ok}, Erros: {count_err}."
    if window is not None:
        window["-LOG-"].print(resumo)
    else:
        print(resumo)


def main():
    sg.theme("DarkBlue3")

    layout = [
        [sg.Text("Pasta com GeoTIFFs:")],
        [
            sg.Input(key="-IN_FOLDER-", size=(60, 1)),
            sg.FolderBrowse("Procurar")
        ],
        [sg.Text("Pasta de saída dos TXT:")],
        [
            sg.Input(key="-OUT_FOLDER-", size=(60, 1)),
            sg.FolderBrowse("Procurar")
        ],
        [sg.Button("Processar", bind_return_key=True), sg.Button("Sair")],
        [sg.Frame("Log", [[sg.Multiline(key="-LOG-", size=(90, 20), autoscroll=True, disabled=True)]])],
    ]

    window = sg.Window("Exportar Metadados de GeoTIFF para TXT", layout, finalize=True)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Sair"):
            break

        if event == "Processar":
            in_folder = values["-IN_FOLDER-"]
            out_folder = values["-OUT_FOLDER-"]

            if not in_folder or not os.path.isdir(in_folder):
                sg.popup_error("Selecione uma pasta válida com os GeoTIFFs.")
                continue
            if not out_folder or not os.path.isdir(out_folder):
                sg.popup_error("Selecione uma pasta válida de saída para os TXT.")
                continue

            window["-LOG-"].update(value="")
            window["-LOG-"].print(f"Processando pasta: {in_folder}")
            process_folder(in_folder, out_folder, window=window)
            sg.popup_ok("Processamento concluído.")

    window.close()


if __name__ == "__main__":
    main()
