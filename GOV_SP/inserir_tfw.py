# -*- coding: utf-8 -*-
import os
import re
from typing import Iterable, Optional, Tuple

# ===== CONFIGURAÇÕES =====
ROOT_DIR = r"\\192.168.2.27\h\SI06\RGB\RGB_2\sem_proj"
TARGET_NUMBERS = [
    "092039","094043","094046","094047","095041","095043","095045","099017","097039","099037", "109019"
]
RECURSIVE = True             # Varre subpastas
STRICT_NUMERIC = True        # Casa IDs sem dígitos colados antes/depois
EPSG = 31983                 # Fixo: SIRGAS 2000 / UTM 23S
SKIP_IF_EXISTS = True        # Pula se <nome>_geo.tif já existir

# ===== DEPENDÊNCIAS =====
try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
except Exception as e:
    raise SystemExit("❌ GDAL não encontrado. Instale com: pip install gdal") from e

# ===== WORLD FILE / GEO =====
def read_worldfile(path: str) -> Tuple[float, float, float, float, float, float]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) < 6:
        raise ValueError(f"Worldfile inválido (precisa de 6 linhas): {path}")
    vals = [float(x.replace(",", ".")) for x in lines[:6]]
    A, D, B, E, C, F = vals
    return A, D, B, E, C, F

def tfw_to_geotransform(A, D, B, E, C, F):
    # Converte TFW (centro do pixel) para GeoTransform (canto do pixel 0,0)
    GT0 = C - 0.5*A - 0.5*B
    GT1 = A
    GT2 = B
    GT3 = F - 0.5*D - 0.5*E
    GT4 = D
    GT5 = E
    return (GT0, GT1, GT2, GT3, GT4, GT5)

def worldfile_candidates(img_path: str):
    """Retorna possíveis nomes de worldfile para a imagem (normal + sufixo '0')."""
    base, ext = os.path.splitext(img_path)
    ext = ext.lower()
    wf_by_ext = []
    if ext in [".tif", ".tiff"]:
        wf_by_ext += ["tfw", "tifw"]
    elif ext in [".jpg", ".jpeg", ".jpe", ".jfif"]:
        wf_by_ext += ["jgw", "jgwx"]
    elif ext == ".png":
        wf_by_ext += ["pgw"]
    elif ext in [".jp2", ".j2k", ".j2c"]:
        wf_by_ext += ["j2w"]
    elif ext == ".bmp":
        wf_by_ext += ["bpw"]
    wf_by_ext += ["wld"]  # genérico
    cands = []
    for wfext in wf_by_ext:
        cands.append(base + "." + wfext)        # normal
        cands.append(base + "0." + wfext)       # sufixo '0' antes do ponto
        cands.append(base + "." + wfext + "0")  # raro: extensão + '0'
    return cands

def find_worldfile(img_path: str) -> Optional[str]:
    for c in worldfile_candidates(img_path):
        if os.path.exists(c):
            return c
    return None

# ===== SAÍDA =====
def out_geotiff_path(img_path: str) -> str:
    d, name = os.path.split(img_path)
    stem, _ = os.path.splitext(name)
    return os.path.join(d, f"{stem}_geo.tif")

# ===== ID MATCH =====
def build_id_matchers(ids, strict_numeric=True):
    pats = []
    for s in sorted(set(ids)):  # remove duplicados
        if strict_numeric:
            pats.append(re.compile(rf"(?<!\d){re.escape(s)}(?!\d)"))
        else:
            pats.append(re.compile(re.escape(s)))
    return pats

def filename_matches_ids(path: str, patterns) -> bool:
    stem = os.path.splitext(os.path.basename(path))[0]
    return any(rx.search(stem) for rx in patterns)

# ===== ITERAÇÃO DE IMAGENS =====
IMG_EXTS = {".tif",".tiff",".jpg",".jpeg",".png",".jp2",".j2k",".bmp"}

def iter_images(root: str, recursive: bool=True) -> Iterable[str]:
    if recursive:
        for d, _sub, files in os.walk(root):
            for f in files:
                if os.path.splitext(f.lower())[1] in IMG_EXTS:
                    yield os.path.join(d, f)
    else:
        for f in os.listdir(root):
            p = os.path.join(root, f)
            if os.path.isfile(p) and os.path.splitext(f.lower())[1] in IMG_EXTS:
                yield p

# ===== GRAVAÇÃO =====
def write_geotiff_with_tfw(img_path: str, tfw_path: str, out_tif: str, epsg: int):
    A, D, B, E, C, F = read_worldfile(tfw_path)
    GT = tfw_to_geotransform(A, D, B, E, C, F)

    try:
        src = gdal.Open(img_path, gdal.GA_ReadOnly)
    except Exception:
        src = None

    if src is None:
        # Fallback: ler via Pillow e gravar com GDAL (para formatos sem driver GDAL)
        try:
            from PIL import Image
            import numpy as np
        except Exception as e:
            raise RuntimeError("Para fallback sem driver GDAL, instale Pillow e numpy.") from e

        pil = Image.open(img_path)
        pil = pil.convert("RGB") if pil.mode not in ("L", "RGB") else pil
        arr = np.array(pil)
        h, w = arr.shape[0], arr.shape[1]
        bands = 1 if arr.ndim == 2 else arr.shape[2]

        driver = gdal.GetDriverByName("GTiff")
        dst = driver.Create(out_tif, w, h, bands, gdal.GDT_Byte, options=["TILED=YES", "COMPRESS=LZW"])
        dst.SetGeoTransform(GT)

        srs = osr.SpatialReference(); srs.ImportFromEPSG(int(epsg))
        dst.SetProjection(srs.ExportToWkt())

        if bands == 1:
            dst.GetRasterBand(1).WriteArray(arr)
        else:
            for b in range(bands):
                dst.GetRasterBand(b+1).WriteArray(arr[:, :, b])
        dst.FlushCache(); dst = None
        return

    # Caminho principal: copiar dados com GDAL
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.CreateCopy(out_tif, src, strict=0, options=["TILED=YES", "COMPRESS=LZW"])
    dst.SetGeoTransform(GT)

    srs = osr.SpatialReference(); srs.ImportFromEPSG(int(EPSG))
    dst.SetProjection(srs.ExportToWkt())

    dst.FlushCache()
    src = None; dst = None

# ===== MAIN =====
def main():
    ids_patterns = build_id_matchers(TARGET_NUMBERS, STRICT_NUMERIC)

    print(f"[i] Raiz: {ROOT_DIR}")
    print(f"[i] Total de IDs na lista: {len(sorted(set(TARGET_NUMBERS)))}")
    print(f"[i] Recursivo: {RECURSIVE} | Casamento estrito: {STRICT_NUMERIC}")
    print(f"[i] EPSG fixo: {EPSG}\n")

    total_scan = 0
    total_listed_found = 0
    total_created = 0
    total_skipped_exist = 0
    total_no_tfw = 0
    total_errors = 0

    for img in iter_images(ROOT_DIR, RECURSIVE):
        total_scan += 1

        # SOMENTE AS FOTOS LISTADAS
        if not filename_matches_ids(img, ids_patterns):
            continue
        total_listed_found += 1

        out_tif = out_geotiff_path(img)
        if SKIP_IF_EXISTS and os.path.exists(out_tif):
            print(f"[-] Já existe: {os.path.basename(out_tif)} — pulando")
            total_skipped_exist += 1
            continue

        tfw = find_worldfile(img)
        if not tfw:
            print(f"[!] TFW não encontrado (nem com sufixo '0'): {img}")
            total_no_tfw += 1
            continue

        try:
            write_geotiff_with_tfw(img, tfw, out_tif, epsg=EPSG)
            print(f"[OK] {os.path.basename(img)}  →  {os.path.basename(out_tif)}  (TFW: {os.path.basename(tfw)})")
            total_created += 1
        except Exception as e:
            print(f"[ERRO] {img}\n      {e}")
            total_errors += 1

    print("\n==== RESUMO ====")
    print(f"Imagens varridas................: {total_scan}")
    print(f"Imagens que batem IDs (listadas): {total_listed_found}")
    print(f"GeoTIFFs criados................: {total_created}")
    print(f"Pulado (_geo já existia)........: {total_skipped_exist}")
    print(f"Sem TFW (incl. sufixo '0')......: {total_no_tfw}")
    print(f"Erros...........................: {total_errors}")

if __name__ == "__main__":
    main()
