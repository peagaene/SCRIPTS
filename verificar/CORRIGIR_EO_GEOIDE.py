# CORRIGIR_EO_GEOIDE.py
import os, math, warnings, re
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog
from pyproj import Transformer, CRS

# SciPy é opcional (IDW). Se não houver, usamos nearest eficiente.
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# =============================
# Leitura robusta do EO (linha-a-linha)
# =============================
def read_eo_table(path):
    """
    Parser robusto para EO:
    - aceita separadores: TAB, 1+ espaços, mistura;
    - aceita vírgula decimal;
    - 7 colunas: ID E N Z OMEGA PHI KAPPA
    - 6 colunas:    E N Z OMEGA PHI KAPPA
    - ignora cabeçalho 'ID E N Z OMEGA PHI KAPPA' se existir.
    """
    rows = []
    bad = 0
    header_re = re.compile(r'^\s*ID\s+E\s+N\s+Z\s+OMEGA\s+PHI\s+KAPPA\s*$', re.IGNORECASE)
    split_re  = re.compile(r'\s+')  # 1+ espaços/tabs

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            # ignora cabeçalho conhecido
            if header_re.match(s):
                continue

            parts = split_re.split(s)

            def norm_num(x):  # normaliza vírgula decimal
                return x.replace(',', '.')

            if len(parts) >= 7:
                # usa as 7 primeiras: ID + 6 números
                idv = parts[0]
                nums = list(map(norm_num, parts[1:7]))
            elif len(parts) == 6:
                idv = None
                nums = list(map(norm_num, parts[0:6]))
            else:
                bad += 1
                continue

            try:
                e     = float(nums[0])
                n     = float(nums[1])
                z     = float(nums[2])
                omega = float(nums[3])
                phi   = float(nums[4])
                kappa = float(nums[5])
            except Exception:
                bad += 1
                continue

            row = {'E': e, 'N': n, 'Z': z, 'OMEGA': omega, 'PHI': phi, 'KAPPA': kappa}
            if idv is not None:
                row['ID'] = idv
            rows.append(row)

    if bad:
        print(f"⚠️ {bad} linha(s) ignorada(s) por formato inválido.")
    if not rows:
        raise ValueError("Nenhuma linha válida encontrada no EO (verifique separadores/decimais).")

    df = pd.DataFrame(rows)
    # sanity check final em E/N/Z
    before = len(df)
    df = df[np.isfinite(df['E']) & np.isfinite(df['N']) & np.isfinite(df['Z'])].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"⚠️ {dropped} linha(s) removida(s) por valores inválidos em E/N/Z.")
    if len(df) == 0:
        raise ValueError("Nenhuma linha válida em E/N/Z após a leitura.")
    return df

# =============================
# Sampler para GRID TXT (UTM ou lon/lat; IDW/nearest)
# =============================
class TextGeoidSampler:
    """
    Lê grid MAPGEO2020 em TXT (3 números por linha: X Y N).
    - Detecta se X,Y parecem lon/lat; caso contrário assume projetado (m).
    - Interpola via cKDTree (IDW) se disponível; senão nearest por blocos.
    """
    def __init__(self, grid_txt_path: str):
        if not os.path.exists(grid_txt_path):
            raise RuntimeError(f"Grid TXT não encontrado: {grid_txt_path}")

        rows = []
        num_re = re.compile(r'[-+]?(?:\d+(?:[.,]\d+)?|\d*[.,]\d+)')
        with open(grid_txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s[0] in "#;":
                    continue
                nums = [m.group(0) for m in num_re.finditer(s)]
                if len(nums) >= 3:
                    x = float(nums[0].replace(',', '.'))
                    y = float(nums[1].replace(',', '.'))
                    n = float(nums[2].replace(',', '.'))
                    rows.append((x, y, n))

        if not rows:
            raise RuntimeError("Não foi possível interpretar o TXT do MAPGEO2020 (esperado: X Y N).")

        arr = np.asarray(rows, dtype=float)
        msk = np.isfinite(arr).all(axis=1)
        if not msk.all():
            print(f"⚠️ {np.count_nonzero(~msk)} linha(s) inválida(s) no grid TXT foram ignoradas.")
            arr = arr[msk]

        self.X = arr[:, 0]
        self.Y = arr[:, 1]
        self.N = arr[:, 2]

        self.xmin, self.xmax = float(np.nanmin(self.X)), float(np.nanmax(self.X))
        self.ymin, self.ymax = float(np.nanmin(self.Y)), float(np.nanmax(self.Y))

        # Brasil lon/lat típico
        self.is_lonlat = (-180 <= self.xmin <= 180 and -180 <= self.xmax <= 180 and
                          -90 <= self.ymin <= 90 and -90 <= self.ymax <= 90)

        if HAS_SCIPY and self.X.size >= 4:
            pts = np.column_stack([self.X, self.Y])
            self.tree = cKDTree(pts)
            self._use_tree = True
        else:
            self.tree = None
            self._use_tree = False

    def _nearest_block(self, x, y, k=64):
        d = 1000.0  # 1 km inicial
        for _ in range(8):
            mask = (self.X >= x - d) & (self.X <= x + d) & (self.Y >= y - d) & (self.Y <= y + d)
            idx = np.where(mask)[0]
            if idx.size:
                sub = idx
                if sub.size > k:
                    dx = self.X[sub] - x
                    dy = self.Y[sub] - y
                    sub = sub[np.argsort(dx*dx + dy*dy)[:k]]
                dx = self.X[sub] - x
                dy = self.Y[sub] - y
                j = sub[np.argmin(dx*dx + dy*dy)]
                return float(self.N[j])
            d *= 2
        return np.nan

    def sample(self, x, y, method="idw"):
        if not (np.isfinite(x) and np.isfinite(y)):
            return np.nan
        if self._use_tree:
            k = min(8, len(self.X))
            dist, idx = self.tree.query([x, y], k=k)
            dist = np.atleast_1d(dist)
            idx = np.atleast_1d(idx)
            vals = self.N[idx]
            zero = (dist == 0)
            if zero.any():
                return float(vals[zero][0])
            w = 1.0 / np.maximum(dist, 1e-12)**2
            return float(np.sum(w * vals) / np.sum(w))
        else:
            return self._nearest_block(x, y)

# =============================
# Sampler para TIF/GTX (GDAL; bilinear/nearest)
# =============================
class GDALGeoidSampler:
    def __init__(self, grid_path: str):
        try:
            from osgeo import gdal
        except Exception:
            raise SystemExit("⚠️ GDAL não encontrado. Instale: conda install -c conda-forge gdal")
        self.gdal = gdal
        self.ds = gdal.Open(grid_path, gdal.GA_ReadOnly)
        if self.ds is None:
            raise RuntimeError(f"Não foi possível abrir o grid: {grid_path}")
        self.band = self.ds.GetRasterBand(1)
        self.nodata = self.band.GetNoDataValue()
        self.gt = self.ds.GetGeoTransform()
        inv = gdal.InvGeoTransform(self.gt)
        if inv is None:
            raise RuntimeError("Falha ao inverter GeoTransform do grid.")
        self.inv_gt = inv
        self.xsize = self.ds.RasterXSize
        self.ysize = self.ds.RasterYSize

    def _xy_to_pxpy(self, x, y):
        px = self.inv_gt[0] + self.inv_gt[1]*x + self.inv_gt[2]*y
        py = self.inv_gt[3] + self.inv_gt[4]*x + self.inv_gt[5]*y
        return px, py

    def _read_safe(self, xoff, yoff, xsize, ysize):
        xoff = max(0, min(self.xsize-1, xoff))
        yoff = max(0, min(self.ysize-1, yoff))
        xsize = max(1, min(self.xsize - xoff, xsize))
        ysize = max(1, min(self.ysize - yoff, ysize))
        return self.band.ReadAsArray(int(xoff), int(yoff), int(xsize), int(ysize))

    def sample(self, lon, lat, method="bilinear"):
        px, py = self._xy_to_pxpy(lon, lat)
        if px < -1 or py < -1 or px > self.xsize or py > self.ysize:
            return np.nan
        ix, iy = int(round(px)), int(round(py))
        nearest_val = self._read_safe(ix, iy, 1, 1)[0, 0]
        if method == "nearest":
            return np.nan if (self.nodata is not None and nearest_val == self.nodata) else float(nearest_val)
        j = int(math.floor(px)); i = int(math.floor(py))
        dx, dy = px - j, py - i
        win = self._read_safe(j, i, 2, 2)
        if win.shape[0] < 2 or win.shape[1] < 2:
            return np.nan if (self.nodata is not None and nearest_val == self.nodata) else float(nearest_val)
        v00, v10, v01, v11 = win[0,0], win[0,1], win[1,0], win[1,1]
        if self.nodata is not None and any(v == self.nodata for v in (v00, v10, v01, v11)):
            return np.nan if (self.nodata is not None and nearest_val == self.nodata) else float(nearest_val)
        v0 = v00*(1-dx) + v10*dx
        v1 = v01*(1-dx) + v11*dx
        return float(v0*(1-dy) + v1*dy)

# =============================
# Escrita de saída (com ID se existir)
# =============================
def write_output(df, path):
    cols = ['E', 'N', 'Z_ORTO', 'OMEGA', 'PHI', 'KAPPA']
    if 'ID' in df.columns:
        cols = ['ID'] + cols
    df[cols].to_csv(path, sep="\t", index=False)

def make_output_path_for_input(input_path):
    folder = os.path.dirname(input_path)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(folder, f"{base}_GEOIDE.txt")

# =============================
# Processamento principal
# =============================
def process(input_path, output_path, grid_path, input_epsg=31983):
    df = read_eo_table(input_path)

    ext = os.path.splitext(grid_path)[1].lower()
    if ext == ".txt":
        txt_sampler = TextGeoidSampler(grid_path)
        if txt_sampler.is_lonlat:
            transformer = Transformer.from_crs(CRS.from_epsg(input_epsg), CRS.from_epsg(4674), always_xy=True)
            def get_N(e, n):
                lon, lat = transformer.transform(e, n)
                return txt_sampler.sample(lon, lat)
        else:
            def get_N(e, n):
                return txt_sampler.sample(e, n)
        sampler_desc = "TXT"
    else:
        tif_sampler = GDALGeoidSampler(grid_path)
        transformer = Transformer.from_crs(CRS.from_epsg(input_epsg), CRS.from_epsg(4674), always_xy=True)
        def get_N(e, n):
            lon, lat = transformer.transform(e, n)
            return tif_sampler.sample(lon, lat, method="bilinear")
        sampler_desc = "GDAL"

    Z_ortos = []
    outside = 0
    for _, row in df.iterrows():
        e, n, h = float(row['E']), float(row['N']), float(row['Z'])
        if not (np.isfinite(e) and np.isfinite(n) and np.isfinite(h)):
            Z_ortos.append(np.nan); outside += 1; continue
        N = get_N(e, n)
        if N is None or not np.isfinite(N):
            outside += 1
            H = h   # mantém elipsoidal se não achar N
        else:
            H = h + N  # lógica invertida (pedido)
        Z_ortos.append(H)

    df['Z_ORTO'] = Z_ortos
    if outside > 0:
        warnings.warn(f"{outside} ponto(s) fora da grade/sem vizinhos; Z mantido elipsoidal nesses casos. [Sampler={sampler_desc}]")
    write_output(df, output_path)

# =============================
# UI (input/grid; saída automática ao lado do input)
# =============================
def main():
    Tk().withdraw()
    input_path = filedialog.askopenfilename(
        title="Selecione o arquivo de EO (TXT/CSV) - ID E N Z OMEGA PHI KAPPA",
        filetypes=[("Text/CSV", "*.txt *.csv"), ("Todos", "*.*")]
    )
    if not input_path:
        print("❌ Nenhum arquivo de entrada selecionado.")
        return

    grid_path = filedialog.askopenfilename(
        title="Selecione o grid MAPGEO2020 (.txt, .tif ou .gtx)",
        filetypes=[("Grid TXT/TIF/GTX", "*.txt *.tif *.gtx"), ("Todos", "*.*")]
    )
    if not grid_path:
        print("❌ Nenhum grid selecionado.")
        return

    # saída automática ao lado do input
    output_path = make_output_path_for_input(input_path)

    # Ajuste aqui se seu EO não for UTM 23S (31983): 22S=31982, 24S=31984, etc.
    process(input_path, output_path, grid_path, input_epsg=31983)
    print(f"✅ Arquivo processado e salvo em: {output_path}")

if __name__ == "__main__":
    main()
