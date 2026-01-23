# === txt_utils.py (final) ===
import re
import chardet
import pandas as pd

def detectar_codificacao(arquivo) -> str:
    with open(arquivo, 'rb') as f:
        return (chardet.detect(f.read(10000)).get('encoding') or 'utf-8')

def _try_read(path, sep, enc):
    if sep in {',', ';', '\t'}:
        return pd.read_csv(path, sep=sep, engine='c', encoding=enc)
    if sep == ' ':
        return pd.read_csv(path, sep=r'\s+', engine='python', encoding=enc)
    if sep is None:
        return pd.read_csv(path, sep=r'\s+', engine='python', encoding=enc)
    return pd.read_csv(path, sep=sep, engine='python', encoding=enc)

def ler_bruto(path_txt: str) -> pd.DataFrame:
    enc = detectar_codificacao(path_txt)
    seps = [';', ',', '\t', ' ', None]
    best_df, best_n = None, -1
    for s in seps:
        try:
            df = _try_read(path_txt, s, enc)
            if df.shape[1] > best_n:
                best_df, best_n = df, df.shape[1]
        except Exception:
            continue
    if best_df is None:
        raise ValueError("Falha ao ler o TXT: não consegui determinar o separador.")
    return best_df

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower().strip())

SYN_TYPE = {'type','tipo','tp','classe','class','categoria','cat','bloco','name','nome','id','feature'}
SYN_E = {'e','x','east','leste','este','longitude','long','easting','utme','utm_e','eutm','coorde',
         'coordx','xc','coord_x','esteutm','lesteutm','xutm'}
SYN_N = {'n','y','north','norte','latitude','lat','northing','utmn','utm_n','nutm','coordn',
         'coordy','yc','coord_y','norteutm','yutm'}
SYN_Z = {'z','alt','altura','cota','elev','elevation','h','cotaaltimetrica','altitude','quota'}

def _auto_map_columns(df: pd.DataFrame):
    cols = list(df.columns); norm = [_norm(str(c)) for c in cols]
    idx = {'type': None, 'E': None, 'N': None, 'Z': None}
    for i, s in enumerate(norm):
        if s in SYN_TYPE and idx['type'] is None: idx['type'] = cols[i]
        if s in SYN_E    and idx['E']    is None: idx['E']    = cols[i]
        if s in SYN_N    and idx['N']    is None: idx['N']    = cols[i]
        if s in SYN_Z    and idx['Z']    is None: idx['Z']    = cols[i]
    return idx

def _positional_map(df: pd.DataFrame):
    cols = list(df.columns)
    if len(cols) >= 3:
        m = {'type': cols[0], 'E': cols[1], 'N': cols[2]}
        if len(cols) >= 4: m['Z'] = cols[3]
        return m
    return None

def detectar_colunas(df: pd.DataFrame):
    m = _auto_map_columns(df)
    if not m['type'] or not m['E'] or not m['N']:
        mp = _positional_map(df)
        if mp:
            if not m['type']: m['type'] = mp.get('type')
            if not m['E']:    m['E']    = mp.get('E')
            if not m['N']:    m['N']    = mp.get('N')
            if not m.get('Z'): m['Z']   = mp.get('Z')
    if not (m['type'] and m['E'] and m['N']):
        raise ValueError(
            "TXT sem colunas mínimas. Precisamos de 3 colunas: "
            "type (ou 'tipo'), E (ou X/Leste/UTM_E), N (ou Y/Norte/UTM_N).\n"
            f"Colunas lidas: {list(df.columns)}"
        )
    return m

def ler_txt(path_txt: str) -> pd.DataFrame:
    df_raw = ler_bruto(path_txt)
    m = detectar_colunas(df_raw)
    df = df_raw.rename(columns={v: k for k, v in m.items() if v is not None})
    cols = ['type', 'E', 'N'] + (['Z'] if 'Z' in df.columns else [])
    df = df[cols].copy()

    df['type'] = df['type'].astype(str).str.strip()

    for c in ['E', 'N', 'Z']:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].str.replace(',', '.', regex=False)
    for c in ['E', 'N', 'Z']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    before = len(df)
    df = df.dropna(subset=['E', 'N'])
    dropped = before - len(df)
    if dropped > 0:
        print(f"[txt] Aviso: {dropped} linha(s) descartada(s) por coordenadas inválidas (E/N).")
    return df
