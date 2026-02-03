import os
import re
from typing import Optional
import tkinter as tk
from tkinter import filedialog, messagebox
try:
    from osgeo import gdal, osr
    _HAS_GDAL = True
except Exception:
    gdal = None
    osr = None
    _HAS_GDAL = False

try:
    import rasterio
    from rasterio.crs import CRS
    _HAS_RASTERIO = True
except Exception:
    rasterio = None
    CRS = None
    _HAS_RASTERIO = False


def selecionar_pasta():
    return filedialog.askdirectory(title="Selecione a pasta dos arquivos")


def remover_revisoes(base: str) -> str:
    """
    Remove sufixos de revisao (_R0, _R1...) no fim para evitar repeticoes.
    Ex.: ES_..._R0_R0 -> ES_...
    """
    partes = base.split("_")
    while partes and re.fullmatch(r"R\d+", partes[-1]):
        partes.pop()
    return "_".join(partes)


def aplicar_bloco(base: str, bloco: str) -> str:
    """
    Insere ou substitui o bloco (uma letra) apos o segundo token (ex.: ES_L09_E_*).
    Se ja houver bloco no slot (uma unica letra), ele e trocado; caso contrario, e inserido.
    """
    if not bloco:
        return base

    bloco = bloco.strip()
    partes = base.split("_")
    if len(partes) < 2:
        return base

    idx = 2
    if len(partes) > idx and len(partes[idx]) == 1:
        partes[idx] = bloco
    else:
        partes.insert(idx, bloco)
    return "_".join(partes)


def ensure_tif_epsg(tif_path: str, default_epsg: int = 31982) -> Optional[int]:
    """
    Garante que um GeoTIFF tenha EPSG definido.
    Se nao for possivel identificar o EPSG, define o default (ex.: 31982).
    Retorna o EPSG identificado/definido ou None em caso de falha silenciosa.
    """
    if not tif_path or not os.path.isfile(tif_path):
        return None

    if _HAS_GDAL:
        ds = gdal.Open(tif_path, gdal.GA_Update)
        if ds is None:
            raise RuntimeError("GDAL nao conseguiu abrir o arquivo.")

        proj = ds.GetProjection()
        epsg = None
        if proj:
            srs = osr.SpatialReference()
            if srs.ImportFromWkt(proj) == 0:
                try:
                    srs.AutoIdentifyEPSG()
                except Exception:
                    pass
                epsg = srs.GetAuthorityCode(None)
                if epsg:
                    try:
                        epsg = int(epsg)
                    except Exception:
                        epsg = None

        if not epsg:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(int(default_epsg))
            ds.SetProjection(srs.ExportToWkt())
            ds.FlushCache()
            epsg = int(default_epsg)

        ds = None
        return epsg

    if _HAS_RASTERIO:
        with rasterio.open(tif_path, "r+") as ds:
            epsg = ds.crs.to_epsg() if ds.crs else None
            if not epsg:
                ds.crs = CRS.from_epsg(int(default_epsg))
                epsg = int(default_epsg)
        return epsg

    raise RuntimeError("GDAL/rasterio nao disponivel para consultar/definir EPSG.")


def renomear(pasta: str, novo_trecho: str, bloco_forcado: Optional[str] = None) -> int:
    """
    Renomeia arquivos LAS da pasta aplicando o trecho escolhido (NP/NPc_T/NPc_C)
    e insere/substitui o bloco_forcado no slot do nome, se fornecido.
    Retorna a quantidade renomeada.
    """
    if not pasta or not os.path.isdir(pasta):
        raise ValueError("Selecione uma pasta valida.")

    renomeados = 0

    for nome in os.listdir(pasta):
        caminho_antigo = os.path.join(pasta, nome)
        if not os.path.isfile(caminho_antigo):
            continue

        base, ext = os.path.splitext(nome)
        ext_lower = ext.lower()

        base = aplicar_regra(base, novo_trecho)
        if not base:
            continue

        if ext_lower in (".tif", ".tiff"):
            try:
                ensure_tif_epsg(caminho_antigo, 31982)
            except Exception as e:
                raise RuntimeError(f"Falha ao ajustar EPSG do TIF '{nome}': {e}")

        if bloco_forcado:
            base = aplicar_bloco(base, bloco_forcado)

        novo_nome = base + ext  # mantem extensao como esta
        caminho_novo = os.path.join(pasta, novo_nome)

        # Se nao mudou nada, nao tenta renomear
        if caminho_novo == caminho_antigo:
            continue

        os.rename(caminho_antigo, caminho_novo)
        renomeados += 1

    return renomeados


def aplicar_regra(base: str, novo_trecho: str) -> str:
    """
    Aplica regras de normalizacao no nome base e devolve novo base.
    Retorna string vazia se nao houver match de NP/NPc/MDT/MDS.
    """
    base_lower = base.lower()
    if not any(x in base_lower for x in ("_np_", "_npc_c_", "_npc_t_", "_mdt_", "_mds_")):
        return ""

    base = remover_revisoes(base)
    base = base.replace("-", "_")
    base = re.sub(r"_(NPc?_C|NPc?_T|NP|MDT|MDS)_", novo_trecho, base, flags=re.IGNORECASE)
    if not base.endswith("_R0"):
        base = f"{base}_R0"
    return base


def renomear_prj(prj_path: str, novo_trecho: str, bloco_forcado: Optional[str] = None) -> int:
    """
    Atualiza linhas "Block ..." do PRJ aplicando a mesma regra de nomes.
    Se bloco_forcado for informado, substitui todos os blocos pelo valor fornecido.
    Cria backup .bak e retorna quantidade de blocos alterados.
    """
    if not prj_path or not os.path.isfile(prj_path):
        raise ValueError("Selecione um arquivo PRJ valido.")

    with open(prj_path, "r", encoding="utf-8", errors="ignore") as f:
        linhas = f.readlines()

    out = []
    changed = 0
    for line in linhas:
        m = re.match(r"\s*Block\s+(.+)", line, flags=re.IGNORECASE)
        if not m:
            out.append(line)
            continue
        raw_name = m.group(1).strip()
        raw_path = os.path.normpath(raw_name)
        parent = os.path.dirname(raw_path)
        fname = os.path.basename(raw_path)
        stem, ext = os.path.splitext(fname)
        ext = ext or ".las"

        novo_base = aplicar_regra(stem, novo_trecho)
        if not novo_base:
            out.append(line)
            continue

        if bloco_forcado:
            novo_base = aplicar_bloco(novo_base, bloco_forcado)

        novo_nome = novo_base + ext.lower()
        novo_path = os.path.join(parent, novo_nome) if parent else novo_nome
        out.append(f"Block {novo_path.replace(os.sep, '/')}\n")
        changed += 1

    if changed == 0:
        return 0

    backup = prj_path + ".bak"
    try:
        with open(backup, "w", encoding="utf-8") as f:
            f.writelines(linhas)
        with open(prj_path, "w", encoding="utf-8") as f:
            f.writelines(out)
    except Exception:
        # Restaura original em caso de falha de escrita
        with open(prj_path, "w", encoding="utf-8") as f:
            f.writelines(linhas)
        raise

    return changed


def iniciar_interface():
    root = tk.Tk()
    root.title("Renomear NP / NPc / MDT / MDS")
    root.geometry("520x280")

    path_var = tk.StringVar()
    prj_var = tk.StringVar()
    tipo_var = tk.StringVar(value="_NP_")
    bloco_var = tk.StringVar()
    forcar_bloco_var = tk.BooleanVar(value=False)
    status_var = tk.StringVar(value="Selecione a pasta e o tipo.")

    def escolher_pasta():
        pasta = selecionar_pasta()
        if pasta:
            path_var.set(pasta)

    def escolher_prj():
        prj = filedialog.askopenfilename(title="Selecione o arquivo PRJ", filetypes=[("PRJ/TXT", "*.prj *.txt"), ("Todos", "*.*")])
        if prj:
            prj_var.set(prj)

    def executar():
        pasta = path_var.get().strip()
        prj_file = prj_var.get().strip()
        novo_trecho = tipo_var.get()
        bloco_forcado = bloco_var.get().strip() if forcar_bloco_var.get() else ""
        try:
            qtd = renomear(pasta, novo_trecho, bloco_forcado or None)
            msg = f"Arquivos: {qtd} renomeado(s)"
            if prj_file:
                try:
                    prj_qtd = renomear_prj(prj_file, novo_trecho, bloco_forcado or None)
                    msg += f" | PRJ: {prj_qtd} bloco(s) ajustado(s)"
                except Exception as e:
                    messagebox.showerror("Erro PRJ", f"Falha ao atualizar PRJ: {e}")
                    msg += " | PRJ: erro"
            status_var.set(f"Concluido. {msg}.")
            messagebox.showinfo("Concluido", status_var.get())
        except Exception as e:
            status_var.set(f"Erro: {e}")
            messagebox.showerror("Erro", str(e))

    frm = tk.Frame(root, padx=10, pady=10)
    frm.pack(fill="both", expand=True)

    tk.Label(frm, text="Pasta dos arquivos:").grid(row=0, column=0, sticky="w")
    tk.Entry(frm, textvariable=path_var, width=50).grid(row=1, column=0, sticky="we", padx=(0, 5))
    tk.Button(frm, text="Procurar...", command=escolher_pasta).grid(row=1, column=1, sticky="e")

    tk.Label(frm, text="Arquivo PRJ (opcional):").grid(row=2, column=0, sticky="w", pady=(8, 0))
    tk.Entry(frm, textvariable=prj_var, width=50).grid(row=3, column=0, sticky="we", padx=(0, 5))
    tk.Button(frm, text="Procurar PRJ...", command=escolher_prj).grid(row=3, column=1, sticky="e")

    tk.Label(frm, text="Bloco (arquivos/PRJ) opcional:").grid(row=4, column=0, sticky="w", pady=(8, 0))
    tk.Entry(frm, textvariable=bloco_var, width=50).grid(row=5, column=0, sticky="we", padx=(0, 5))
    tk.Checkbutton(frm, text="Forcar bloco (arquivos + PRJ)", variable=forcar_bloco_var).grid(row=5, column=1, sticky="w")

    tk.Label(frm, text="Tipo:").grid(row=6, column=0, sticky="w", pady=(10, 2))
    tipos = [
        ("NP", "_NP_"),
        ("NPc-T (Terreno)", "_NPc_T_"),
        ("NPc-C", "_NPc_C_"),
        ("MDT", "_MDT_"),
        ("MDS", "_MDS_"),
    ]
    for idx, (label, value) in enumerate(tipos):
        tk.Radiobutton(frm, text=label, variable=tipo_var, value=value).grid(row=7, column=idx, sticky="w")

    tk.Button(frm, text="Renomear", command=executar, width=15).grid(row=8, column=0, pady=(12, 5), sticky="w")
    tk.Label(frm, textvariable=status_var, fg="blue").grid(row=9, column=0, columnspan=2, sticky="w")

    frm.grid_columnconfigure(0, weight=1)
    root.mainloop()


if __name__ == "__main__":
    iniciar_interface()


