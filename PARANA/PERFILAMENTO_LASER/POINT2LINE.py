import geopandas as gpd
from shapely.geometry import LineString
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread, Lock
import queue
import fnmatch

fila = queue.Queue()
ultimo_diretorio = os.getcwd()
thread_em_execucao = False
thread_lock = Lock()
log_widget = None

def normalizar_caminho_windows(caminho_original: str) -> str:
    """Converte caminhos no formato "/x/…" para "X:/…" no Windows e normaliza barras.

    Também remove aspas extras (simples ou duplas) caso existam.
    """
    if not caminho_original:
        return caminho_original

    caminho = caminho_original.strip().strip('"').strip("'")

    if os.name == 'nt' and len(caminho) >= 3 and caminho[0] == '/' and caminho[2] == '/':
        # Ex.: "/i/dir/arquivo.shp" -> "I:/dir/arquivo.shp"
        potencial_drive = caminho[1].upper()
        if potencial_drive.isalpha():
            caminho = f"{potencial_drive}:{caminho[2:]}"

    # Normaliza separadores e remove ".." se existirem
    caminho = os.path.normpath(caminho)
    return caminho

def shapefile_existe(caminho_shp: str) -> bool:
    """Verifica a presença mínima dos arquivos de um shapefile (.shp, .shx, .dbf)."""
    base, ext = os.path.splitext(caminho_shp)
    if ext.lower() != '.shp':
        return False
    componentes = [f"{base}.shp", f"{base}.shx", f"{base}.dbf"]
    return all(os.path.exists(c) for c in componentes)

def remover_componentes_shapefile(caminho_shp: str) -> None:
    """Remove arquivos relacionados a um shapefile para permitir regravação limpa."""
    base, _ = os.path.splitext(caminho_shp)
    extensoes = [
        '.shp', '.shx', '.dbf', '.prj', '.cpg', '.qpj', '.sbn', '.sbx', '.qix'
    ]
    for ext in extensoes:
        arquivo = f"{base}{ext}"
        try:
            if os.path.exists(arquivo):
                os.remove(arquivo)
        except Exception:
            # Se não conseguir remover algum sidecar, seguimos; o driver pode sobrescrever
            pass

def log(mensagem: str) -> None:
    """Escreve no console e no widget de log sem pausar a execução."""
    print(mensagem)
    try:
        if log_widget is not None:
            log_widget.configure(state='normal')
            log_widget.insert('end', mensagem + "\n")
            log_widget.see('end')
            log_widget.configure(state='disabled')
    except Exception:
        pass

def processar_fila(tolerancia):
    global thread_em_execucao

    with thread_lock:
        if thread_em_execucao:
            return  # Já está processando
        thread_em_execucao = True

    try:
        while not fila.empty():
            shp_path = fila.get()
            try:
                # Normaliza possíveis caminhos no formato "/i/..." vindos de outros softwares
                shp_path = normalizar_caminho_windows(shp_path)

                # Validação explícita para evitar erro genérico do GDAL
                if not shapefile_existe(shp_path):
                    log(
                        (
                            f"[ERRO] Shapefile ausente/incompleto (.shp/.shx/.dbf): {shp_path}"
                        )
                    )
                    continue

                gdf = gpd.read_file(shp_path)

                # Se já for linha, não processa
                tipos = set(gdf.geometry.geom_type.astype(str).str.lower())
                if 'linestring' in tipos or 'multilinestring' in tipos:
                    log(f"[AVISO] {os.path.basename(shp_path)} já é uma linha. Pulando.")
                    continue

                if 'TIME' not in gdf.columns:
                    log(f"[ERRO] {os.path.basename(shp_path)} não possui a coluna 'TIME'.")
                    continue

                gdf['TIME'] = gdf['TIME'].astype(float)
                gdf = gdf.sort_values(by='TIME')

                coords_array = gdf.geometry.apply(lambda p: (p.x, p.y)).to_numpy()
                linha = LineString(coords_array)
                linha = linha.simplify(tolerance=tolerancia, preserve_topology=True)

                gdf_linha = gpd.GeoDataFrame(geometry=[linha], crs=gdf.crs)
                nome_saida = shp_path  # substituir o próprio arquivo

                # Remover componentes antigos para evitar conflitos de schema/geom
                remover_componentes_shapefile(nome_saida)

                gdf_linha.to_file(nome_saida, driver='ESRI Shapefile')

                log(f"[OK] {os.path.basename(nome_saida)} sobrescrito.")
            except Exception as e:
                log(f"[ERRO] Falha em {os.path.basename(shp_path)}: {str(e)}")

        log("[INFO] Conversão concluída.")
    finally:
        with thread_lock:
            thread_em_execucao = False

def selecionar_arquivos(entry_tolerancia):
    global ultimo_diretorio

    try:
        tolerancia = float(entry_tolerancia.get())
    except ValueError:
        log("[ERRO] Tolerância inválida. Use um número (ex: 0.05).")
        return

    arquivos = filedialog.askopenfilenames(
        filetypes=[("Shapefiles", "*.shp")],
        initialdir=ultimo_diretorio
    )

    if not arquivos:
        return

    ultimo_diretorio = os.path.dirname(arquivos[0])

    for shp in arquivos:
        fila.put(normalizar_caminho_windows(shp))

    # Sempre tenta iniciar a thread (só inicia se nenhuma estiver ativa)
    Thread(target=processar_fila, args=(tolerancia,), daemon=True).start()

def selecionar_pasta(entry_tolerancia):
    global ultimo_diretorio

    try:
        tolerancia = float(entry_tolerancia.get())
    except ValueError:
        log("[ERRO] Tolerância inválida. Use um número (ex: 0.05).")
        return

    diretorio = filedialog.askdirectory(initialdir=ultimo_diretorio)
    if not diretorio:
        return

    ultimo_diretorio = diretorio

    # Varrer recursivamente procurando por *_TRJ.shp
    encontrados = []
    for raiz, _, arquivos in os.walk(diretorio):
        for nome in arquivos:
            if fnmatch.fnmatch(nome.lower(), "*_trj.shp"):
                encontrados.append(os.path.join(raiz, nome))

    if not encontrados:
        log("[INFO] Não foram encontrados shapefiles *_TRJ.shp nesta pasta.")
        return

    for shp in encontrados:
        fila.put(normalizar_caminho_windows(shp))

    Thread(target=processar_fila, args=(tolerancia,), daemon=True).start()

def iniciar_interface():
    root = tk.Tk()
    root.title("Conversor de Pontos para Linha (TIME)")

    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack()

    btn = tk.Button(frame, text="Selecionar Shapefiles", command=lambda: selecionar_arquivos(entry_tol), width=30, height=2)
    btn.grid(row=0, column=0, columnspan=2, pady=(0, 6))

    btn2 = tk.Button(frame, text="Selecionar Pasta (_TRJ.shp)", command=lambda: selecionar_pasta(entry_tol), width=30, height=2)
    btn2.grid(row=1, column=0, columnspan=2, pady=(0, 10))

    tk.Label(frame, text="Tolerância de simplificação (ex: 0.05):").grid(row=2, column=0, sticky="e")
    entry_tol = tk.Entry(frame, width=10)
    entry_tol.insert(0, "0.05")
    entry_tol.grid(row=2, column=1, sticky="w")

    tk.Label(frame, text="Os arquivos originais serão substituídos", fg="gray").grid(row=3, column=0, columnspan=2, pady=(10, 6))

    # Log de execução
    global log_widget
    log_widget = tk.Text(frame, height=10, width=60, state='disabled')
    log_widget.grid(row=4, column=0, columnspan=2, sticky='nsew')
    scroll = tk.Scrollbar(frame, command=log_widget.yview)
    log_widget['yscrollcommand'] = scroll.set
    scroll.grid(row=4, column=2, sticky='ns')

    root.mainloop()

iniciar_interface()
