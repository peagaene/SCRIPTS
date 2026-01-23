import os
import re
import sys
import json
import time
import threading
import subprocess
import shutil
import tempfile
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit,
    QFileDialog, QLabel, QCheckBox, QProgressBar, QTabWidget, QListWidget,
    QHBoxLayout, QSpinBox
)
from PyQt5.QtCore import pyqtSignal, QObject

# ===== CONFIG =====
LASZIP_PATH = r'D:\LAStools\bin\laszip.exe'
LASINFO_PATH = r'D:\LAStools\bin\lasinfo.exe'
INVENTARIO_DIR = r'D:\00_Pedro\INVENTARIO_STORAGE'
DEFAULT_MAX_WORKERS = 3
DEFAULT_ETA_EVERY = 5
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# ===== SINAIS =====
class WorkerSignals(QObject):
    log = pyqtSignal(str)
    done = pyqtSignal()
    progress = pyqtSignal(int)

class InventorySignals(QObject):
    log = pyqtSignal(str)
    done = pyqtSignal()
    result = pyqtSignal(str, list)  # planilha, lista pastas LAS

# ===== UTILS =====
_version_regex = re.compile(r"version\s+([\d.]+)", re.IGNORECASE)

def quiet_popen(cmd):
    si = None
    creation = 0
    if os.name == "nt":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        creation = CREATE_NO_WINDOW
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            startupinfo=si, creationflags=creation)

def get_las_version(lasinfo_path, fname: Path):
    try:
        p = quiet_popen([lasinfo_path, "-i", _win_long_path(str(fname))])
        out, _ = p.communicate()
        if p.returncode != 0:
            return None
        m = _version_regex.search(out or "")
        return m.group(1) if m else None
    except Exception:
        return None

def _win_long_path(path_str: str) -> str:
    if os.name != "nt":
        return path_str
    if path_str.startswith("\\\\?\\"):
        return path_str
    # Long path prefix to avoid MAX_PATH issues
    if path_str.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path_str.lstrip("\\")
    return "\\\\?\\" + path_str

def _diagnose_las_read(path_obj: Path):
    try:
        if not path_obj.exists():
            return False, "arquivo nao existe"
        if not path_obj.is_file():
            return False, "nao e arquivo"
        try:
            size = path_obj.stat().st_size
        except OSError as e:
            return False, f"stat falhou: {e}"
        if size == 0:
            return False, "arquivo vazio"
        # Try a small read to catch permission/lock issues early
        with open(path_obj, "rb") as f:
            f.read(1)
        return True, ""
    except Exception as e:
        return False, f"nao foi possivel ler: {e}"

def _has_non_ascii(s: str) -> bool:
    return any(ord(ch) > 127 for ch in s)

def _ascii_safe_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_only).strip("._-")
    return safe or "arquivo"

def safe_folder_size_bytes(path: str) -> int:
    """Calcula tamanho total da pasta de forma resiliente, ignorando erros de leitura."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                # Ignora arquivos inacessíveis para não interromper o inventário
                continue
    return total

def format_size(num_bytes: int) -> str:
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < step:
            return f"{size:0.2f} {unit}"
        size /= step
    return f"{size * step:0.2f} TB"

# ===== THREAD DE CONVERSÃO =====
class MultiFolderConverterThread(threading.Thread):
    def __init__(self, folder_list, delete_input, signals, convert_to='laz',
                 max_workers=DEFAULT_MAX_WORKERS, eta_every=DEFAULT_ETA_EVERY):
        super().__init__()
        self.folder_list = list(folder_list)
        self.delete_input = delete_input
        self.signals = signals
        self.stop_requested = False
        self.force_stop_requested = False
        self.convert_to = convert_to.lower()
        self.max_workers = max(1, int(max_workers))
        self.eta_every = max(1, int(eta_every))
        self._running_processes = set()
        self._lock = threading.Lock()

    def stop(self):
        self.stop_requested = True

    def force_stop(self):
        self.force_stop_requested = True
        with self._lock:
            procs = list(self._running_processes)
        for p in procs:
            try:
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=2)
                    except Exception:
                        p.kill()
            except Exception:
                pass

    def _convert_one(self, input_path: Path, input_ext: str, output_ext: str,
                     i: int, total: int, start_time: float, folder_root: str):
        if self.force_stop_requested:
            return ("FORCE_STOP", input_path.name, "", folder_root)

        output_path = input_path.with_suffix(output_ext)
        if output_path.exists():
            return ("SKIP", input_path.name, "já existe", folder_root)

        ok, reason = _diagnose_las_read(input_path)
        if not ok:
            return ("ERROR", input_path.name, reason, folder_root)

        # Checa versão e pula se for antiga
        las_ver = get_las_version(LASINFO_PATH, input_path)
        if las_ver in ("1.0", "1.1"):
            return ("SKIP", input_path.name, "versão antiga (1.0/1.1) — pulado", folder_root)

        temp_converted_path = input_path

        # Compressão
        input_str = str(temp_converted_path)
        output_str = str(output_path)
        def run_laszip(in_path: str, out_path: str):
            p = quiet_popen([LASZIP_PATH, "-i", in_path, "-o", out_path])
            with self._lock:
                self._running_processes.add(p)
            out, err = p.communicate()
            with self._lock:
                self._running_processes.discard(p)
            return p.returncode, out, err

        rc, out, err = run_laszip(input_str, output_str)
        if rc != 0 and os.name == "nt" and max(len(input_str), len(output_str)) >= 240:
            # Retry with long path prefix for Windows shares/long paths
            rc, out, err = run_laszip(_win_long_path(input_str), _win_long_path(output_str))

        if rc != 0 and os.name == "nt" and (_has_non_ascii(input_str) or _has_non_ascii(output_str)):
            # Fallback: use a temp ASCII-only path for tools that fail on accents
            temp_dir = tempfile.mkdtemp(prefix="laszip_")
            temp_input = Path(temp_dir) / _ascii_safe_name(input_path.name)
            temp_output = Path(temp_dir) / _ascii_safe_name(output_path.name)
            try:
                shutil.copy2(input_path, temp_input)
                rc, out, err = run_laszip(str(temp_input), str(temp_output))
                if rc == 0 and temp_output.exists():
                    shutil.move(str(temp_output), str(output_path))
                    if not output_path.exists():
                        rc = 1
                        err = "falha ao mover resultado do temp para o destino"
            finally:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

        if rc != 0:
            return ("ERROR", input_path.name, (err or out or "").strip(), folder_root)

        if self.delete_input:
            try:
                input_path.unlink(missing_ok=True)
            except Exception:
                pass

        elapsed = time.time() - start_time
        avg = elapsed / max(1, i)
        eta = datetime.now() + timedelta(seconds=avg * (total - i))
        return ("OK", input_path.name, eta.strftime("%H:%M:%S"), folder_root)

    def run(self):
        if self.convert_to == 'laz':
            input_ext, output_ext = '.las', '.laz'
        elif self.convert_to == 'las':
            input_ext, output_ext = '.laz', '.las'
        else:
            self.signals.log.emit(f"❌ Extensão inválida: {self.convert_to}")
            self.signals.done.emit()
            return

        file_entries = [(Path(root) / f, folder) for folder in self.folder_list
                        for root, _, fs in os.walk(folder)
                        for f in fs if f.lower().endswith(input_ext)]
        total = len(file_entries)
        conv = skip = errc = 0
        conv_per_folder = {}
        self.signals.log.emit(f"🔍 {total} arquivos {input_ext} encontrados.\n")
        start_time = time.time()

        if not total:
            self.signals.progress.emit(100)
            self.signals.done.emit()
            return

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self._convert_one, f, input_ext, output_ext, i + 1, total, start_time, folder_root)
                        for i, (f, folder_root) in enumerate(file_entries)]
            for done, fut in enumerate(as_completed(futures), 1):
                if self.force_stop_requested: break
                status, name, info, folder_root = fut.result()
                if status == "OK":
                    conv += 1
                    if folder_root not in conv_per_folder:
                        conv_per_folder[folder_root] = 0
                    conv_per_folder[folder_root] += 1
                    if done % self.eta_every == 0 or done == total:
                        self.signals.log.emit(f"[{done}/{total}] ✅ {name} ⏳ ETA: {info}")
                    else:
                        self.signals.log.emit(f"[{done}/{total}] ✅ {name}")
                elif status == "SKIP":
                    skip += 1
                    self.signals.log.emit(f"[{done}/{total}] ⏭️ {name} ({info})")
                elif status == "ERROR":
                    errc += 1
                    self.signals.log.emit(f"[{done}/{total}] ❌ {name}\n{info}")
                elif status == "FORCE_STOP":
                    self.signals.log.emit("🛑 Forçar parada.")
                    break
                self.signals.progress.emit(int(done / total * 100))

        self.signals.log.emit("\n=== RESUMO ===")
        self.signals.log.emit(f"✔️ {conv} ⏭️ {skip} ❌ {errc}")
        if conv_per_folder:
            self.signals.log.emit("=== POR PASTA (convertidos) ===")
            for folder_root, qty in conv_per_folder.items():
                label = folder_root if folder_root else "<desconhecida>"
                self.signals.log.emit(f"📁 {label}: {qty} arquivos")
        self.signals.progress.emit(100)
        self.signals.done.emit()

# ===== THREAD DE INVENTÁRIO =====
class InventoryThread(threading.Thread):
    def __init__(self, folders, signals):
        super().__init__()
        self.folders = list(folders)
        self.signals = signals

    def run(self):
        palavras_chave = ["stripalign","lms","temp","out_area","out area",
                          "trj","mms_panoramica","mms panoramica","develop"]
        extensao_las = ".las"
        resultados, pastas_com_las = [], set()

        try:
            for storage in self.folders:
                self.signals.log.emit(f"🔎 Varredura: {storage}")
                for raiz, _, arquivos in os.walk(storage):
                    partes = raiz.replace(storage, '').strip(os.sep).split(os.sep)
                    if len(partes) < 2: continue
                    projeto = partes[0]
                    nome_pasta = partes[-1].strip()
                    nome_lower = nome_pasta.lower()
                    alerta = []
                    if any(p in nome_lower for p in palavras_chave):
                        alerta.append("Pasta crítica")
                    rt_size = None
                    if nome_lower == "rt":
                        rt_size = safe_folder_size_bytes(raiz)
                        if rt_size >= 100 * 1024 * 1024:
                            alerta.append("Pasta crítica")
                    if any(f.lower().endswith(extensao_las) for f in arquivos):
                        alerta.append("Contém LAS")
                        pastas_com_las.add(os.path.normpath(raiz))
                    tamanho_bytes = tamanho_legivel = None
                    if "Pasta crítica" in alerta:
                        tamanho_bytes = rt_size if rt_size is not None else safe_folder_size_bytes(raiz)
                        tamanho_legivel = format_size(tamanho_bytes)
                        # Ignora pastas críticas vazias ou muito pequenas (<1MB)
                        if tamanho_bytes < 1024 * 1024:
                            continue
                    resultados.append({
                        "Storage": storage,
                        "Projeto": projeto,
                        "Subpasta": raiz,
                        "Nível": len(partes),
                        "Alerta": ", ".join(sorted(set(alerta))) if alerta else "",
                        "Tamanho_bytes": tamanho_bytes,
                        "Tamanho_legível": tamanho_legivel or ""
                    })

            df = pd.DataFrame(resultados)
            df_crit = df[df['Alerta'].str.contains("Pasta crítica", case=False, na=False)].copy()

            try:
                base_folder = os.path.basename(os.path.normpath(self.folders[0]))
                if not base_folder or base_folder == os.sep:
                    base_folder = Path(self.folders[0]).parts[-1]
                if not base_folder:
                    base_folder = datetime.now().strftime("inv_%Y%m%d_%H%M%S")
                if not base_folder.lower().startswith("inv_"):
                    base_folder = f"inv_{base_folder}"
                nome_planilha = f"{base_folder}.xlsx"
            except Exception:
                nome_planilha = f"inv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

            caminho_saida = os.path.join(INVENTARIO_DIR, nome_planilha)
            os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
            with pd.ExcelWriter(caminho_saida, engine="xlsxwriter") as xw:
                df_crit.to_excel(xw, index=False, sheet_name="Pastas_criticas")
                df.to_excel(xw, index=False, sheet_name="Todas_as_pastas")

            with open("pastas_com_las.json", "w", encoding="utf-8") as f:
                json.dump(sorted(pastas_com_las), f, indent=2, ensure_ascii=False)

            self.signals.result.emit(caminho_saida, sorted(pastas_com_las))
        finally:
            self.signals.done.emit()

# ===== UI =====
class LASConverterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Inventário e Conversor LAS/LAZ")
        self.setGeometry(300, 300, 820, 660)
        self.tabs = QTabWidget()
        self.layout_main = QVBoxLayout(self)
        self.layout_main.addWidget(self.tabs)
        self.folders = []
        self.worker = None
        self.setup_conversion_tab()
        self.setup_inventory_tab()

    def setup_conversion_tab(self):
        self.tab_convert = QWidget()
        self.tabs.addTab(self.tab_convert, "Conversão")
        layout = QVBoxLayout(self.tab_convert)
        row = QHBoxLayout()
        self.btn_select_folder = QPushButton("Adicionar Pasta")
        self.btn_select_folder.clicked.connect(self.select_folder)
        row.addWidget(self.btn_select_folder)
        row.addWidget(QLabel("Trabalhadores:"))
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, 8)
        self.spin_workers.setValue(DEFAULT_MAX_WORKERS)
        row.addWidget(self.spin_workers)
        layout.addLayout(row)
        self.checkbox_delete = QCheckBox("Excluir arquivos após conversão")
        layout.addWidget(self.checkbox_delete)
        self.btn_las_to_laz = QPushButton("LAS → LAZ")
        self.btn_las_to_laz.clicked.connect(lambda: self.start_conversion('laz'))
        layout.addWidget(self.btn_las_to_laz)
        self.btn_laz_to_las = QPushButton("LAZ → LAS")
        self.btn_laz_to_las.clicked.connect(lambda: self.start_conversion('las'))
        layout.addWidget(self.btn_laz_to_las)
        self.btn_stop = QPushButton("Parar suave")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_conversion)
        layout.addWidget(self.btn_stop)
        self.btn_force_stop = QPushButton("Forçar parada")
        self.btn_force_stop.setEnabled(False)
        self.btn_force_stop.clicked.connect(self.force_stop_conversion)
        layout.addWidget(self.btn_force_stop)
        self.btn_clear = QPushButton("Limpar Pastas")
        self.btn_clear.clicked.connect(self.clear_folders)
        layout.addWidget(self.btn_clear)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

    def setup_inventory_tab(self):
        self.tab_inventory = QWidget()
        self.tabs.addTab(self.tab_inventory, "Inventário")
        layout = QVBoxLayout(self.tab_inventory)
        self.btn_select_inventory = QPushButton("Adicionar Pasta Inventário")
        self.btn_select_inventory.clicked.connect(self.select_inventory_folder)
        layout.addWidget(self.btn_select_inventory)
        self.list_inventory = QListWidget()
        layout.addWidget(self.list_inventory)
        self.btn_run_inventory = QPushButton("Executar Inventário")
        self.btn_run_inventory.clicked.connect(self.run_inventory)
        layout.addWidget(self.btn_run_inventory)
        self.btn_clear_inventory = QPushButton("Limpar Inventário")
        self.btn_clear_inventory.clicked.connect(self.clear_inventory_folders)
        layout.addWidget(self.btn_clear_inventory)
        self.log_inventory = QTextEdit()
        self.log_inventory.setReadOnly(True)
        layout.addWidget(self.log_inventory)

    # Conversão
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta")
        if folder and folder not in self.folders:
            self.folders.append(folder)
            self.output.append(f"📁 {folder}")
        elif folder in self.folders:
            self.output.append(f"⚠️ Já existe: {folder}")

    def clear_folders(self):
        self.folders.clear()
        self.output.append("🪑 Lista de pastas limpa.")

    def log(self, message):
        self.output.append(message)

    def update_progress(self, value):
        self.progress.setValue(value)

    def start_conversion(self, convert_to):
        if not self.folders:
            self.output.append("⚠️ Nenhuma pasta adicionada!")
            return
        max_workers = self.spin_workers.value()
        self.btn_las_to_laz.setEnabled(False)
        self.btn_laz_to_las.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_force_stop.setEnabled(True)
        self.signals = WorkerSignals()
        self.signals.log.connect(self.log)
        self.signals.done.connect(self.conversion_finished)
        self.signals.progress.connect(self.update_progress)
        self.worker = MultiFolderConverterThread(
            self.folders, self.checkbox_delete.isChecked(),
            self.signals, convert_to, max_workers=max_workers, eta_every=DEFAULT_ETA_EVERY
        )
        self.worker.start()
        self.output.append(f"▶️ Iniciando {convert_to.upper()} com {max_workers} trabalhadores...")

    def stop_conversion(self):
        if self.worker: self.worker.stop()

    def force_stop_conversion(self):
        if self.worker: self.worker.force_stop()

    def conversion_finished(self):
        self.output.append("✅ Conversão finalizada.")
        self.btn_las_to_laz.setEnabled(True)
        self.btn_laz_to_las.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_force_stop.setEnabled(False)

    # Inventário
    def select_inventory_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta")
        if folder:
            self.list_inventory.addItem(folder)

    def clear_inventory_folders(self):
        self.list_inventory.clear()
        self.log_inventory.append("🪑 Inventário limpo.")

    def run_inventory(self):
        folders = [self.list_inventory.item(i).text() for i in range(self.list_inventory.count())]
        if not folders:
            self.log_inventory.append("⚠️ Nenhum diretório.")
            return
        self.btn_run_inventory.setEnabled(False)
        self.inv_signals = InventorySignals()
        self.inv_signals.log.connect(self.log_inventory.append)
        self.inv_signals.done.connect(lambda: self.btn_run_inventory.setEnabled(True))
        self.inv_signals.done.connect(lambda: self.log_inventory.append("✅ Inventário finalizado."))
        def on_result(planilha, pastas_las):
            self.log_inventory.append(f"📄 {planilha}")
            self.log_inventory.append("🧾 pastas_com_las.json")
            self.folders = pastas_las
            self.output.append("📌 Pastas carregadas:")
            for p in self.folders:
                self.output.append(f"📁 {p}")
        self.inv_signals.result.connect(on_result)
        th = InventoryThread(folders, self.inv_signals)
        th.start()

# MAIN
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LASConverterApp()
    window.show()
    sys.exit(app.exec_())
