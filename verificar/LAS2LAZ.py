import os
import subprocess
import threading
import time
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit,
    QFileDialog, QLabel, QCheckBox, QProgressBar
)
from PyQt5.QtCore import pyqtSignal, QObject

# Caminho do LASzip
LASZIP_PATH = r'D:\LAStools\bin\laszip.exe'

# Classe para emitir sinais para a GUI
class WorkerSignals(QObject):
    log = pyqtSignal(str)
    done = pyqtSignal()
    progress = pyqtSignal(int)

class MultiFolderConverterThread(threading.Thread):
    def __init__(self, folder_list, delete_input, signals, convert_to='laz'):
        super().__init__()
        self.folder_list = folder_list
        self.delete_input = delete_input
        self.signals = signals
        self.stop_requested = False
        self.convert_to = convert_to.lower()

    def run(self):
        if self.convert_to == 'laz':
            input_ext = '.las'
            output_ext = '.laz'
        elif self.convert_to == 'las':
            input_ext = '.laz'
            output_ext = '.las'
        else:
            self.signals.log.emit(f"❌ Extensão de conversão inválida: {self.convert_to}")
            self.signals.done.emit()
            return

        input_files = []
        for root_folder in self.folder_list:
            for root, _, files in os.walk(root_folder):
                for f in files:
                    if f.lower().endswith(input_ext):
                        input_files.append(os.path.join(root, f))

        total = len(input_files)
        convertidos = pulados = erros = 0

        self.signals.log.emit(f"🔍 {total} arquivos {input_ext} encontrados em {len(self.folder_list)} pasta(s).\n")

        start_time = time.time()

        for i, input_path in enumerate(input_files, 1):
            if self.stop_requested:
                self.signals.log.emit("🚩 Conversão interrompida pelo usuário.\n")
                break

            output_path = input_path.replace(input_ext, output_ext)
            current_time_str = datetime.now().strftime("%H:%M:%S")

            if os.path.exists(output_path):
                pulados += 1
                self.signals.log.emit(f"[{i}/{total}] ⏭️ Pulado (já existe): {os.path.basename(output_path)}")
                self.signals.progress.emit(int(i / total * 100))
                continue

            command = [LASZIP_PATH, '-i', input_path, '-o', output_path]
            try:
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout, stderr=result.stderr)

                convertidos += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (total - i)
                eta = datetime.now() + timedelta(seconds=remaining)

                self.signals.log.emit(f"[{i}/{total}] ✅ Convertido: {os.path.basename(input_path)} às {current_time_str}")
                self.signals.log.emit(f"   ⏳ Estimativa de término: {eta.strftime('%H:%M:%S')}")

                if self.delete_input:
                    os.remove(input_path)
                    self.signals.log.emit(f"   🗑️ Apagado: {os.path.basename(input_path)}")

            except subprocess.CalledProcessError as e:
                erros += 1
                msg_erro = e.stderr or str(e)
                self.signals.log.emit(f"[{i}/{total}] ❌ ERRO: {os.path.basename(input_path)}\n{msg_erro.strip()}")

            self.signals.progress.emit(int(i / total * 100))

        self.signals.log.emit("\n=== RESUMO ===")
        self.signals.log.emit(f"✔️ Convertidos: {convertidos}")
        self.signals.log.emit(f"⏭️ Pulados: {pulados}")
        self.signals.log.emit(f"❌ Erros: {erros}")
        self.signals.progress.emit(100)
        self.signals.done.emit()

    def stop(self):
        self.stop_requested = True

class LASConverterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Conversor LAS/LAZ")
        self.setGeometry(300, 300, 600, 550)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("Pastas com arquivos .las/.laz:")
        self.layout.addWidget(self.label)

        self.btn_select_folder = QPushButton("Adicionar Pasta")
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.layout.addWidget(self.btn_select_folder)

        self.checkbox_delete = QCheckBox("Excluir arquivos de entrada após conversão")
        self.layout.addWidget(self.checkbox_delete)

        self.btn_las_to_laz = QPushButton("Converter LAS para LAZ")
        self.btn_las_to_laz.clicked.connect(lambda: self.start_conversion('laz'))
        self.layout.addWidget(self.btn_las_to_laz)

        self.btn_laz_to_las = QPushButton("Converter LAZ para LAS")
        self.btn_laz_to_las.clicked.connect(lambda: self.start_conversion('las'))
        self.layout.addWidget(self.btn_laz_to_las)

        self.btn_stop = QPushButton("Parar")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_conversion)
        self.layout.addWidget(self.btn_stop)

        self.btn_clear = QPushButton("Limpar Pastas Selecionadas")
        self.btn_clear.clicked.connect(self.clear_folders)
        self.layout.addWidget(self.btn_clear)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.layout.addWidget(self.progress)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)

        self.folders = []
        self.worker = None

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta")
        if folder and folder not in self.folders:
            self.folders.append(folder)
            self.output.append(f"📁 Pasta adicionada: {folder}")
        elif folder in self.folders:
            self.output.append(f"⚠️ Pasta já está na lista: {folder}")

    def clear_folders(self):
        self.folders.clear()
        self.output.append("🪑 Lista de pastas foi limpa.")

    def log(self, message):
        self.output.append(message)

    def update_progress(self, value):
        self.progress.setValue(value)

    def start_conversion(self, convert_to):
        if not self.folders:
            self.output.append("⚠️ Adicione pelo menos uma pasta primeiro!")
            return

        self.btn_las_to_laz.setEnabled(False)
        self.btn_laz_to_las.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.signals = WorkerSignals()
        self.signals.log.connect(self.log)
        self.signals.done.connect(self.conversion_finished)
        self.signals.progress.connect(self.update_progress)

        self.worker = MultiFolderConverterThread(
            self.folders, self.checkbox_delete.isChecked(), self.signals, convert_to
        )
        self.worker.start()

    def stop_conversion(self):
        if self.worker:
            self.worker.stop()

    def conversion_finished(self):
        self.output.append("✅ Conversão finalizada.")
        self.btn_las_to_laz.setEnabled(True)
        self.btn_laz_to_las.setEnabled(True)
        self.btn_stop.setEnabled(False)


# Execução do app
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = LASConverterApp()
    window.show()
    sys.exit(app.exec_())
