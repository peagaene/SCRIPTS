import os
import sys
import subprocess

# GUI
try:
    import PySimpleGUI as sg
    _GUI_LIB = "psg"
except Exception:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    _GUI_LIB = "tk"

# Caminho do script principal
MAIN_SCRIPT = "LAS_CROP_GUI.py"
DEFAULT_ARTICULATION_SHP = r"\\192.168.2.28\i\80225_PROJETO_IAT_PARANA\5 Processamento Laser\Articulacao_Auxiliar_5000+50.shp"

class MultiBlockGUI:
    def __init__(self):
        self.blocks = []          # lista com IDs inteiros dos blocos ativos
        self.window = None

    def create_block_layout(self, block_num):
        """Retorna o layout (lista de listas) para um bloco, dentro de um Frame com key única."""
        return [
            [sg.Frame(
                f"Bloco {block_num}",
                [
                    [sg.Text("Lote:"), sg.Input(key=f"lote_{block_num}", size=(8,1)),
                     sg.Text("Bloco:"), sg.Input(key=f"bloco_{block_num}", size=(8,1)),
                     sg.Button("🗑️ Remover", key=f"remove_{block_num}", button_color=("white", "#cc0000"))],
                    [sg.Text("Pasta Default:"), sg.Input(key=f"out_default_{block_num}", size=(60,1)),
                     sg.FolderBrowse("Browse", key=f"browse_default_{block_num}")],
                    [sg.Text("Pasta Classificados:"), sg.Input(key=f"out_classified_{block_num}", size=(60,1)),
                     sg.FolderBrowse("Browse", key=f"browse_classified_{block_num}")],
                ],
                key=f"frame_{block_num}",
                relief=sg.RELIEF_GROOVE, border_width=2
            )]
        ]

    def build_layout(self):
        return [
            [sg.Text("Corte LAZ → LAS por articulação - Múltiplos Blocos", font=("Arial", 16, "bold"))],
            [sg.HSeparator()],
            [sg.Text("Pasta de entrada (.laz 1 km²)"), sg.Input(key="in_folder"), sg.FolderBrowse("Browse")],
            [sg.Text("Shapefile de articulação:"), sg.Input(key="art_shp", default_text=DEFAULT_ARTICULATION_SHP, size=(60,1)),
             sg.FileBrowse("Browse", file_types=(("Shapefile", "*.shp"),))],
            [sg.HSeparator()],
            [sg.Text("GERENCIAMENTO DE BLOCOS", font=("Arial", 12, "bold"))],
            [sg.Button("➕ Adicionar Bloco", key="add_block", size=(15,1)),
             sg.Button("🗑️ Limpar Todos", key="clear_blocks", size=(15,1), button_color=("white", "#cc0000"))],
            # Coluna onde os blocos serão inseridos dinamicamente
            [sg.Column([[]], key="blocks_column", scrollable=True, vertical_scroll_only=True, size=(900, 400))],
            [sg.HSeparator()],
            [sg.Text("CONFIGURAÇÕES GERAIS", font=("Arial", 12, "bold"))],
            [sg.Text("EPSG:"), sg.Input(key="epsg", size=(8,1), default_text="31982"),
             sg.Text("Buffer (m):"), sg.Input(key="buffer", size=(8,1), default_text="0"),
             sg.Checkbox("Sobrescrever", key="overwrite")],
            [sg.Checkbox("Converter LAS para LAZ", key="convert_to_laz")],
            [sg.Text("Workers:"), sg.Spin(values=[1,2,3,4,5,6,7,8], initial_value=4, key="workers"),
             sg.Checkbox("Processamento em paralelo", key="parallel")],
            [sg.HSeparator()],
            [sg.Button("🚀 EXECUTAR", key="execute", size=(20,2), button_color=("white", "#2e7d32")),
             sg.Button("❌ Fechar", key="close", size=(20,2), button_color=("white", "#cc0000"))],
            [sg.Multiline(size=(120,15), key="log", autoscroll=True, reroute_stdout=True, reroute_stderr=True)]
        ]

    def _next_block_id(self):
        """Retorna um ID incremental para o bloco."""
        return (max(self.blocks) + 1) if self.blocks else 1

    def add_block(self):
        """Cria e insere visualmente um novo bloco."""
        block_num = self._next_block_id()
        self.blocks.append(block_num)

        new_block_layout = self.create_block_layout(block_num)
        # Insere dinamicamente o layout dentro da Column
        self.window.extend_layout(self.window["blocks_column"], new_block_layout)
        self.window.refresh()

    def remove_block(self, block_num):
        """Esconde o frame do bloco e remove o ID da lista ativa."""
        if block_num in self.blocks:
            # Esconde visualmente
            frame_key = f"frame_{block_num}"
            if frame_key in self.window.AllKeysDict:
                self.window[frame_key].update(visible=False)
            # Remove da lista ativa
            self.blocks.remove(block_num)

    def clear_blocks(self):
        """Esconde todos os frames e limpa a lista de blocos."""
        for block_num in list(self.blocks):
            frame_key = f"frame_{block_num}"
            if frame_key in self.window.AllKeysDict:
                self.window[frame_key].update(visible=False)
        self.blocks.clear()

    def collect_blocks(self, values):
        """Coleta os blocos ainda ativos (visíveis) com todos os campos preenchidos."""
        blocks = []
        for block_num in self.blocks:
            lote = (values.get(f"lote_{block_num}") or "").strip()
            bloco = (values.get(f"bloco_{block_num}") or "").strip()
            out_default = (values.get(f"out_default_{block_num}") or "").strip()
            out_classified = (values.get(f"out_classified_{block_num}") or "").strip()
            if lote and bloco and out_default and out_classified:
                blocks.append({
                    'lote': lote,
                    'bloco': bloco,
                    'out_default': out_default,
                    'out_classified': out_classified
                })
        return blocks

    def execute_blocks(self, values):
        """Executa cada bloco sequencialmente chamando o script principal."""
        blocks = self.collect_blocks(values)

        if not blocks:
            sg.popup_error("Configure pelo menos um bloco completo!", keep_on_top=True)
            return

        in_folder = (values.get("in_folder") or "").strip()
        if not in_folder:
            sg.popup_error("Selecione a pasta de entrada!", keep_on_top=True)
            return

        art = (values.get("art_shp") or DEFAULT_ARTICULATION_SHP).strip()
        epsg = str(values.get("epsg") or "31982")
        buffer_m = str(values.get("buffer") or "0")
        workers = str(values.get("workers") or "4")

        for i, block in enumerate(blocks, 1):
            try:
                print(f"\n🔄 PROCESSANDO BLOCO {i}/{len(blocks)}: LOTE={block['lote']}, BLOCO={block['bloco']}")
                cmd = [
                    sys.executable, MAIN_SCRIPT,
                    "--in", in_folder,
                    "--art", art,
                    "--out-default", block['out_default'],
                    "--out-classified", block['out_classified'],
                    "--lote", block['lote'],
                    "--bloco", block['bloco'],
                    "--epsg", epsg,
                    "--buffer", buffer_m,
                    "--workers", workers
                ]
                if values.get("overwrite"):
                    cmd.append("--overwrite")
                if values.get("convert_to_laz"):
                    cmd.append("--convert-to-laz")

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    if result.stdout:
                        print(result.stdout.strip())
                    print(f"✅ Bloco {i} concluído com sucesso!")
                else:
                    if result.stdout:
                        print(result.stdout.strip())
                    print(f"❌ Erro no Bloco {i}:\n{result.stderr.strip()}")

            except Exception as e:
                print(f"❌ Erro no Bloco {i}: {e}")

        sg.popup("✅ Todos os blocos foram processados!", keep_on_top=True)


    def run(self):
        layout = self.build_layout()
        self.window = sg.Window(
            "LAS Crop - Múltiplos Blocos",
            layout,
            finalize=True,
            resizable=True
        )

        # Adiciona um bloco inicial
        self.add_block()

        while True:
            event, values = self.window.read()
            if event in (sg.WINDOW_CLOSED, "close"):
                break

            if event == "add_block":
                try:
                    self.add_block()
                except Exception as e:
                    sg.popup_error(f"Erro ao adicionar bloco: {e}", keep_on_top=True)

            elif event == "clear_blocks":
                if sg.popup_yes_no("Tem certeza que deseja remover todos os blocos?", keep_on_top=True) == "Yes":
                    self.clear_blocks()

            elif isinstance(event, str) and event.startswith("remove_"):
                try:
                    block_num = int(event.split("_")[1])
                    if sg.popup_yes_no(f"Remover Bloco {block_num}?", keep_on_top=True) == "Yes":
                        self.remove_block(block_num)
                except Exception as e:
                    sg.popup_error(f"Erro ao remover: {e}", keep_on_top=True)

            elif event == "execute":
                try:
                    self.execute_blocks(values)
                except Exception as e:
                    sg.popup_error(f"❌ Erro: {e}", keep_on_top=True)

        self.window.close()

def main():
    if _GUI_LIB == "psg":
        app = MultiBlockGUI()
        app.run()
    else:
        # Fallback simples
        root = tk.Tk()
        root.title("LAS Crop - Múltiplos Blocos")
        messagebox.showinfo("Info", "PySimpleGUI não disponível. Use o script principal LAS_CROP_GUI.py")
        root.destroy()

if __name__ == "__main__":
    main()
