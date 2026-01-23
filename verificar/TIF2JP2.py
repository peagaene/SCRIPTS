import os
import subprocess
from tqdm import tqdm

# ===== VERIFICAÇÃO DO DRIVER JP2OpenJPEG =====
def verificar_driver_jp2():
    try:
        resultado = subprocess.run(["gdal_translate", "--formats"], capture_output=True, text=True)
        if "JP2OpenJPEG" not in resultado.stdout:
            print("\033[91m❌ O driver JP2OpenJPEG não está disponível no GDAL instalado.\033[0m")
            print("Verifique se o GDAL foi compilado com suporte a JP2OpenJPEG ou use outro driver como JPEG2000 (Jasper).")
            exit(1)
        else:
            print("\033[94m🔍 Driver JP2OpenJPEG encontrado e pronto para uso.\033[0m")
    except FileNotFoundError:
        print("\033[91m❌ gdal_translate não encontrado. Verifique se o GDAL está instalado e no PATH do sistema.\033[0m")
        exit(1)

verificar_driver_jp2()


# ===== CONFIGURAÇÕES =====
CONFIG = {
    "dirs_input_tiff": [
        r'\\192.168.2.28\i\5_ORTOMOSAICOS\SI_07\2_IR\1_GEOTIFF',
    ],
    "dir_output_jp2": r'\\192.168.2.28\i\5_ORTOMOSAICOS\SI_07\2_IR\2_JPG2000',
    "compression_quality": 10,
    "relatorio_saida": r'D:\SI_04\relatorio_convertido.txt',
}

# ===== PREPARAÇÃO =====
os.makedirs(CONFIG["dir_output_jp2"], exist_ok=True)
os.makedirs(os.path.dirname(CONFIG["relatorio_saida"]), exist_ok=True)

# ===== COLETAR IMAGENS =====
path_imagens = []
for input_dir in CONFIG["dirs_input_tiff"]:
    for f in os.listdir(input_dir):
        if f.lower().endswith('.tif'):
            path_imagens.append(os.path.join(input_dir, f))

# ===== RELATÓRIOS =====
exportacoes_sucesso = []
exportacoes_falha = []

# ===== PROCESSAMENTO =====
for path_tiff in tqdm(path_imagens, desc="Convertendo TIFF para JP2"):
    try:
        nome_base = os.path.splitext(os.path.basename(path_tiff))[0]
        output_jp2 = os.path.join(CONFIG["dir_output_jp2"], f"{nome_base}.jp2")

        if os.path.exists(output_jp2):
            tqdm.write(f"\033[96m🟡 {nome_base}.jp2 já existe, pulando...\033[0m")
            exportacoes_sucesso.append(nome_base)
            continue

        # Montar o comando GDAL
        comando = [
            "gdal_translate",
            "-of", "JP2OpenJPEG",
            "-co", f"QUALITY={CONFIG['compression_quality']}",
            "-co", "REVERSIBLE=NO",
            path_tiff,
            output_jp2
        ]

        resultado = subprocess.run(comando, capture_output=True, text=True)

        if resultado.returncode == 0:
            tqdm.write(f"\033[92m✅ {nome_base}.jp2 exportado\033[0m")
            exportacoes_sucesso.append(nome_base)
        else:
            raise RuntimeError(resultado.stderr)

    except Exception as e:
        msg = str(e)
        tqdm.write(f"\033[91m❌ Erro ao converter {path_tiff}: {msg}\033[0m")
        exportacoes_falha.append((path_tiff, msg))

# ===== RELATÓRIO FINAL =====
with open(CONFIG["relatorio_saida"], 'w', encoding='utf-8') as relatorio:
    relatorio.write(f"Relatório de Conversão TIFF → JP2\n\n")
    relatorio.write(f"Total de imagens: {len(path_imagens)}\n")
    relatorio.write(f"Conversões com sucesso: {len(exportacoes_sucesso)}\n")
    relatorio.write(f"Falhas: {len(exportacoes_falha)}\n\n")

    relatorio.write("=== Conversões Bem-sucedidas ===\n")
    for sucesso in exportacoes_sucesso:
        relatorio.write(f"- {sucesso}\n")

    relatorio.write("\n=== Conversões com Falha ===\n")
    for falha in exportacoes_falha:
        relatorio.write(f"- {falha[0]}: {falha[1]}\n")

print("\n\033[92m✅ Conversão concluída. Relatório salvo em:\033[0m", CONFIG["relatorio_saida"])
