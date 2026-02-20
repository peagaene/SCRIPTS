import os

caminho = r"\\192.168.2.252\Apollo_R\2022_IGC_SP22\01 ORIGINAIS\BLOCO_SI_01"

total = 0

print("\n=========== CONTAGEM POR PASTA ===========\n")

for raiz, pastas, arquivos in os.walk(caminho):
    if "RGB" in raiz.upper():

        arquivos_iiq = [
            arq for arq in arquivos
            if arq.lower().endswith(".iiq")
        ]

        if arquivos_iiq:
            print(f"{raiz} -> {len(arquivos_iiq)} arquivos .iiq")
            total += len(arquivos_iiq)

print("\nTOTAL DE .iiq:", total)