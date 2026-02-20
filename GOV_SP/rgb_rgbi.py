import os
import re

# ================================
# CAMINHOS
# ================================
pasta_1 = r"\\192.168.2.28\h\IMAGENS_RGBI_SI01"
caminho_2 = [r"\\192.168.2.28\g\DEVELOP_SI_01\20230804_F1_12SEN309\RGB", r"\\192.168.2.28\f"]

# ================================
# EXTRAÇÃO DOS NÚMEROS
# ================================

def extrair_ft(nome):
    match = re.search(r"FT(\d+)", nome)
    return match.group(1) if match else None

def extrair_cap(nome):
    match = re.search(r"cap-(\d+)", nome, re.IGNORECASE)
    return match.group(1) if match else None

# ================================
# NÚMEROS DA PASTA 1
# ================================
fts_pasta_1 = set()

for arq in os.listdir(pasta_1):
    ft = extrair_ft(arq)
    if ft:
        fts_pasta_1.add(ft)

print("\nArquivos válidos na Pasta 1:", len(fts_pasta_1))

# ================================
# PROCURA NAS PASTAS RGB
# ================================
faltando_na_pasta_1 = []
rgb = set()

print("\n=========== VERIFICAÇÃO RGB ===========\n")

for base in caminho_2:  # <- percorre cada diretório da lista
    for raiz, pastas, arquivos in os.walk(base):

        if "RGB" in os.path.basename(raiz).upper():
            print(f"Analisando: {raiz}")

            for arq in arquivos:
                cap = extrair_cap(arq)
                rgb.add(cap)
            
                if cap and cap not in fts_pasta_1:
                    faltando_na_pasta_1.append((cap, arq, raiz))

print("\nArquivos válidos na Pasta 2:", len(rgb))

# ================================
# RELATÓRIO
# ================================
print("\n=========== RESULTADO ===========\n")

if faltando_na_pasta_1:

    print("Presentes na Pasta 2 (RGB) mas AUSENTES na Pasta 1:\n")

    for cap, arq, raiz in faltando_na_pasta_1:
        print(f"   Arquivo: {arq}")
        print(f"   Pasta   : {raiz}\n")

else:
    print("Nenhuma inconsistência encontrada ✔")

print("\n========================================")