import os
from collections import defaultdict

# Função para formatar o tamanho dos arquivos de forma adaptativa (KB, MB, GB, TB)
def formatar_tamanho(tamanho_bytes):
    if tamanho_bytes < 1024:
        return f"{tamanho_bytes} bytes"
    elif tamanho_bytes < 1024**2:
        return f"{tamanho_bytes / 1024:.2f} KB"
    elif tamanho_bytes < 1024**3:
        return f"{tamanho_bytes / 1024**2:.2f} MB"
    elif tamanho_bytes < 1024**4:
        return f"{tamanho_bytes / 1024**3:.2f} GB"
    else:
        return f"{tamanho_bytes / 1024**4:.2f} TB"

# Função para listar todos os arquivos em um diretório e subdiretórios, agrupados por pasta e extensão
def listar_arquivos_em_HD(diretorio, arquivo_saida):
    arquivos_por_pasta = defaultdict(lambda: defaultdict(list))  # Dicionário para agrupar arquivos por pasta e extensão
    total_tamanho = defaultdict(lambda: defaultdict(int))  # Dicionário para armazenar tamanho total por pasta e extensão
    total_arquivos = 0
    total_pastas = 0

    # Definir extensões a serem ignoradas
    extensoes_ignoradas = {'.atx', '.prj', '.tfw', '.j2w', '.gdbtablx', '.gdbtable', '.gdbindexes', '.freelist', '.horizon', 'timestamps', 'gdb', '.spx', '.subfile', '.subfilx', ' '}

    # Percorre todo o diretório
    with open(arquivo_saida, 'w', encoding='utf-8') as f_out:
        for root, dirs, files in os.walk(diretorio):
            total_pastas += 1
            for file in files:
                extensao = os.path.splitext(file)[1].lower()  # Extensão do arquivo (em minúsculas)

                # Pula as extensões que não devem ser listadas
                if extensao in extensoes_ignoradas:
                    continue

                caminho_completo = os.path.join(root, file)
                tamanho = os.path.getsize(caminho_completo)  # Tamanho do arquivo em bytes
                pasta_relativa = os.path.relpath(root, diretorio)  # Caminho relativo da pasta

                # Armazena o nome do arquivo e atualiza o total por pasta e extensão
                arquivos_por_pasta[pasta_relativa][extensao].append(file)
                total_tamanho[pasta_relativa][extensao] += tamanho
                total_arquivos += 1

        # Escreve as informações no arquivo de saída
        for pasta, extensoes in arquivos_por_pasta.items():
            f_out.write(f"\n--- Pasta: {pasta} ---\n")
            for extensao, arquivos in sorted(extensoes.items()):
                f_out.write(f"\n--- Arquivos com extensão {extensao} ---\n")
                for arquivo in arquivos:
                    f_out.write(f"{arquivo}\n")  # Escreve apenas o nome do arquivo
                # Escreve o total de arquivos e tamanho total por extensão
                f_out.write(f"Total de arquivos com extensão {extensao}: {len(arquivos)}\n")
                f_out.write(f"Tamanho total: {formatar_tamanho(total_tamanho[pasta][extensao])}\n")

        # Escreve o resumo geral no final
        f_out.write(f"\n--- Resumo Geral ---\n")
        f_out.write(f"Total de pastas percorridas: {total_pastas}\n")
        f_out.write(f"Total de arquivos: {total_arquivos}\n")
        tamanho_total_geral = sum(sum(tamanhos.values()) for tamanhos in total_tamanho.values())
        f_out.write(f"Tamanho total de todos os arquivos: {formatar_tamanho(tamanho_total_geral)}\n")

# Definir o diretório do HD e o arquivo de saída
diretorio_HD = r'I:\DEVELOP_SI_02\20230623_F1_12SEN309\RGB'  # Substitua pelo diretório desejado
diretorio_saida = r'K:\SI_02'  # Defina o diretório de saída

# Garante que o diretório de saída existe, se não, cria
if not os.path.exists(diretorio_saida):
    os.makedirs(diretorio_saida)

# Define o caminho completo do arquivo de saída
arquivo_saida = os.path.join(diretorio_saida, "ORTOMOSAICOS_02_05.txt")

# Chamar a função para listar os arquivos e salvar as informações
listar_arquivos_em_HD(diretorio_HD, arquivo_saida)

print(f"As informações foram salvas em {arquivo_saida}")
