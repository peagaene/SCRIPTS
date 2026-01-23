# Função para ler o arquivo de saída e listar apenas os arquivos com extensão .tif
def listar_arquivos_tif(arquivo_entrada):
    lista_tif = []
    
    # Abre o arquivo de entrada e lê linha por linha
    with open(arquivo_entrada, 'r', encoding='utf-8') as f_in:
        linhas = f_in.readlines()

    # Varre as linhas para buscar nomes de arquivos com extensão .tif
    for i in range(len(linhas)):
        if linhas[i].startswith("Nome:") and linhas[i].strip().endswith(".tif"):
            nome_arquivo = linhas[i].split("Nome: ")[1].strip()  # Extrai apenas o nome do arquivo
            lista_tif.append(nome_arquivo)
    
    return lista_tif

# Função para exportar a lista de arquivos .tif para um arquivo .txt
def exportar_lista_tif(lista_tif, arquivo_saida):
    with open(arquivo_saida, 'w', encoding='utf-8') as f_out:
        for arquivo in lista_tif:
            f_out.write(arquivo + "\n")
    print(f"A lista de arquivos .tif foi exportada para {arquivo_saida}")

# Caminho do arquivo de entrada e saída
arquivo_entrada = r'E:\2212_GOV_SAO_PAULO\RESUMO_HD\BE21\IMAGENS_SI08.txt'  # Substitua pelo caminho correto
arquivo_saida_tif = r'E:\2212_GOV_SAO_PAULO\RESUMO_HD\BE21\lista_arquivos_tif2.txt'  # Substitua pelo caminho de saída desejado

# Chamar a função para listar os arquivos .tif
arquivos_tif = listar_arquivos_tif(arquivo_entrada)

# Chamar a função para exportar a lista de arquivos .tif para um arquivo .txt
exportar_lista_tif(arquivos_tif, arquivo_saida_tif)
