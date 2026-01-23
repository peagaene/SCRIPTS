import os
import shutil

# Defina o caminho da pasta de origem e a pasta de destino
pasta_origem = r'H:\20230608\CIR\4 band CIR'
        
pasta_destino = r'E:\SI_07_IMAGENS'

# Ler o arquivo de nomes
with open('BLOCO 07.txt', 'r') as arquivo_nomes:
    nomes = arquivo_nomes.read().splitlines()

# Listar arquivos na pasta de origem
    arquivos_na_pasta_origem = os.listdir(pasta_origem)

# Criar uma lista para armazenar os nomes de arquivos não encontrados
nomes_nao_encontrados = []

# Iterar sobre os nomes e procurar os arquivos correspondentes
for nome in nomes:
    # Procurar por correspondência ignorando a extensão
    nome_sem_extensao = os.path.splitext(nome)[0]
    arquivo_correspondente = next((arquivo for arquivo in arquivos_na_pasta_origem if nome_sem_extensao in arquivo), None)

    if arquivo_correspondente:
        # Construir o caminho completo do arquivo na pasta de origem e destino
        caminho_arquivo_origem = os.path.join(pasta_origem, arquivo_correspondente)
        caminho_arquivo_destino = os.path.join(pasta_destino, arquivo_correspondente)

        # Copiar o arquivo para a pasta de destino
        shutil.copy(caminho_arquivo_origem, caminho_arquivo_destino)
    else:
        # Se o arquivo não for encontrado, adicione o nome à lista de não encontrados
        nomes_nao_encontrados.append(nome)
        
# Criar um arquivo de texto com os nomes não encontrados
with open('nomes_nao_encontrados.txt', 'w') as arquivo_nao_encontrado:
    for nome in nomes_nao_encontrados:
        arquivo_nao_encontrado.write(nome + '\n')
     
print("Concluído!")
