import os

# Caminho completo para o diretório onde os arquivos estão localizados
caminho_do_diretorio = r'E:\03 - DEVELOP\RGB\20231108_F1_12SEN309'

# Lista os arquivos no diretório
arquivos = os.listdir(caminho_do_diretorio)

# Extrai os nomes dos arquivos sem as extensões
nomes_arquivos = [os.path.splitext(nome)[0] for nome in arquivos if os.path.isfile(os.path.join(caminho_do_diretorio, nome))]

# Gera o conteúdo para o arquivo de texto
conteudo_arquivo = "\n".join(nomes_arquivos)

# Caminho completo para o arquivo de saída
caminho_arquivo_saida = os.path.join(caminho_do_diretorio, 'nomes_arquivos.csv')

# Escreve o conteúdo no arquivo de saída
with open(caminho_arquivo_saida, 'w') as arquivo_saida:
    arquivo_saida.write(conteudo_arquivo)

print("Arquivo de nomes de arquivos (sem extensões) gerado: nomes_arquivos.csv")
