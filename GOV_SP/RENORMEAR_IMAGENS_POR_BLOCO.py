import os

# Caminho da pasta onde estão os arquivos
pasta = r'\\192.168.2.26\g\IMAGENS_RGBI_SI15\20231214\CIR\4 band CIR'

# Loop para percorrer todos os arquivos da pasta
for nome_arquivo in os.listdir(pasta):
    # Verifica se o arquivo começa com "SI_10_"
    if nome_arquivo.startswith("SI_11_"):
        # Cria o novo nome do arquivo alterando "SI_10_" para "SI_08_"
        novo_nome = nome_arquivo.replace("SI_11_", "SI_15_", 1)
        
        # Caminho completo dos arquivos antigo e novo
        caminho_antigo = os.path.join(pasta, nome_arquivo)
        caminho_novo = os.path.join(pasta, novo_nome)
        
        # Renomeia o arquivo
        os.rename(caminho_antigo, caminho_novo)
        print(f'Arquivo renomeado: {nome_arquivo} -> {novo_nome}')

print("Renomeação concluída.")
