import os

# Caminho da pasta onde estão os arquivos
pasta = r'F:\5_ORTOMOSAICO\SI_05\IR'

# Pergunta ao operador se as imagens são RGB ou IR
tipo_ortomosaico = input("Os ortomosaicos são RGB ou IR? Digite 'RGB' ou 'IR': ").strip().upper()

# Verifica se a entrada é válida
if tipo_ortomosaico not in ['RGB', 'IR']:
    print("Tipo inválido. Por favor, execute novamente e digite 'RGB' ou 'IR'.")
else:
    sufixo = "_ORTO_RGB" if tipo_ortomosaico == 'RGB' else "_ORTO_IR"
    
    # Loop para percorrer todos os arquivos da pasta
    for nome_arquivo in os.listdir(pasta):
        # Caminho completo do arquivo
        caminho_antigo = os.path.join(pasta, nome_arquivo)
        
        # Verifica se é um diretório, se for, ignora
        if os.path.isdir(caminho_antigo):
            continue
        
        # Obtém o nome do arquivo e a extensão separadamente
        nome, extensao = os.path.splitext(nome_arquivo)
        
        # Remove o caractere "_" do início, se houver
        if nome.startswith('_'):
            nome = nome[1:]

        # Verifica se o nome já possui o sufixo
        if nome.endswith(sufixo):
            print(f'Arquivo já possui o sufixo: {nome_arquivo} -> {nome_arquivo}')
            continue
        
        # Cria o novo nome do arquivo adicionando o sufixo adequado antes da extensão
        novo_nome = f"{nome}{sufixo}{extensao}"
        
        # Caminho completo do novo arquivo
        caminho_novo = os.path.join(pasta, novo_nome)
        
        # Renomeia o arquivo
        os.rename(caminho_antigo, caminho_novo)
        print(f'Arquivo renomeado: {nome_arquivo} -> {novo_nome}')

    print("Renomeação concluída.")
