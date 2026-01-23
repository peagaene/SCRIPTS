import geopandas as gpd
import pandas as pd

# Carregar o arquivo DBF usando o geopandas
caminho_arquivo = r'D:\2212_GOV_SAO_PAULO\VOO\SI_06\TPHOTO\shp\NOME_IMAGEM_SI_06.dbf' #AQUI VOCE MUDA COLOCANDO O CAMINHO CERTO E O NOME DO BLOCO MANTENDO O >DBF
dados = gpd.read_file(caminho_arquivo)

# Visualizar os dados para entender a estrutura e as colunas disponíveis
print(dados.head())

# Extrair apenas os números entre o '-' e '_' na coluna 'NAME' e verificar se são ímpares
def extrair_e_verificar_impar(nome):
    partes = nome.split('-')  # Divide o texto pelo '-'
    if len(partes) > 1:
        numeros = partes[1].split('_')  # Pega a segunda parte e divide pelo '_'
        if len(numeros) > 0:
            numero = int(numeros[0])  # Pega o primeiro número após o '-'
            return numero % 2 != 0  # Verifica se é ímpar
    return False  # Se não encontrar o padrão desejado, consideramos como falso


# Aplicar a função para verificar ímpares na coluna 'NAME' e criar um novo DataFrame com os resultados
dados['Impar'] = dados['NAME'].apply(extrair_e_verificar_impar)

# Filtrar os dados para manter apenas as linhas onde o número final é ímpar
dados_impares = dados[dados['Impar']]

# Salvar o novo shapefile
caminho_novo_shapefile = r'D:\2212_GOV_SAO_PAULO\VOO\SI_06\TPHOTO\shp\NOME_IMAGEM_IMPAR_SI_06' #AQUI VOCE MUDA COLOCANDO O DIRETORIO E O NOME DO ARQUIVO QUE SERA GERADo
dados_impares.to_file(caminho_novo_shapefile)
