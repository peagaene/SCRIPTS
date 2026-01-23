import pandas as pd
from tqdm import tqdm  # Para criar a barra de progresso

# Definir os caminhos dos arquivos diretamente
caminho_arquivo1 = r'E:\2212_GOV_SAO_PAULO\VOO\SI_16\SI_16_EO.txt'  # Exemplo: 'C:/meus_arquivos/arquivo1.txt'
caminho_arquivo2 = r'D:\SI_16\20240303_F1_12SEN309.txt'  # Exemplo: 'C:/meus_arquivos/arquivo2.txt'
caminho_saida = r'D:\SI_16\Lista_nova.txt'  # Exemplo: 'C:/meus_arquivos/arquivo_atualizado.txt'

print("Carregando os arquivos...")

# Carregar os arquivos txt
df1 = pd.read_csv(caminho_arquivo1, sep='\t')  # Arquivo com a coluna 'Imagem'
df2 = pd.read_csv(caminho_arquivo2, sep='\t')  # Arquivo com a coluna 'NomeImagem'

print("Arquivos carregados com sucesso!")

# Criar uma nova coluna com os últimos 6 caracteres
df1['Ultimos6'] = df1['Imagem'].apply(lambda x: x[-6:])
df2['Ultimos6'] = df2['NomeImagem'].apply(lambda x: x[-6:])

# Iniciar a barra de progresso
print("Iniciando o processo de comparação...")

# Usar tqdm para criar uma barra de progresso
# Realiza a junção (merge) entre df1 e df2 com base nos últimos 6 caracteres
df_merge = pd.merge(df1, df2.drop(columns=['NomeImagem']), on='Ultimos6', how='left', suffixes=('', '_df2'))

# Manter a coluna 'Imagem' e todas as colunas de df2, exceto 'NomeImagem'
colunas_saida = ['Imagem'] + [col for col in df2.columns if col != 'NomeImagem']
df_saida = df_merge[colunas_saida]

# Remover a coluna auxiliar 'Ultimos6' antes de salvar
# df_saida.drop(columns=['Ultimos6'], inplace=True)  # Não é necessário, pois não foi incluída

# Salvar o arquivo atualizado
df_saida.to_csv(caminho_saida, sep='\t', index=False)

print(f"Processo concluído! Arquivo atualizado salvo em: {caminho_saida}")
