import os
import shutil
import pandas as pd
from tqdm import tqdm

#Bloco
bloco = '09'

#Dir LISTA RGBI
dir_fotos =r'\\192.168.2.28\d\RGBI_09_TESTE\FAIXA_120\CIR\4 band CIR'

#dir comb
dir_comb = r'\\192.168.2.27\d\URGENTE\COMB\SI_09'

comb_files_dir = [f'{dir_comb}\{filename}' for filename in os.listdir(dir_comb) if filename.lower().endswith('.xlsx')]

df = pd.DataFrame()
for i in range(len(comb_files_dir)):
    dfj = pd.read_excel(comb_files_dir[i], sheet_name = 'Fotos') #Ou 'Sheet1' Fotos
    df = pd.concat([df, dfj])

df = df.dropna(subset=["Faixa"])
df.to_excel('D:/Combinado.xlsx')

#Preparar os dados
img_analisada = df['Imagem analisada'].tolist()
nome_faixa = df['Faixa'].tolist()

faixa = [filename.split("_")[1] for filename in nome_faixa]
faixa_format =  [f'00{filename}' if float(filename) < 10 else f'0{filename}' if float(filename) < 100 else str(filename) for filename in faixa]

num_foto = [filename[4:] for filename in img_analisada]

img_cap_rgbi = [f'{filename}_rgbi.tif' for filename in img_analisada]
#img_cap_rgbi = [f'{filename}.tif' for filename in img_analisada]
print(img_cap_rgbi)

novo_nome = [f'SI_{bloco}_F23_VF_FX{faixa_format[i]}_FT{num_foto[i]}.tif' for i in range(len(num_foto))]

df['Novo Nome'] = novo_nome

#Mudar o nome das imagens
dir_nome_antigo = [f'{dir_fotos}\{filename}' for filename in img_cap_rgbi]
dir_novo_nome = [f'{dir_fotos}\{filename}' for filename in novo_nome]

df['dir_antigo'] = dir_nome_antigo
df['dir_novo'] = dir_novo_nome

print(dir_nome_antigo[701])
print(dir_novo_nome[701])
#df.to_excel(r'D:\teste.xlsx', engine='xlsxwriter')
#print(df)

#Listar os arquivos TIF da pasta
tif_name = [filename[:-9] for filename in os.listdir(dir_fotos) if filename.lower().endswith('.tif')]
#tif_name = [filename[:-4] for filename in os.listdir(dir_fotos) if filename.lower().endswith('.tif')]

#print(tif_name)

for i in tqdm(range(len(dir_novo_nome)), desc = 'Processando'):
    if img_analisada[i] in tif_name:
        shutil.move(dir_nome_antigo[i], dir_novo_nome[i])
        