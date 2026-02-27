from pathlib import Path

def rename_files(directory):
    for file in Path(directory).glob('*.las'): #ADICIONAR O TIPO DE ARQUIVO QUE DESEJA RENOMEAR
        if file.stem.startswith('ES_L09_K_NPc_C_'): #REGRA PARA RENOMEAR APENAS OS ARQUIVOS QUE COMEÇAM COM "Bloco-I_", É POSSIVEL MUDAR A REGRA
            stem = file.stem
            resto = stem[len("ES_L09_K_NPc_C_"):] #EXTRAI O RESTO DO NOME APÓS "Bloco-I_"
            resto = resto.replace('-', '_') #SUBSTITUI HÍFENS POR UNDERLINE, SE HOUVER, PARA MANTER O PADRÃO DE NOMES
            new_name = f"ES_L09_K_MDT_{resto}.las"    #CONSTRÓI O NOVO NOME COM O PADRÃO DESEJADO
            file.rename(file.parent / new_name) #RENOMEIA O ARQUIVO PARA O NOVO NOME, MANTENDO O MESMO DIRETÓRIO, PODE MUDAR PARA PRINT PARA VER O NOME ANTIGO E O NOVO SEM RENOMEAR OS ARQUIVOS
            #print(f"Renomeado: {file.name} -> {new_name}") #IMPRIME O NOME ANTIGO E O NOVO PARA VERIFICAÇÃO

diretory = rename_files(r'G:\7_MDT\LOTE_09\BLOCO_K\1_LAS\Nova pasta')

