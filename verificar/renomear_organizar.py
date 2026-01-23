import os
import shutil

# Diretórios raiz com os blocos a processar
bases = [
    r'\\192.168.2.27\h',
    r'\\192.168.2.28\g'
]

# Nome do diretório de destino padronizado
nome_base_saida = "5_ORTOMOSAICOS"

# Mapas de nome para destino
mapa_espectro = {'RGB': '1_RGB', 'IR': '2_IR'}
mapa_extensao = {'.tif': '1_GEOTIFF', '.tiff': '1_GEOTIFF', '.jp2': '2_JPG2000'}

# Percorre todas as bases
for base in bases:
    for raiz, dirs, files in os.walk(base):
        raiz_upper = raiz.upper()

        # Ignora pastas da Lixeira e a pasta de saída
        if any(palavra in raiz_upper for palavra in ['RECYCLE', 'TRASH', nome_base_saida.upper()]):
            continue

        for nome_arquivo in files:
            extensao = os.path.splitext(nome_arquivo)[1].lower()
            if extensao not in mapa_extensao:
                continue  # pula arquivos que não são tif/jp2

            caminho_origem = os.path.join(raiz, nome_arquivo)

            # Define o espectro com base nos nomes das pastas RGB ou IR
            partes = raiz.split(os.sep)
            espectro = next((e for e in ['RGB', 'IR'] if e in partes), None)
            if not espectro:
                print(f"❌ Espectro não identificado em: {caminho_origem}")
                continue

            # Detecta o nome do bloco (ex: SI_12)
            bloco = next((p for p in partes if p.upper().startswith('SI_')), None)
            if not bloco:
                print(f"❌ Bloco não identificado para: {caminho_origem}")
                continue

            sufixo = '_ORTO_RGB' if espectro == 'RGB' else '_ORTO_IR'
            nome_sem_ext = os.path.splitext(nome_arquivo)[0].lstrip('_')

            if not nome_sem_ext.endswith(sufixo):
                nome_sem_ext += sufixo

            novo_nome = f"{nome_sem_ext}{extensao}"

            # Destino formatado
            pasta_destino = os.path.join(
                os.path.dirname(base), nome_base_saida,
                bloco,
                mapa_espectro[espectro],
                mapa_extensao[extensao]
            )
            os.makedirs(pasta_destino, exist_ok=True)

            caminho_destino = os.path.join(pasta_destino, novo_nome)

            # Verifica se já existe no destino
            if os.path.exists(caminho_destino):
                print(f"⚠️ Arquivo já existe no destino, pulando: {novo_nome}")
                continue

            # Move o arquivo
            shutil.move(caminho_origem, caminho_destino)
            print(f"✅ {nome_arquivo} → {os.path.relpath(caminho_destino, os.path.dirname(base))}")

print("\n✅ Organização concluída.")
