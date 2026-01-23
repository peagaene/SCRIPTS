import os
import shutil

# Caminhos
pasta_las = r"\\192.168.2.27\f\SI_09_NPC_ENTREGA\Nova pasta"
pasta_laz = r"\\192.168.2.27\f\SI_09_NPC_ENTREGA"
pasta_destino = r"\\192.168.2.27\f\SI_09_NPC_ENTREGA\laz_sem_las"

# Cria a pasta de destino se não existir
os.makedirs(pasta_destino, exist_ok=True)

# Coletar nomes base (sem extensão) dos arquivos .las
nomes_las = {os.path.splitext(f)[0] for f in os.listdir(pasta_las) if f.lower().endswith(".laz")}

# Processar arquivos .laz
for nome_arquivo in os.listdir(pasta_laz):
    if nome_arquivo.lower().endswith(".laz"):
        nome_base = os.path.splitext(nome_arquivo)[0]
        if nome_base not in nomes_las:
            caminho_origem = os.path.join(pasta_laz, nome_arquivo)
            caminho_destino = os.path.join(pasta_destino, nome_arquivo)
            shutil.move(caminho_origem, caminho_destino)
            print(f"Movido: {nome_arquivo}")
