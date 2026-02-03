from pathlib import Path

def listar_pastas_para_txt(caminho_base, pasta_saida):
    caminho = Path(caminho_base)
    pasta_saida = Path(pasta_saida)
    arquivo_saida = pasta_saida / "msa1050_6.txt"

    if not caminho.exists():
        print(f"Caminho não encontrado: {caminho}")
        return

    pasta_saida.mkdir(parents=True, exist_ok=True)

    pastas = [p.name for p in caminho.iterdir() if p.is_dir()]

    if not pastas:
        print("Nenhuma pasta encontrada.")
        return

    with open(arquivo_saida, "w", encoding="utf-8") as f:
        for pasta in pastas:
            f.write(pasta + "\n")

    print(f"Arquivo gerado com sucesso em: {arquivo_saida}")


# EXEMPLO DE USO
if __name__ == "__main__":
    caminho_origem = r"\\192.168.2.239\msa1050_6"
    pasta_saida = r"D:\00_Pedro\INVENTARIO_STORAGE"

    listar_pastas_para_txt(caminho_origem, pasta_saida)
