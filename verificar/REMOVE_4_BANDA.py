import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

def encontrar_tifs(pastas):
    arquivos = []
    for pasta in pastas:
        for raiz, _, files in os.walk(pasta):
            for f in files:
                if f.lower().endswith(".tif"):
                    arquivos.append(os.path.join(raiz, f))
    return arquivos

def processar_imagem(caminho):
    nome = os.path.basename(caminho)
    pasta = os.path.dirname(caminho)
    temp = os.path.join(pasta, f"__tmp_{nome}")

    try:
        with rasterio.open(caminho) as src:
            if src.count != 4:
                return f"⚠️ {nome} não possui 4 bandas. Pulando."

            perfil = src.profile.copy()
            perfil.update(count=3, nodata=None, alpha=False)

            with rasterio.open(temp, "w", **perfil) as dst:
                for _, window in src.block_windows(1):
                    r, g, b = src.read((1, 2, 3), window=window)
                    a = src.read(4, window=window)

                    # Máscara de transparência onde alpha > 0
                    mask = (a > 0).astype("uint8") * 255

                    dst.write(r, 1, window=window)
                    dst.write(g, 2, window=window)
                    dst.write(b, 3, window=window)
                    dst.write_mask(mask, window=window)

        os.remove(caminho)
        os.rename(temp, caminho)
        return f"✅ {nome} convertido com sucesso."

    except Exception as e:
        if os.path.exists(temp):
            os.remove(temp)
        return f"❌ Erro ao processar {nome}: {e}"

def processar_todas(pastas):
    arquivos = encontrar_tifs(pastas)
    print(f"🔍 {len(arquivos)} imagens encontradas.")
    for caminho in tqdm(arquivos, desc="Processando imagens"):
        print(processar_imagem(caminho))

# ===== USO =====
if __name__ == "__main__":
    pastas = [
        r"\\192.168.2.25\h\09\RGB\Nova pasta",
    ]
    processar_todas(pastas)
