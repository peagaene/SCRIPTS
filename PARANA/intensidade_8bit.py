from __future__ import annotations

import argparse
from pathlib import Path

import rasterio


def convert_to_1band(input_path: Path, output_path: Path) -> None:
    with rasterio.open(input_path) as src:
        if src.count not in (3, 4):
            raise ValueError(
                f"Esperado 3 ou 4 bandas, mas encontrei {src.count} em: {input_path}"
            )

        profile = src.profile.copy()
        profile.update(count=1)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **profile) as dst:
            # Para imagens com 4 bandas, a ultima e descartada.
            # Como as bandas RGB sao identicas, gravamos apenas a banda 1.
            for _, window in src.block_windows(1):
                data = src.read(1, window=window)
                dst.write(data, 1, window=window)


def list_tiffs(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    # Remove duplicidades caso o sistema de arquivos seja case-insensitive.
    seen = set()
    unique = []
    for f in files:
        k = str(f.resolve()).lower()
        if k not in seen:
            seen.add(k)
            unique.append(f)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Converte TIFFs de 3 ou 4 bandas (com valores identicos entre bandas RGB) "
            "em TIFFs de 1 banda."
        )
    )
    parser.add_argument("input_folder", type=Path, help="Pasta com arquivos .tif/.tiff")
    parser.add_argument(
        "-o",
        "--output-folder",
        type=Path,
        default=None,
        help="Pasta de saida. Se nao informado, usa uma subpasta '1banda' dentro da pasta de entrada.",
    )

    args = parser.parse_args()
    input_folder: Path = args.input_folder

    if not input_folder.exists() or not input_folder.is_dir():
        raise FileNotFoundError(f"Pasta de entrada invalida: {input_folder}")

    output_folder = args.output_folder or (input_folder / "1banda")
    output_folder.mkdir(parents=True, exist_ok=True)

    tiffs = list_tiffs(input_folder)
    if not tiffs:
        print(f"Nenhum .tif/.tiff encontrado em: {input_folder}")
        return

    converted = 0
    skipped = 0

    for tif in tiffs:
        out_tif = output_folder / tif.name
        try:
            convert_to_1band(tif, out_tif)
            converted += 1
            print(f"[OK] {tif.name} -> {out_tif}")
        except Exception as exc:
            skipped += 1
            print(f"[ERRO] {tif.name}: {exc}")

    print(f"\nConcluido. Convertidos: {converted} | Com erro: {skipped}")


if __name__ == "__main__":
    main()
