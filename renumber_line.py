# requer: pip install laspy lazrs numpy
from pathlib import Path
import argparse
import numpy as np
import laspy


def atribuir_line_por_arquivo(input_dir, output_dir, start=1):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(input_dir.glob("*.laz"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum .laz encontrado em: {input_dir}")

    for i, arq in enumerate(arquivos, start=start):
        las = laspy.read(arq)
        n = len(las.points)

        if i > 65535:
            raise ValueError(
                "point_source_id suporta no maximo 65535. "
                "Use --start menor ou divida em lotes."
            )

        # Campo padrao LAS geralmente interpretado como line number/flightline.
        # Todos os pontos do arquivo recebem o mesmo valor.
        las.point_source_id = np.full(n, i, dtype=np.uint16)

        out_path = output_dir / arq.name
        las.write(out_path)
        print(f"{arq.name}: point_source_id={i} ({n} pontos)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define point_source_id unico por arquivo LAZ em ordem crescente."
    )
    parser.add_argument("--input", required=True, help="Pasta com .laz")
    parser.add_argument("--output", required=True, help="Pasta de saida")
    parser.add_argument("--start", type=int, default=1, help="Primeiro numero")
    args = parser.parse_args()

    atribuir_line_por_arquivo(args.input, args.output, args.start)
