from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median


def transformar_txt_endz(
    arquivo_entrada: str | Path,
    arquivo_saida: str | Path,
    dz_col_index: int = 2,
    keep_negative_only: bool = True,
    median_factor: float | None = 1.8,
    abs_min: float | None = None,
    abs_max: float | None = None,
    extra_col_1: float = 0.0,
    extra_col_2: float = 0.0,
    decimals: int = 3,
) -> None:
    """
    Le TXT com colunas numericas e monta TXT final no formato:
    X Y C1 C2 DZ

    Por padrao:
    - X=coluna 0, Y=coluna 1, DZ=coluna 2
    - Mantem apenas DZ negativo e grava |DZ|
    - Remove outliers com |DZ| > median_factor * |mediana(DZ)|
    - Adiciona colunas extras C1 e C2 (default: 0 0)
    """
    entrada = Path(arquivo_entrada)
    saida = Path(arquivo_saida)
    saida.parent.mkdir(parents=True, exist_ok=True)

    registros: list[tuple[float, float, float]] = []
    valores_dz: list[float] = []

    with entrada.open("r", encoding="utf-8", errors="ignore") as f:
        for numero_linha, linha in enumerate(f, start=1):
            linha = linha.strip()
            if not linha:
                continue
            partes = linha.split()
            if len(partes) <= dz_col_index or len(partes) < 2:
                raise ValueError(
                    f"Linha {numero_linha} invalida: faltam colunas para X Y DZ (dz_col_index={dz_col_index})."
                )
            try:
                x = float(partes[0])
                y = float(partes[1])
                dz = float(partes[dz_col_index])
            except ValueError as e:
                raise ValueError(f"Linha {numero_linha} com valor nao numerico: {linha}") from e
            registros.append((x, y, dz))
            valores_dz.append(dz)

    if not registros:
        saida.write_text("", encoding="utf-8")
        return

    limite_median: float | None = None
    if median_factor is not None:
        med = median(valores_dz)
        limite_median = median_factor * abs(med)

    fmt = f"{{:.{decimals}f}}"
    saidas: list[str] = []
    for x, y, dz in registros:
        if keep_negative_only and dz > 0:
            continue

        dz_abs = abs(dz)

        if limite_median is not None and dz_abs > limite_median:
            continue
        if abs_min is not None and dz_abs < abs_min:
            continue
        if abs_max is not None and dz_abs > abs_max:
            continue

        saidas.append(
            f"{fmt.format(x)} {fmt.format(y)} {fmt.format(extra_col_1)} {fmt.format(extra_col_2)} {fmt.format(dz_abs)}\n"
        )

    with saida.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(saidas)

    print(f"[OK] Linhas entrada: {len(registros)} | Linhas saida: {len(saidas)} | Arquivo: {saida}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Organiza TXT EN_DZ e aplica filtros antes do output final.")
    ap.add_argument("arquivo_entrada", type=Path)
    ap.add_argument("arquivo_saida", type=Path)
    ap.add_argument("--dz-col-index", type=int, default=2, help="Indice da coluna DZ no TXT de entrada. Default=2.")
    ap.add_argument(
        "--allow-positive",
        action="store_true",
        help="Permite manter DZ positivo (por padrao remove positivos).",
    )
    ap.add_argument(
        "--median-factor",
        type=float,
        default=1.8,
        help="Remove |DZ| > fator * |mediana(DZ)|. Use valor negativo para desativar.",
    )
    ap.add_argument("--abs-min", type=float, default=None, help="Remove |DZ| menor que este valor.")
    ap.add_argument("--abs-max", type=float, default=None, help="Remove |DZ| maior que este valor.")
    ap.add_argument("--extra-col-1", type=float, default=0.0, help="Valor da coluna extra C1. Default=0.")
    ap.add_argument("--extra-col-2", type=float, default=0.0, help="Valor da coluna extra C2. Default=0.")
    ap.add_argument("--decimals", type=int, default=3, help="Casas decimais na saida.")
    args = ap.parse_args()

    median_factor = None if args.median_factor < 0 else args.median_factor
    transformar_txt_endz(
        arquivo_entrada=args.arquivo_entrada,
        arquivo_saida=args.arquivo_saida,
        dz_col_index=args.dz_col_index,
        keep_negative_only=not args.allow_positive,
        median_factor=median_factor,
        abs_min=args.abs_min,
        abs_max=args.abs_max,
        extra_col_1=args.extra_col_1,
        extra_col_2=args.extra_col_2,
        decimals=args.decimals,
    )


if __name__ == "__main__":
    main()
