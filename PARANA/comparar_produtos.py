#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comparador de pastas por NOME (ignorando extensão e prefixos/sufixos)
--------------------------------------------------------------------

Regras de comparação de nome:

Exemplo de arquivo:
    ES_L09_I_MDS_2862_1_SE_A_I_R0.tif

• Tudo ANTES do primeiro bloco numérico de 4 dígitos é ignorado (prefixos como
  ES_L09_I_MDS_, ES_L09_I_NPc_C_, etc. não entram na comparação).
• A partir do primeiro bloco de 4 dígitos em diante é considerado o "código".
• Sufixos do tipo "_R0", "-R0", "_R1" etc. são removidos.
• Hífens e underlines são tratados como equivalentes ("-" ou "_" → "-").

Assim, todos abaixo viram a mesma chave de comparação:
    ES_L09_I_MDS_2862_1_SE_A_I_R0
    ES_L09_I_MDS_2862-1-SE-A-I_R1
    ES_L09_I_NPc_C_2862_1_SE_A_I
    2862-1-SE-A-I

E serão considerados "o mesmo arquivo" para efeito de comparação entre pastas.

Interface:
• PySimpleGUI para escolher Pasta A e Pasta B
• Referência automática (pasta com MENOS arquivos únicos vira referência) ou forçar A→B / B→A
• Copiar resultado para área de transferência
• Exportar CSV com ausentes em A e em B
• Limpa automaticamente o resultado ao trocar as pastas

Requisitos: PySimpleGUI (pip install PySimpleGUI)
"""
from __future__ import annotations

import os
import csv
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import PySimpleGUI as sg


# ------------------------- Funções utilitárias ------------------------- #

def _normaliza_nome_sem_extensao(nome: str) -> str:
    """Normaliza o nome para comparação.

    Passos:
    1. Remove espaços nas pontas.
    2. Corta tudo antes do primeiro bloco de 4 dígitos (se existir).
    3. Remove sufixo do tipo "_R0", "-R1" etc. no final.
    4. Substitui hífen/underline por um hífen único.
    5. Usa casefold() para comparação sem diferenciar maiúsc./minúsc.
    """
    s = nome.strip()

    # 2) A partir do primeiro bloco de 4 dígitos
    m = re.search(r"(\d{4})", s)
    if m:
        s = s[m.start():]

    # 3) Remove sufixos tipo _R0, -R0, _R1 etc. no final
    s = re.sub(r"[-_]R\d+$", "", s, flags=re.IGNORECASE)

    # 4) Normaliza hífen/underline
    s = re.sub(r"[-_]", "-", s)

    # 5) casefold para comparação case-insensitive
    return s.casefold()


def coletar_basenomes(pasta: Path) -> Tuple[Set[str], Dict[str, str]]:
    """Retorna (conjunto_normalizado, mapa_normalizado->original) para os nomes.

    Ignora subpastas; só arquivos diretos.
    """
    if not pasta.exists() or not pasta.is_dir():
        raise FileNotFoundError(f"Pasta inválida: {pasta}")

    conjunto_norm: Set[str] = set()
    mapa_norm_para_original: Dict[str, str] = {}

    with os.scandir(pasta) as it:
        for entry in it:
            if not entry.is_file():
                continue
            stem_original = os.path.splitext(entry.name)[0]
            norm = _normaliza_nome_sem_extensao(stem_original)
            if norm not in conjunto_norm:
                conjunto_norm.add(norm)
                # Guarda o texto original do primeiro que aparece (para exibir)
                mapa_norm_para_original[norm] = stem_original

    return conjunto_norm, mapa_norm_para_original


def comparar_basenomes(
    pasta_ref: Path,
    pasta_alvo: Path,
) -> Tuple[List[str], int, int]:
    """Compara nomes normalizados da pasta_ref contra pasta_alvo.

    Retorna (lista_de_ausentes_no_alvo_com_texto_original_da_ref, total_ref, total_alvo).
    """
    ref_set, ref_map = coletar_basenomes(pasta_ref)
    alvo_set, _ = coletar_basenomes(pasta_alvo)

    ausentes_norm = sorted(ref_set - alvo_set)
    ausentes_texto = [ref_map[n] for n in ausentes_norm]
    return ausentes_texto, len(ref_set), len(alvo_set)


def salvar_csv(caminho_csv: Path, faltando_em_a: List[str], faltando_em_b: List[str]):
    """Salva um CSV com duas colunas: FALTANDO_EM_A e FALTANDO_EM_B."""
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["FALTANDO_EM_A", "FALTANDO_EM_B"])  # A = Pasta A, B = Pasta B
        maxlen = max(len(faltando_em_a), len(faltando_em_b))
        for i in range(maxlen):
            a = faltando_em_a[i] if i < len(faltando_em_a) else ""
            b = faltando_em_b[i] if i < len(faltando_em_b) else ""
            w.writerow([a, b])


# ----------------------------- Interface -------------------------------- #

def construir_layout():
    col_pastas = [
        [
            sg.Text("Pasta A:"),
            sg.Input(key="-PASTA_A-", expand_x=True, enable_events=True),
            sg.FolderBrowse("Procurar…"),
        ],
        [
            sg.Text("Pasta B:"),
            sg.Input(key="-PASTA_B-", expand_x=True, enable_events=True),
            sg.FolderBrowse("Procurar…"),
        ],
        [
            sg.Text("Referência:"),
            sg.Radio("Automática (usa a MENOR)", "REF", True, key="-REF_AUTO-"),
            sg.Radio("Forçar A → B", "REF", key="-REF_A-"),
            sg.Radio("Forçar B → A", "REF", key="-REF_B-"),
        ],
        [
            sg.Button("Comparar", key="-COMPARAR-", bind_return_key=True),
            sg.Button("Copiar resultado", key="-COPIAR-"),
            sg.Button("Exportar CSV…", key="-CSV-"),
            sg.Push(),
            sg.Button("Abrir A", key="-ABRIR_A-"),
            sg.Button("Abrir B", key="-ABRIR_B-"),
            sg.Button("Sair"),
        ],
    ]

    col_resultado = [
        [sg.Text("Resumo:")],
        [
            sg.Multiline(
                "",
                size=(80, 18),
                key="-OUT-",
                autoscroll=True,
                disabled=True,
                font=("Consolas", 10),
            )
        ],
    ]

    layout = [
        [sg.Frame("Pastas", col_pastas, expand_x=True)],
        [sg.Frame("Resultado", col_resultado, expand_x=True, expand_y=True)],
    ]

    return layout


def escolher_referencia(pasta_a: Path, pasta_b: Path, ref_auto: bool, ref_a: bool) -> Tuple[Path, Path, str]:
    """Decide direção da comparação e retorna (pasta_ref, pasta_alvo, etiqueta)."""
    if ref_auto:
        set_a, _ = coletar_basenomes(pasta_a)
        set_b, _ = coletar_basenomes(pasta_b)
        if len(set_a) <= len(set_b):
            return pasta_a, pasta_b, "A (menor) → B"
        else:
            return pasta_b, pasta_a, "B (menor) → A"
    elif ref_a:
        return pasta_a, pasta_b, "A → B (forçado)"
    else:
        return pasta_b, pasta_a, "B → A (forçado)"


def formatar_resultado(
    etiqueta_ref: str,
    faltando_no_alvo: List[str],
    qtd_ref: int,
    qtd_alvo: int,
    faltando_em_a: List[str],
    faltando_em_b: List[str],
) -> str:
    """Monta o texto exibido na caixa de resultado."""
    linhas: List[str] = []
    linhas.append(f"Referência: {etiqueta_ref}")
    linhas.append("")

    # Mostrar contagens A/B de forma legível
    if etiqueta_ref.startswith("A"):
        linhas.append(f"Arquivos únicos (normalizados): A={qtd_ref}  |  B={qtd_alvo}")
    elif etiqueta_ref.startswith("B"):
        linhas.append(f"Arquivos únicos (normalizados): A={qtd_alvo}  |  B={qtd_ref}")
    else:
        linhas.append(f"Arquivos únicos (normalizados): A≈{qtd_ref}  |  B≈{qtd_alvo}")

    linhas.append("-")
    linhas.append("Ausentes na pasta alvo (referência → alvo):")
    if faltando_no_alvo:
        for nome in faltando_no_alvo:
            linhas.append(f"  • {nome}")
    else:
        linhas.append("  • Nenhum — a pasta alvo possui todos os nomes da referência.")

    linhas.append("")
    linhas.append("(Diagnóstico completo)")
    linhas.append("Faltando em A (nomes presentes em B e ausentes em A):")
    if faltando_em_a:
        for n in faltando_em_a:
            linhas.append(f"  - {n}")
    else:
        linhas.append("  - Nenhum")

    linhas.append("Faltando em B (nomes presentes em A e ausentes em B):")
    if faltando_em_b:
        for n in faltando_em_b:
            linhas.append(f"  - {n}")
    else:
        linhas.append("  - Nenhum")

    return "\n".join(linhas)


# ----------------------------- Main ------------------------------------- #

def main():
    sg.theme("SystemDefault")

    janela = sg.Window(
        "Comparar pastas por nome (ignora prefixo/sufixo e extensão)",
        construir_layout(),
        resizable=True,
        finalize=True,
    )
    # Deixar tab size bonitinho
    janela["-OUT-"].Widget.configure(tabs="4")

    faltando_em_a: List[str] = []
    faltando_em_b: List[str] = []

    while True:
        ev, vals = janela.read()
        if ev in (sg.WIN_CLOSED, "Sair"):
            break

        # Limpa automaticamente a saída ao alterar qualquer pasta
        if ev in ("-PASTA_A-", "-PASTA_B-"):
            janela["-OUT-"].update("")
            faltando_em_a = []
            faltando_em_b = []
            continue

        if ev == "-ABRIR_A-" and vals.get("-PASTA_A-"):
            sg.execute_command_subprocess("explorer" if os.name == "nt" else "open",
                                          vals["-PASTA_A-"])
        if ev == "-ABRIR_B-" and vals.get("-PASTA_B-"):
            sg.execute_command_subprocess("explorer" if os.name == "nt" else "open",
                                          vals["-PASTA_B-"])

        if ev == "-COMPARAR-":
            try:
                pasta_a = Path(vals.get("-PASTA_A-", "")).expanduser()
                pasta_b = Path(vals.get("-PASTA_B-", "")).expanduser()
                if not pasta_a or not pasta_b:
                    sg.popup_error("Selecione as duas pastas (A e B).")
                    continue

                ref_auto = bool(vals.get("-REF_AUTO-"))
                ref_a = bool(vals.get("-REF_A-"))
                pasta_ref, pasta_alvo, etiqueta = escolher_referencia(
                    pasta_a, pasta_b, ref_auto, ref_a
                )

                # Diagnóstico completo (ambas as direções)
                faltando_em_a, _, _ = comparar_basenomes(pasta_b, pasta_a)  # o que falta em A
                faltando_em_b, _, _ = comparar_basenomes(pasta_a, pasta_b)  # o que falta em B

                # Lista principal: faltando na pasta alvo, considerando escolha de referência
                faltando_no_alvo, qtd_ref, qtd_alvo = comparar_basenomes(pasta_ref, pasta_alvo)

                texto = formatar_resultado(
                    etiqueta,
                    faltando_no_alvo,
                    qtd_ref,
                    qtd_alvo,
                    faltando_em_a,
                    faltando_em_b,
                )
                janela["-OUT-"].update(texto)
            except Exception as e:
                sg.popup_error(f"Erro ao comparar pastas:\n{e}")

        if ev == "-COPIAR-":
            conteudo = janela["-OUT-"].get().strip()
            if not conteudo:
                sg.popup("Nada para copiar. Faça uma comparação primeiro.")
            else:
                sg.clipboard_set(conteudo)
                sg.popup("Resultado copiado para a área de transferência.")

        if ev == "-CSV-":
            if not (faltando_em_a or faltando_em_b):
                conteudo = janela["-OUT-"].get().strip()
                if not conteudo:
                    sg.popup("Faça uma comparação antes de exportar.")
                    continue
            caminho_csv = sg.popup_get_file(
                "Salvar CSV",
                save_as=True,
                default_extension=".csv",
                file_types=(("CSV", "*.csv"),),
                no_window=True,
            )
            if caminho_csv:
                try:
                    salvar_csv(Path(caminho_csv), faltando_em_a, faltando_em_b)
                    sg.popup(f"CSV salvo em:\n{caminho_csv}")
                except Exception as e:
                    sg.popup_error(f"Não foi possível salvar o CSV:\n{e}")

    janela.close()


if __name__ == "__main__":
    main()
