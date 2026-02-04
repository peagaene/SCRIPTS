import os
import threading
from datetime import datetime

import PySimpleGUI as sg

from gerar_planilha_campo import gerar_planilha_campo
from gerar_planilha_controle import gerar_planilha_controle
from gerar_relatorio_bordo import gerar_relatorio_bordo
from gerar_relatorio_densidade import gerar_relatorio_densidade
from gerar_trajetoria_unificada import gerar_trajetoria_unificada

DIR_EXECUCAO_VOO = r"\\192.168.2.28\i\80225_PROJETO_IAT_PARANA\3 Execução de voo"
DIR_RINEX = r"\\192.168.2.28\i\80225_PROJETO_IAT_PARANA\4 Ponto de apoio\RINEX"

BLOCOS_INFOS = []
KEY_TO_INFO = {}


def carregar_blocos_info():
    """Lê a estrutura LOTE_XX/BLOCO_YY e guarda os metadados para uso na interface."""
    global BLOCOS_INFOS, KEY_TO_INFO
    BLOCOS_INFOS = []
    KEY_TO_INFO = {}

    if not os.path.isdir(DIR_EXECUCAO_VOO):
        return BLOCOS_INFOS

    lotes = sorted([nome for nome in os.listdir(DIR_EXECUCAO_VOO) if nome.upper().startswith("LOTE_")])
    for lote_nome in lotes:
        dir_lote = os.path.join(DIR_EXECUCAO_VOO, lote_nome)
        if not os.path.isdir(dir_lote):
            continue

        blocos = sorted([nome for nome in os.listdir(dir_lote) if nome.upper().startswith("BLOCO_")])
        for bloco_nome in blocos:
            dir_bloco = os.path.join(dir_lote, bloco_nome)
            if not os.path.isdir(dir_bloco):
                continue

            lote_id = lote_nome.split("_")[-1]
            info = {
                "lote_nome": lote_nome,
                "lote_id": lote_id,
                "dir_lote": dir_lote,
                "bloco_nome": bloco_nome,
                "dir_bloco": dir_bloco,
            }
            BLOCOS_INFOS.append(info)

    return BLOCOS_INFOS


def obter_datas_blocos(blocos_info):
    datas_set = set()
    for info in blocos_info:
        dir_bloco = info.get("dir_bloco")
        if not dir_bloco or not os.path.isdir(dir_bloco):
            continue

        for pasta in os.listdir(dir_bloco):
            if not pasta.startswith("2025"):
                continue
            try:
                data_str, _ = pasta.split("_")
                data_dt = datetime.strptime(data_str, "%Y%m%d").date()
                datas_set.add(data_dt)
            except Exception:
                pass
    return sorted(datas_set)


def obter_datas_rinex():
    datas = set()
    if not os.path.isdir(DIR_RINEX):
        return sorted(datas)

    for base_folder in os.listdir(DIR_RINEX):
        caminho_base = os.path.join(DIR_RINEX, base_folder)
        if not os.path.isdir(caminho_base):
            continue
        for pasta in os.listdir(caminho_base):
            if not pasta.startswith("2025"):
                continue
            try:
                data = datetime.strptime(pasta, "%Y%m%d").date()
            except ValueError:
                continue
            datas.add(data)
    return sorted(datas)


def blocos_com_checkbox():
    blocos_info = carregar_blocos_info()
    elementos = []
    for idx, info in enumerate(blocos_info):
        lote_label = info["lote_nome"].replace("LOTE_", "Lote ")
        bloco_label = f"{info['bloco_nome']} ({lote_label})"
        chk_key = f"CHK_{idx}"
        btn_key = f"BTN_{idx}"
        KEY_TO_INFO[chk_key] = info
        KEY_TO_INFO[btn_key] = info
        elementos.append([
            sg.Checkbox(bloco_label, key=chk_key, enable_events=True),
            sg.Button(">>", key=btn_key, size=(3, 1), pad=((10, 0), 0)),
        ])
    return elementos


layout_laser = [
    [sg.Text("Selecione os blocos:")],
    [
        sg.Column(blocos_com_checkbox(), size=(360, 220), scrollable=True),
        sg.VSeperator(),
        sg.Column([
            [sg.Text("Dias disponíveis:")],
            [sg.Listbox(values=[], size=(22, 10), select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, key="DATAS_DISP")],
        ]),
    ],
    [sg.Checkbox("Gerar Planilha de Controle", key="CTRL")],
    [sg.Checkbox("Gerar Relatório de Bordo", key="RB")],
    [sg.Checkbox("Exportar Trajetória Unificada (SHP/KMZ)", key="TRJ")],
    [sg.Checkbox("Calcular Densidade LAS", key="DENS")],
]

layout_campo = [
    [sg.Text("Dias com RINEX:")],
    [sg.Listbox(
        values=[d.strftime("%Y-%m-%d") for d in obter_datas_rinex()],
        size=(25, 6),
        select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE,
        key="DATAS_RINEX",
    )],
]

layout = [
    [sg.TabGroup([[sg.Tab("LASER", layout_laser), sg.Tab("CAMPO (Classe I)", layout_campo)]])],
    [sg.Multiline(size=(80, 12), key="LOG", autoscroll=True, disabled=True)],
    [sg.Button("Executar"), sg.Button("Cancelar")],
]

janela = sg.Window("Exportador - Projeto Paraná", layout)

while True:
    evento, valores = janela.read()
    if evento in (sg.WINDOW_CLOSED, "Cancelar"):
        break

    if isinstance(evento, str) and evento.startswith("BTN_"):
        info = KEY_TO_INFO.get(evento)
        datas = obter_datas_blocos([info]) if info else []
        janela["DATAS_DISP"].update(values=[d.strftime("%Y-%m-%d") for d in datas])

    if evento == "Executar":
        blocos_info = [
            KEY_TO_INFO[chave]
            for chave in valores
            if isinstance(chave, str) and chave.startswith("CHK_") and valores[chave] and chave in KEY_TO_INFO
        ]
        if not blocos_info:
            blocos_info = list(BLOCOS_INFOS)

        if not blocos_info:
            janela["LOG"].update("[AVISO] Nenhum bloco encontrado para processar.", append=True)
            continue

        datas_filtro = [datetime.strptime(d, "%Y-%m-%d").date() for d in valores.get("DATAS_DISP", [])]
        datas_rinex = [datetime.strptime(d, "%Y-%m-%d").date() for d in valores.get("DATAS_RINEX", [])]

        opcoes = {
            "controle": valores.get("CTRL", False),
            "relatorio_bordo": valores.get("RB", False),
            "trajetoria": valores.get("TRJ", False),
            "densidade": valores.get("DENS", False),
        }

        janela["LOG"].update("[INFO] Iniciando processamento...\n", append=True)

        def processar():
            if opcoes["controle"]:
                try:
                    gerar_planilha_controle(DIR_EXECUCAO_VOO, blocos_info, datas_filtro)
                    janela.write_event_value("-LOG-", "[OK] Planilha de controle gerada.\n")
                except Exception as e:
                    janela.write_event_value("-LOG-", f"[ERRO] Planilha de controle: {e}")

            if opcoes["relatorio_bordo"]:
                try:
                    gerar_relatorio_bordo(DIR_EXECUCAO_VOO, blocos_info, datas_filtro)
                    janela.write_event_value("-LOG-", "[OK] Relatório de bordo gerado.\n")
                except Exception as e:
                    janela.write_event_value("-LOG-", f"[ERRO] Relatório de bordo: {e}")

            if opcoes["densidade"]:
                try:
                    gerar_relatorio_densidade(
                        DIR_EXECUCAO_VOO,
                        blocos_info,
                        datas_filtro,
                        log_func=lambda msg: janela.write_event_value("-LOG-", msg),
                    )
                    janela.write_event_value("-LOG-", "[OK] Densidade LAS gerada.\n")
                except Exception as e:
                    janela.write_event_value("-LOG-", f"[ERRO] Densidade LAS: {e}")

            if opcoes["trajetoria"]:
                try:
                    gerar_trajetoria_unificada(
                        DIR_EXECUCAO_VOO,
                        blocos_info,
                        datas_filtro,
                        log_func=lambda msg: janela.write_event_value("-LOG-", msg),
                    )
                    janela.write_event_value(
                        "-LOG-",
                        "[OK] Trajetória unificada exportada.",
                    )

                except Exception as e:
                    janela.write_event_value("-LOG-", f"[ERRO] Trajetória unificada: {e}")

            if datas_rinex:
                try:
                    resultados = gerar_planilha_campo(
                        dir_rinex_base=DIR_RINEX,
                        dir_template=DIR_EXECUCAO_VOO,
                        dir_saida=os.path.join(DIR_EXECUCAO_VOO, "ENTREGA", "1_CLASSE_I", "1_1_DADOS_RINEX"),
                        datas_filtradas=datas_rinex,
                        log_func=lambda msg: janela.write_event_value("-LOG-", msg),
                    )
                    encontrados = {data for data, _, _ in resultados}
                    for data, base_nome, pasta_saida in resultados:
                        janela.write_event_value(
                            "-LOG-",
                            f"[OK] Planilha de campo gerada ({base_nome}) para {data.strftime('%Y-%m-%d')}.\n  ??? {pasta_saida}",
                        )
                    faltantes = [d for d in datas_rinex if d not in encontrados]
                    for d in faltantes:
                        janela.write_event_value(
                            "-LOG-",
                            f"[AVISO] Nenhum RINEX correspondente encontrado para {d.strftime('%Y-%m-%d')}.",
                        )
                except Exception as e:
                    janela.write_event_value("-LOG-", f"[ERRO] Planilha de campo: {e}")

        threading.Thread(target=processar, daemon=True).start()

    if evento == "-LOG-":
        janela["LOG"].update(valores["-LOG-"], append=True)

janela.close()
