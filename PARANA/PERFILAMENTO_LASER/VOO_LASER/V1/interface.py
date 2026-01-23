
import os
import PySimpleGUI as sg
import threading
from datetime import datetime
from gerar_planilha_controle import gerar_planilha_controle
from gerar_relatorio_bordo import gerar_relatorio_bordo
from gerar_relatorio_densidade import gerar_relatorio_densidade
from gerar_trajetoria_unificada import gerar_trajetoria_unificada
from gerar_planilha_campo import gerar_planilha_campo

DIR_EXECUCAO_VOO = r"\\192.168.2.28\i\80225_PROJETO_IAT_PARANA\3 Execução de voo"
DIR_RINEX = r"\\192.168.2.28\i\80225_PROJETO_IAT_PARANA\4 Ponto de apoio\RINEX"

def obter_datas_blocos(blocos):
    datas_set = set()
    for bloco in blocos:
        dir_bloco = os.path.join(DIR_EXECUCAO_VOO, bloco)
        if os.path.isdir(dir_bloco):
            for pasta in os.listdir(dir_bloco):
                if pasta.startswith("2025"):
                    try:
                        data_str, _ = pasta.split("_")
                        data_dt = datetime.strptime(data_str, "%Y%m%d").date()
                        datas_set.add(data_dt)
                    except:
                        pass
    return sorted(datas_set)

def obter_datas_rinex():
    datas = set()
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
    blocos = sorted([b for b in os.listdir(DIR_EXECUCAO_VOO) if b.startswith("BLOCO_")])
    return [[
        sg.Checkbox(bloco, key=f"CHK_{bloco}", enable_events=True),
        sg.Button("📅", key=f"BTN_{bloco}", size=(2, 1), pad=((10, 0), 0))
    ] for bloco in blocos]

layout_laser = [
    [sg.Text("Selecione os blocos:")],
    [
        sg.Column(blocos_com_checkbox(), size=(280, 160), scrollable=True),
        sg.VSeperator(),
        sg.Column([
            [sg.Text("Dias disponíveis:")],
            [sg.Listbox(values=[], size=(20, 8), select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, key="DATAS_DISP")]
        ])
    ],
    [sg.Checkbox("Gerar Planilha de Controle", key="CTRL")],
    [sg.Checkbox("Gerar Relatório de Bordo", key="RB")],
    [sg.Checkbox("Exportar Trajetória Unificada (SHP/KMZ)", key="TRJ")],
    [sg.Checkbox("Calcular Densidade LAS", key="DENS")],
]

layout_campo = [
    [sg.Text("Dias com RINEX:")],
    [sg.Listbox(values=[d.strftime("%Y-%m-%d") for d in obter_datas_rinex()],
                size=(25, 6), select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, key="DATAS_RINEX")]
]

layout = [
    [sg.TabGroup([[sg.Tab("LASER", layout_laser), sg.Tab("CAMPO", layout_campo)]])],
    [sg.Multiline(size=(80, 10), key="LOG", autoscroll=True, disabled=True)],
    [sg.Button("Executar"), sg.Button("Cancelar")]
]

janela = sg.Window("Exportador - Projeto Paraná", layout)

while True:
    evento, valores = janela.read()
    if evento in (sg.WINDOW_CLOSED, "Cancelar"):
        break

    if evento.startswith("BTN_"):
        bloco_clicado = evento.replace("BTN_", "")
        datas = obter_datas_blocos([bloco_clicado])
        janela["DATAS_DISP"].update(values=[d.strftime("%Y-%m-%d") for d in datas])

    if evento == "Executar":
        blocos = [k.replace("CHK_", "") for k in valores if isinstance(k, str) and k.startswith("CHK_") and valores[k]]
        if not blocos:
            blocos = sorted([b for b in os.listdir(DIR_EXECUCAO_VOO) if b.startswith("BLOCO_")])

        datas_filtro = [datetime.strptime(d, "%Y-%m-%d").date() for d in valores.get("DATAS_DISP", [])]
        datas_rinex = [datetime.strptime(d, "%Y-%m-%d").date() for d in valores.get("DATAS_RINEX", [])]

        opcoes = {
            "controle": valores.get("CTRL", False),
            "relatorio_bordo": valores.get("RB", False),
            "trajetoria": valores.get("TRJ", False),
            "densidade": valores.get("DENS", False)
        }

        janela["LOG"].update("[INFO] Iniciando processamento...", append=True)
        janela["LOG"].update(f"[INFO] Blocos selecionados: {blocos}", append=True)
        janela["LOG"].update(f"[INFO] Datas LAS: {datas_filtro if datas_filtro else 'Nenhuma (todas)'}", append=True)
        janela["LOG"].update(f"[INFO] Datas RINEX: {datas_rinex if datas_rinex else 'Nenhuma (todas)'}", append=True)
        janela["LOG"].update(f"[INFO] Opções: {opcoes}", append=True)

        def processar():
            if opcoes["controle"]:
                try:
                    gerar_planilha_controle(DIR_EXECUCAO_VOO, blocos, datas_filtro)
                    janela.write_event_value("-LOG-", "[OK] Planilha de controle gerada.\n  → ENTREGA\\1_PLANILHA_CONTROLE")
                except Exception as e:
                    janela.write_event_value("-LOG-", f"[ERRO] Planilha de controle: {e}")

            if opcoes["relatorio_bordo"]:
                try:
                    gerar_relatorio_bordo(DIR_EXECUCAO_VOO, blocos, datas_filtro)
                    janela.write_event_value("-LOG-", "[OK] Relatório de bordo gerado.\n  → ENTREGA\\3_RELATORIO_BORDO")
                except Exception as e:
                    janela.write_event_value("-LOG-", f"[ERRO] Relatório de bordo: {e}")

            if opcoes["densidade"]:
                try:
                    gerar_relatorio_densidade(DIR_EXECUCAO_VOO, blocos, datas_filtro,
                                            log_func=lambda msg: janela.write_event_value("-LOG-", msg))
                    janela.write_event_value("-LOG-", "[OK] Densidade LAS gerada.\n  → DENSIDADE_LAS")
                except Exception as e:
                    janela.write_event_value("-LOG-", f"[ERRO] Densidade LAS: {e}")

            if opcoes["trajetoria"]:
                try:
                    gerar_trajetoria_unificada(DIR_EXECUCAO_VOO, blocos, datas_filtro,
                                            log_func=lambda msg: janela.write_event_value("-LOG-", msg))
                    janela.write_event_value("-LOG-", "[OK] Trajetória unificada exportada.\n  → ENTREGA\\4_TRAJETORIA\\BLOCO\\SHP e \\KMZ")

                except Exception as e:
                    janela.write_event_value("-LOG-", f"[ERRO] Trajetória unificada: {e}")

            if datas_rinex:
                try:
                    resultados = gerar_planilha_campo(
                        dir_rinex_base=DIR_RINEX,
                        dir_template=DIR_EXECUCAO_VOO,
                        dir_saida=os.path.join(DIR_EXECUCAO_VOO, "ENTREGA", "2_APOIO_DE_CAMPO"),
                        datas_filtradas=datas_rinex,
                        log_func=lambda msg: None
                    )
                    encontrados = {data for data, _, _ in resultados}
                    for data, base_nome, pasta_saida in resultados:
                        janela.write_event_value("-LOG-", f"[OK] Planilha de campo gerada ({base_nome}) para {data.strftime('%Y-%m-%d')}.\n  ??? {pasta_saida}")
                    faltantes = [d for d in datas_rinex if d not in encontrados]
                    for d in faltantes:
                        janela.write_event_value("-LOG-", f"[AVISO] Nenhum RINEX correspondente encontrado para {d.strftime('%Y-%m-%d')}.")
                except Exception as e:
                    janela.write_event_value("-LOG-", f"[ERRO] Planilha de campo: {e}")




        threading.Thread(target=processar, daemon=True).start()

    if evento == "-LOG-":
        janela["LOG"].update(valores["-LOG-"], append=True)

janela.close()
