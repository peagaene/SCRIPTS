"""
Main UI window for REURB exporter.
"""
from __future__ import annotations

import sys
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox

from reurb.ui.widgets import _row, _browse_entry
from reurb.ui.log_handler import _QueueLogStream


def abrir_ui(params_default, on_execute=None) -> dict:
    root = tk.Tk()
    root.title("REURB - Exportador")
    root.geometry("1200x650")

    nb = ttk.Notebook(root)
    tab_geral = ttk.Frame(nb)
    tab_textos = ttk.Frame(nb)
    nb.add(tab_geral, text="Geral")
    nb.add(tab_textos, text="Textos")
    nb.pack(fill="both", expand=True, padx=10, pady=8)

    # === Arquivos ===
    v_nome = tk.StringVar(value="PROJETO")
    v_txt = tk.StringVar(value="E:/PROJETO_NEIDE/teste/junq/EDITADO.txt")
    v_dados = tk.StringVar(value="E:/PROJETO_NEIDE/teste/junq/sistvia.dxf")
    v_mdt = tk.StringVar(value="E:/PROJETO_NEIDE/teste/junq/JD_JUNQUEIROPOLIS.tif")
    v_out = tk.StringVar(value="E:/PROJETO_NEIDE/teste/junq")

    tab_geral.columnconfigure(0, weight=3)
    tab_geral.columnconfigure(1, weight=2)
    tab_geral.rowconfigure(0, weight=0)
    tab_geral.rowconfigure(1, weight=0)
    tab_geral.rowconfigure(2, weight=0)
    tab_geral.rowconfigure(3, weight=1)

    frm_files = ttk.LabelFrame(tab_geral, text="Arquivos")
    frm_files.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=10, pady=8)
    _browse_entry(frm_files, 0, "TXT (blocos, opcional):", v_txt, "file", (("TXT", "*.txt"),))
    _browse_entry(frm_files, 1, "DXF Dados (opcional):", v_dados, "file", (("DXF", "*.dxf"),))
    _browse_entry(frm_files, 2, "MDT (GeoTIFF, opcional):", v_mdt, "file", (("GeoTIFF", "*.tif;*.tiff"),))
    _browse_entry(frm_files, 3, "Pasta de saida:", v_out, "dir")

    ttk.Label(frm_files, text="Nome do projeto/area:").grid(row=4, column=0, sticky="w", padx=6, pady=6)
    ttk.Entry(frm_files, textvariable=v_nome, width=32).grid(row=4, column=1, sticky="w", padx=6, pady=6)

    # === Simbologia fixa ===
    v_simb_tipo = tk.StringVar(value="SIMBOLOGIA")

    # === O que exportar ===
    frm_opts = ttk.LabelFrame(tab_geral, text="O que exportar")
    frm_opts.grid(row=0, column=1, sticky="nsew", padx=10, pady=8)
    v_do_txt = tk.BooleanVar(value=False)
    v_do_per = tk.BooleanVar(value=False)
    v_do_cur = tk.BooleanVar(value=False)
    v_do_dren = tk.BooleanVar(value=False)
    v_do_vias = tk.BooleanVar(value=False)
    v_do_lotes_dim = tk.BooleanVar(value=False)
    exports_vars = (v_do_txt, v_do_per, v_do_cur, v_do_dren, v_do_vias, v_do_lotes_dim)
    ttk.Checkbutton(frm_opts, text="TXT (blocos)", variable=v_do_txt).grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frm_opts, text="Perimetro", variable=v_do_per).grid(row=0, column=1, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frm_opts, text="Curvas de nivel", variable=v_do_cur).grid(row=1, column=0, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frm_opts, text="Setas de drenagem (MDT)", variable=v_do_dren).grid(row=1, column=1, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frm_opts, text="Vias (medidas)", variable=v_do_vias).grid(row=2, column=0, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frm_opts, text="Lotes (dimensoes)", variable=v_do_lotes_dim).grid(row=2, column=1, sticky="w", padx=6, pady=4)

    def _select_all():
        for var in exports_vars:
            var.set(True)

    ttk.Button(frm_opts, text="Selecionar todos", command=_select_all).grid(row=0, column=3, rowspan=2, padx=6, pady=4)

    # === Fonte do perimetro ===
    frm_per_src = ttk.LabelFrame(tab_geral, text="Perimetro: fonte para Tabela e Vertices")
    frm_per_src.grid(row=1, column=1, sticky="nsew", padx=10, pady=8)
    v_per_src = tk.StringVar(value="")
    ttk.Radiobutton(frm_per_src, text="PER_INTERESSE", variable=v_per_src, value="interesse").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ttk.Radiobutton(frm_per_src, text="PER_LEVANTAMENTO", variable=v_per_src, value="levantamento").grid(row=0, column=1, sticky="w", padx=6, pady=4)

    # === Curvas de nivel ===
    frm_curvas = ttk.LabelFrame(tab_geral, text="Curvas de nivel")
    frm_curvas.grid(row=2, column=1, sticky="nsew", padx=10, pady=8)
    v_label_mode = tk.StringVar(value="ends" if bool(getattr(params_default, "curva_label_ends_only", False)) else "padrao")
    ttk.Radiobutton(frm_curvas, text="Padrao (ao longo da curva)", variable=v_label_mode, value="padrao").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ttk.Radiobutton(frm_curvas, text="Somente nas extremidades (inicial/final)", variable=v_label_mode, value="ends").grid(row=1, column=0, sticky="w", padx=6, pady=4)

    # === TEXTOS ===
    def _get(p, d):
        try:
            return float(getattr(params_default, p))
        except Exception:
            return d

    tab_textos.columnconfigure(0, weight=1)
    frm_txt = ttk.LabelFrame(tab_textos, text="Textos e Dimensoes")
    frm_txt.grid(row=0, column=0, sticky="nsew", padx=10, pady=8)
    v_h_soleira = tk.DoubleVar(value=_get("altura_texto_soleira", 0.75))
    v_h_area = tk.DoubleVar(value=_get("altura_texto_area", 0.60))
    v_h_p = tk.DoubleVar(value=_get("altura_texto_P", 0.50))
    v_h_tab = tk.DoubleVar(value=_get("altura_texto_tabela", 2.00))
    v_h_curva = tk.DoubleVar(value=_get("altura_texto_curva", 0.40))
    v_h_via = tk.DoubleVar(value=_get("altura_texto_via", 0.40))
    v_dimtxt_ord_via = tk.DoubleVar(value=_get("dimtxt_ordinate_via", 0.50))
    v_dimtxt_ord_per = tk.DoubleVar(value=_get("dimtxt_ordinate_perim", 0.50))
    _row(frm_txt, 0, "Texto Soleira (m):", v_h_soleira)
    _row(frm_txt, 1, "Texto Areas (m):", v_h_area)
    _row(frm_txt, 2, "Texto P (m):", v_h_p)
    _row(frm_txt, 3, "Texto Tabela (m):", v_h_tab)
    _row(frm_txt, 4, "Texto Curva (m):", v_h_curva)
    _row(frm_txt, 5, "Texto Via (m):", v_h_via)
    _row(frm_txt, 6, "DIM Text (Ordinate) - VIA:", v_dimtxt_ord_via, tip="Altura do texto das cotas do tipo Ordinate usadas em vias (se aplicavel).")
    _row(frm_txt, 7, "DIM Text (Ordinate) - PERIMETRO:", v_dimtxt_ord_per, tip="Altura do texto das cotas Ordinate do perimetro (azimute/distancia).")

    # === Log ===
    frm_log = ttk.LabelFrame(tab_geral, text="Log")
    frm_log.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=8)
    log_text = tk.Text(frm_log, height=10, wrap="word")
    log_scroll = ttk.Scrollbar(frm_log, command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_text.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
    log_scroll.pack(side="right", fill="y", padx=(0, 6), pady=6)

    log_q: queue.Queue = queue.Queue()

    def _poll_log():
        try:
            while True:
                msg = log_q.get_nowait()
                log_text.insert("end", msg)
                log_text.see("end")
        except queue.Empty:
            pass
        root.after(100, _poll_log)

    _poll_log()

    # === Rodape ===
    frm_bottom = ttk.Frame(root)
    frm_bottom.pack(fill="x", padx=10, pady=10)
    out = {}

    def _ok():
        if not v_out.get():
            messagebox.showerror("Atencao", "Pasta de saida e obrigatoria.")
            return
        per_src_val = v_per_src.get().strip() or None
        vias_cfg = {
            "via_offset_texto": float(getattr(params_default, "via_offset_texto", 0.50)),
            "via_cross_span": float(getattr(params_default, "via_cross_span", 80.0)),
            "via_nome_maiusculas": bool(getattr(params_default, "via_nome_maiusculas", False)),
            "via_offset_lote_lote_extra_m": float(getattr(params_default, "via_offset_lote_lote_extra_m", 0.60)),
        }
        curvas_cfg = {
            "curva_equidist": float(getattr(params_default, "curva_equidist", 1.0)),
            "curva_mestra_cada": int(getattr(params_default, "curva_mestra_cada", 5)),
            "curva_char_w_factor": float(getattr(params_default, "curva_char_w_factor", 0.60)),
            "curva_gap_margin": float(getattr(params_default, "curva_gap_margin", 0.50)),
            "curva_label_step_m": float(getattr(params_default, "curva_label_step_m", 80.0)),
            "curva_min_len": float(getattr(params_default, "curva_min_len", 10.0)),
            "curva_min_area": float(getattr(params_default, "curva_min_area", 20.0)),
            "curva_smooth_sigma_px": float(getattr(params_default, "curva_smooth_sigma_px", 1.0)),
            "curva_label_ends_only": (v_label_mode.get() == "ends"),
        }
        setas_cfg = {
            "setas_buffer_distancia": float(getattr(params_default, "setas_buffer_distancia", 5.0)),
            "setas_seg_curto_max": int(getattr(params_default, "setas_seg_curto_max", 3)),
            "setas_seg_medio_max": int(getattr(params_default, "setas_seg_medio_max", 4)),
            "setas_seg_longo_max": int(getattr(params_default, "setas_seg_longo_max", 5)),
        }
        out.update({
            "nome_area": v_nome.get().strip() or "PROJETO",
            "paths": {
                "txt": v_txt.get() or None,
                "simb_tipo": v_simb_tipo.get(),
                "dados": v_dados.get() or None,
                "mdt": v_mdt.get() or None,
                "saida": v_out.get(),
            },
            "exports": {
                "txt": bool(v_do_txt.get()),
                "perimetros": bool(v_do_per.get()),
                "curvas": bool(v_do_cur.get()),
                "drenagem": bool(v_do_dren.get()),
                "vias": bool(v_do_vias.get()),
                "lotes_dim": bool(v_do_lotes_dim.get()),
                "per_source": per_src_val,
            },
            "textos": {
                "altura_texto_soleira": float(v_h_soleira.get()),
                "altura_texto_area": float(v_h_area.get()),
                "altura_texto_P": float(v_h_p.get()),
                "altura_texto_tabela": float(v_h_tab.get()),
                "altura_texto_curva": float(v_h_curva.get()),
                "altura_texto_via": float(v_h_via.get()),
                "dimtxt_ordinate_via": float(v_dimtxt_ord_via.get()),
                "dimtxt_ordinate_perim": float(v_dimtxt_ord_per.get()),
            },
            "vias": vias_cfg,
            "curvas": curvas_cfg,
            "setas": setas_cfg,
        })
        if on_execute:
            def _run():
                old_stdout, old_stderr = sys.stdout, sys.stderr
                q_logger = _QueueLogStream(log_q)
                sys.stdout = q_logger
                sys.stderr = q_logger
                try:
                    on_execute(out)
                except Exception as e:
                    log_q.put(f"[ERRO] {e}\n")
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
            threading.Thread(target=_run, daemon=True).start()
            return
        root.destroy()

    ttk.Button(frm_bottom, text="Cancelar", command=root.destroy).pack(side="right", padx=6)
    ttk.Button(frm_bottom, text="Executar", command=_ok).pack(side="right", padx=6)

    root.mainloop()
    return out


__all__ = ["abrir_ui"]
