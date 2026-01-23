# === ui.py ===
import os
import json
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

_CFG = os.path.join(os.path.expanduser("~"), ".itesp_reurb_ui.json")

def _load_state():
    try:
        with open(_CFG, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_state(state: dict):
    try:
        with open(_CFG, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

class ToolTip:
    def __init__(self, widget, text, wrap=56):
        self.widget = widget; self.text = text; self.wrap = wrap; self.tip = None
        widget.bind("<Enter>", self._show); widget.bind("<Leave>", self._hide)
    def _show(self, *_):
        if self.tip or not self.text: return
        x, y = self.widget.winfo_pointerxy()
        tw = tk.Toplevel(self.widget); self.tip = tw
        tw.wm_overrideredirect(True); tw.wm_geometry(f"+{x+14}+{y+14}")
        lbl = tk.Label(tw, text=self.text, justify="left", background="#111", foreground="#fff",
                       relief="solid", borderwidth=1, padx=8, pady=6, wraplength=self.wrap*8, font=("Segoe UI", 9))
        lbl.pack()
    def _hide(self, *_):
        if self.tip: self.tip.destroy(); self.tip = None

def _row(parent, r, label, var, width=12, unit="", tip=""):
    ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=4)
    ent = ttk.Entry(parent, textvariable=var, width=width)
    ent.grid(row=r, column=1, sticky="we", padx=6, pady=4)
    ttk.Label(parent, text=unit).grid(row=r, column=2, sticky="w", padx=2)
    if tip: ToolTip(ent, tip)
    return ent

def _browse_entry(parent, r, label, var, kind="file", patterns=(("Todos","*.*"),), tip=""):
    ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=4)
    ent = ttk.Entry(parent, textvariable=var, width=64)
    ent.grid(row=r, column=1, sticky="we", padx=6, pady=4, columnspan=2)
    def go():
        init = os.path.dirname(var.get()) if var.get() else _load_state().get("lastdir", os.path.expanduser("~"))
        path = filedialog.askopenfilename(parent=parent, initialdir=init, filetypes=patterns) if kind=="file" \
               else filedialog.askdirectory(parent=parent, initialdir=init)
        if path:
            var.set(path); _save_state({"lastdir": os.path.dirname(path)})
    ttk.Button(parent, text="...", width=3, command=go).grid(row=r, column=3, padx=4)
    if tip: ToolTip(ent, tip)
    return ent



def abrir_ui(params_default, on_execute=None) -> dict:
    root = tk.Tk()
    root.title("REURB - Exportador")
    root.geometry("880x560")

    nb = ttk.Notebook(root)
    tab_geral = ttk.Frame(nb)
    tab_textos = ttk.Frame(nb)
    nb.add(tab_geral, text="Geral")
    nb.add(tab_textos, text="Textos")
    nb.pack(fill="both", expand=True)

    # === Arquivos ===
    v_nome = tk.StringVar(value="PROJETO")
    v_txt = tk.StringVar(value="")
    v_dados = tk.StringVar(value="")
    v_mdt = tk.StringVar(value="")
    v_out = tk.StringVar(value="")

    frm_files = ttk.LabelFrame(tab_geral, text="Arquivos")
    frm_files.pack(fill="x", padx=10, pady=8)
    _browse_entry(frm_files, 0, "TXT (blocos, opcional):", v_txt, "file", (("TXT","*.txt"),))
    _browse_entry(frm_files, 1, "DXF Dados (opcional):", v_dados, "file", (("DXF","*.dxf"),))
    _browse_entry(frm_files, 2, "MDT (GeoTIFF, opcional):", v_mdt, "file", (("GeoTIFF","*.tif;*.tiff"),))
    _browse_entry(frm_files, 3, "Pasta de saida:", v_out, "dir")

    ttk.Label(frm_files, text="Nome do projeto/area:").grid(row=4, column=0, sticky="w", padx=6, pady=6)
    ttk.Entry(frm_files, textvariable=v_nome, width=32).grid(row=4, column=1, sticky="w", padx=6, pady=6)

    # === Simbologia fixa ===
    frm_simb = ttk.LabelFrame(tab_geral, text="Simbologia (caminho fixo)")
    frm_simb.pack(fill="x", padx=10, pady=8)
    v_simb_tipo = tk.StringVar(value="SIMBOLOGIA")
    ttk.Label(frm_simb, text="SIMBOLOGIA.dxf (padrao)").grid(row=0, column=0, sticky="w", padx=6, pady=4)

    # === O que exportar ===
    frm_opts = ttk.LabelFrame(tab_geral, text="O que exportar")
    frm_opts.pack(fill="x", padx=10, pady=8)
    v_do_txt = tk.BooleanVar(value=False)
    v_do_per = tk.BooleanVar(value=False)
    v_do_cur = tk.BooleanVar(value=False)
    v_do_dren = tk.BooleanVar(value=False)
    exports_vars = (v_do_txt, v_do_per, v_do_cur, v_do_dren)
    ttk.Checkbutton(frm_opts, text="TXT (blocos)", variable=v_do_txt).grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frm_opts, text="Perimetro", variable=v_do_per).grid(row=0, column=1, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frm_opts, text="Curvas de nivel", variable=v_do_cur).grid(row=1, column=0, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frm_opts, text="Setas de drenagem (MDT)", variable=v_do_dren).grid(row=1, column=1, sticky="w", padx=6, pady=4)
    def _select_all():
        for var in exports_vars:
            var.set(True)
    ttk.Button(frm_opts, text="Selecionar todos", command=_select_all).grid(row=0, column=3, rowspan=2, padx=6, pady=4)

    # === Fonte do perimetro ===
    frm_per_src = ttk.LabelFrame(tab_geral, text="Perimetro: fonte para Tabela e Vertices")
    frm_per_src.pack(fill="x", padx=10, pady=8)
    v_per_src = tk.StringVar(value="")
    ttk.Radiobutton(frm_per_src, text="PER_INTERESSE", variable=v_per_src, value="interesse").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ttk.Radiobutton(frm_per_src, text="PER_LEVANTAMENTO", variable=v_per_src, value="levantamento").grid(row=0, column=1, sticky="w", padx=6, pady=4)

    # === EPSG OSM / Curvas ===
    frm_epsg_curvas = ttk.Frame(tab_geral)
    frm_epsg_curvas.pack(fill="x", padx=10, pady=8)

    frm_epsg = ttk.LabelFrame(frm_epsg_curvas, text="EPSG para consulta OSM")
    frm_epsg.grid(row=0, column=0, sticky="nsew")
    v_epsg_mode = tk.StringVar(value="31983")
    v_epsg_custom = tk.StringVar(value="")
    ttk.Radiobutton(frm_epsg, text="31982 (UTM 22S)", variable=v_epsg_mode, value="31982").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ttk.Radiobutton(frm_epsg, text="31983 (UTM 23S)", variable=v_epsg_mode, value="31983").grid(row=0, column=1, sticky="w", padx=6, pady=4)
    ttk.Radiobutton(frm_epsg, text="Custom:", variable=v_epsg_mode, value="custom").grid(row=1, column=0, sticky="w", padx=6, pady=4)
    ent_epsg = ttk.Entry(frm_epsg, textvariable=v_epsg_custom, width=10)
    ent_epsg.grid(row=1, column=1, sticky="w", padx=6, pady=4)
    ToolTip(ent_epsg, "Informe um EPSG projetado compativel (ex.: 31983).")

    frm_curvas = ttk.LabelFrame(frm_epsg_curvas, text="Curvas de nivel")
    frm_curvas.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
    v_label_ends = tk.BooleanVar(value=bool(getattr(params_default, "curva_label_ends_only", False)))
    ttk.Checkbutton(frm_curvas, text="Rotular somente nas extremidades (pontos inicial e final)", variable=v_label_ends).grid(row=0, column=0, sticky="w", padx=6, pady=4)
    
    # === Controle de Setas de Drenagem ===
    frm_setas = ttk.LabelFrame(tab_geral, text="Setas de Drenagem")
    frm_setas.pack(fill="x", padx=10, pady=8)
    
    v_buffer_setas = tk.DoubleVar(value=float(getattr(params_default, "setas_buffer_distancia", 5.0)))
    v_setas_curto = tk.IntVar(value=int(getattr(params_default, "setas_seg_curto_max", 3)))
    v_setas_medio = tk.IntVar(value=int(getattr(params_default, "setas_seg_medio_max", 4)))
    v_setas_longo = tk.IntVar(value=int(getattr(params_default, "setas_seg_longo_max", 5)))
    
    _row(frm_setas, 0, "Buffer entre setas (m):", v_buffer_setas, tip="Distância mínima entre setas para evitar sobreposição")
    _row(frm_setas, 1, "Setas - Segmento curto (≤30m):", v_setas_curto, tip="Máximo de setas para segmentos curtos")
    _row(frm_setas, 2, "Setas - Segmento médio (≤60m):", v_setas_medio, tip="Máximo de setas para segmentos médios")
    _row(frm_setas, 3, "Setas - Segmento longo (>60m):", v_setas_longo, tip="Máximo de setas para segmentos longos")

    frm_epsg_curvas.columnconfigure(0, weight=1)
    frm_epsg_curvas.columnconfigure(1, weight=1)

    # === TEXTOS ===
    def _get(p, d):
        try:
            return float(getattr(params_default, p))
        except Exception:
            return d

    frm_txt = ttk.Frame(tab_textos)
    frm_txt.pack(fill="x", padx=10, pady=8)
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

    # === Rodape ===
    frm_bottom = ttk.Frame(root)
    frm_bottom.pack(fill="x", padx=10, pady=10)
    out = {}

    def _ok():
        if not v_out.get():
            messagebox.showerror("Atencao", "Pasta de saida e obrigatoria.")
            return
        per_src_val = v_per_src.get().strip() or None
        epsg_mode = v_epsg_mode.get()
        epsg_custom_str = v_epsg_custom.get().strip()
        epsg_custom_val = None
        if epsg_mode == "custom" and epsg_custom_str.isdigit():
            try:
                epsg_custom_val = int(epsg_custom_str)
            except Exception:
                epsg_custom_val = None
        osm_cfg = {
            "epsg_mode": epsg_mode,
            "epsg_custom": epsg_custom_val,
            "user_agent": "SAI-BRASIL-Reurb/1.0 (contato: email@exemplo.com)",
            "inflate_bbox_m": 15.0,
            "expansion_steps_m": (0,),
            "fallback_around_m": 15,
        }
        vias_cfg = {
            "via_offset_texto": float(getattr(params_default, "via_offset_texto", 0.50)),
            "via_cross_span": float(getattr(params_default, "via_cross_span", 80.0)),
            "via_nome_offset_m": float(getattr(params_default, "via_nome_offset_m", 0.60)),
            "via_nome_offset_side": getattr(params_default, "via_nome_offset_side", "auto"),
            "via_nome_sufixo": getattr(params_default, "via_nome_sufixo", " (Asfalto)"),
            "via_nome_maiusculas": bool(getattr(params_default, "via_nome_maiusculas", False)),
            "via_offset_lote_lote_extra_m": float(getattr(params_default, "via_offset_lote_lote_extra_m", 0.60)),
            "via_nome_shift_along_m": float(getattr(params_default, "via_nome_shift_along_m", 6.0)),
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
            "curva_label_ends_only": bool(v_label_ends.get()),
        }
        setas_cfg = {
            "setas_buffer_distancia": float(v_buffer_setas.get()),
            "setas_seg_curto_max": int(v_setas_curto.get()),
            "setas_seg_medio_max": int(v_setas_medio.get()),
            "setas_seg_longo_max": int(v_setas_longo.get()),
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
                "vias": False,
                "per_source": per_src_val,
            },
            "osm": osm_cfg,
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
            try:
                on_execute(out)
            except Exception as e:
                messagebox.showerror("Erro", f"Falha na execucao: {e}")
            return
        root.destroy()

    ttk.Button(frm_bottom, text="Cancelar", command=root.destroy).pack(side="right", padx=6)
    ttk.Button(frm_bottom, text="Executar", command=_ok).pack(side="right", padx=6)

    root.mainloop()
    return out
