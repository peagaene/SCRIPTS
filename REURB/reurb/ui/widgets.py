"""
UI widgets and helpers.
"""
from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import filedialog, ttk

from reurb.utils.logging_utils import REURBLogger

_CFG = os.path.join(os.path.expanduser("~"), ".itesp_reurb_ui.json")
logger = REURBLogger(__name__, verbose=False)


def _load_state():
    try:
        with open(_CFG, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"Falha ao carregar estado da UI: {e}")
        return {}


def _save_state(state: dict):
    try:
        with open(_CFG, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"Falha ao salvar estado da UI: {e}")


class ToolTip:
    def __init__(self, widget, text, wrap=56):
        self.widget = widget
        self.text = text
        self.wrap = wrap
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, *_):
        if self.tip or not self.text:
            return
        x, y = self.widget.winfo_pointerxy()
        tw = tk.Toplevel(self.widget)
        self.tip = tw
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x + 14}+{y + 14}")
        lbl = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#111",
            foreground="#fff",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6,
            wraplength=self.wrap * 8,
            font=("Segoe UI", 9),
        )
        lbl.pack()

    def _hide(self, *_):
        if self.tip:
            self.tip.destroy()
            self.tip = None


def _row(parent, r, label, var, width=12, unit="", tip=""):
    ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=4)
    ent = ttk.Entry(parent, textvariable=var, width=width)
    ent.grid(row=r, column=1, sticky="we", padx=6, pady=4)
    ttk.Label(parent, text=unit).grid(row=r, column=2, sticky="w", padx=2)
    if tip:
        ToolTip(ent, tip)
    return ent


def _browse_entry(parent, r, label, var, kind="file", patterns=(("Todos", "*.*"),), tip=""):
    ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=4)
    ent = ttk.Entry(parent, textvariable=var, width=64)
    ent.grid(row=r, column=1, sticky="we", padx=6, pady=4, columnspan=2)

    def go():
        init = (
            os.path.dirname(var.get())
            if var.get()
            else _load_state().get("lastdir", os.path.expanduser("~"))
        )
        path = (
            filedialog.askopenfilename(parent=parent, initialdir=init, filetypes=patterns)
            if kind == "file"
            else filedialog.askdirectory(parent=parent, initialdir=init)
        )
        if path:
            var.set(path)
            _save_state({"lastdir": os.path.dirname(path)})

    ttk.Button(parent, text="...", width=3, command=go).grid(row=r, column=3, padx=4)
    if tip:
        ToolTip(ent, tip)
    return ent


__all__ = ["ToolTip", "_row", "_browse_entry"]
