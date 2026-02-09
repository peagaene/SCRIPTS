"""
UI log stream helpers.
"""
from __future__ import annotations

import queue
import tkinter as tk

from reurb.utils.logging_utils import REURBLogger

logger = REURBLogger(__name__, verbose=False)


class _UILogStream:
    def __init__(self, widget: tk.Text):
        self.widget = widget

    def write(self, s: str):
        if not s:
            return
        try:
            self.widget.insert("end", s)
            self.widget.see("end")
            self.widget.update_idletasks()
        except Exception as e:
            logger.debug(f"Falha ao escrever no widget de log: {e}")

    def flush(self):
        pass


class _QueueLogStream:
    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, s: str):
        if s:
            self.q.put(s)

    def flush(self):
        pass


__all__ = ["_UILogStream", "_QueueLogStream"]
