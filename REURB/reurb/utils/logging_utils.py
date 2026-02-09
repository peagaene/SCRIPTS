"""
Logging utilities.
"""
from __future__ import annotations

import logging
from typing import Optional


class REURBLogger:
    """Logger centralizado com contexto."""

    def __init__(self, name: str = "REURB", verbose: bool = True):
        self.logger = logging.getLogger(name)
        self.verbose = bool(verbose)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

    def info(self, msg: str, **kwargs) -> None:
        self.logger.info(msg, extra=kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        self.logger.warning(msg, extra=kwargs)

    def error(self, msg: str, exc_info: bool = False, **kwargs) -> None:
        self.logger.error(msg, exc_info=exc_info, extra=kwargs)

    def debug(self, msg: str, **kwargs) -> None:
        if self.verbose:
            self.logger.debug(msg, extra=kwargs)


def log(msg: str, verbose: bool = True) -> None:
    """Printa mensagem quando verbose=True (compatibilidade)."""
    if verbose:
        print(msg)


__all__ = ["REURBLogger", "log"]
