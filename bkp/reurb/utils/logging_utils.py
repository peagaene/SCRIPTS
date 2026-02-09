"""
Logging utilities.
"""
from __future__ import annotations


def log(msg: str, verbose: bool = True) -> None:
    """Printa mensagem quando verbose=True."""
    if verbose:
        print(msg)


__all__ = ["log"]
