"""
Resource management utilities.
"""
from __future__ import annotations

from contextlib import contextmanager


class ResourceManager:
    @contextmanager
    def managed_mdt(self, path: str):
        """Context manager que garante fechamento do MDT."""
        src = None
        try:
            import rasterio
            src = rasterio.open(path)
            yield src
        finally:
            if src:
                try:
                    src.close()
                except Exception:
                    pass


__all__ = ["ResourceManager"]
