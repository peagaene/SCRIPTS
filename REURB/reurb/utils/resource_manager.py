"""
Resource management utilities.
"""
from __future__ import annotations

from contextlib import contextmanager

from reurb.utils.logging_utils import REURBLogger

logger = REURBLogger(__name__, verbose=False)


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
                except Exception as e:
                    logger.debug(f"Falha ao fechar MDT: {e}")


__all__ = ["ResourceManager"]
