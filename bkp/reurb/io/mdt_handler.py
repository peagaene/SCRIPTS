"""
MDT handling utilities.
Migrated from reurb_auto_all.py.
"""
from __future__ import annotations

import numpy as np
import rasterio


def make_get_elevation(path_mdt):
    src = rasterio.open(path_mdt)
    transform = src.transform
    arr = src.read(1)
    nodata = src.nodata

    def get_el(x, y):
        c, r = ~transform * (x, y)
        c, r = int(c), int(r)
        if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
            v = float(arr[r, c])
            if nodata is not None and np.isclose(v, nodata):
                return None
            return round(v, 3)
        return None

    return src, get_el


__all__ = ["make_get_elevation"]
