# mdt_utils.py
import rasterio, numpy as np

def make_get_elevation(path_mdt):
    src = rasterio.open(path_mdt)
    transform = src.transform; arr = src.read(1); nodata = src.nodata
    def get_el(x,y):
        c,r = ~transform * (x,y); c,r = int(c),int(r)
        if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
            v=float(arr[r,c]); 
            if nodata is not None and np.isclose(v, nodata): return None
            return round(v,3)
        return None
    return src, get_el
