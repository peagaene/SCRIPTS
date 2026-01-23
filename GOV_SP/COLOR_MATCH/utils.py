from osgeo import gdal, osr
import numpy as np


def _gdal_dtype_from_numpy(dtype):
    if np.issubdtype(dtype, np.uint8):
        return gdal.GDT_Byte
    if np.issubdtype(dtype, np.uint16):
        return gdal.GDT_UInt16
    if np.issubdtype(dtype, np.int16):
        return gdal.GDT_Int16
    if np.issubdtype(dtype, np.uint32):
        return gdal.GDT_UInt32
    if np.issubdtype(dtype, np.int32):
        return gdal.GDT_Int32
    if np.issubdtype(dtype, np.float32):
        return gdal.GDT_Float32
    if np.issubdtype(dtype, np.float64):
        return gdal.GDT_Float64
    return gdal.GDT_Byte


def save_image_with_epsg(image, output_path, epsg=31983, geotransform=None, projection=None, data_type=None):
    """
    Save a NumPy image with georeferencing (GeoTransform and Projection) as GeoTIFF.
    When projection is not provided, the EPSG code is used.
    """
    height, width, channels = image.shape

    driver = gdal.GetDriverByName("GTiff")
    if data_type is None:
        data_type = _gdal_dtype_from_numpy(image.dtype)
    dataset = driver.Create(output_path, width, height, channels, data_type)

    if projection is None:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        projection = srs.ExportToWkt()
    dataset.SetProjection(projection)

    for i in range(channels):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(image[:, :, i])

    if geotransform:
        dataset.SetGeoTransform(geotransform)

    dataset.FlushCache()
    dataset = None
