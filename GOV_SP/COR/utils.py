from osgeo import gdal, osr

def save_image_with_epsg(image, output_path, epsg=31983, geotransform=None, projection=None):
    """
    Salva uma imagem (array NumPy) com georreferenciamento (GeoTransform e Projeção)
    em um arquivo GeoTIFF. Se não for passada uma projeção, usa o EPSG especificado.
    """
    
    # 1) Extrai dimensões
    height, width, channels = image.shape
    
    # 2) Obtém o driver "GTiff" (GeoTIFF) do GDAL e cria o dataset de saída
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, width, height, channels, gdal.GDT_Byte)
    
    # 3) Define a projeção (CRS)
    # Se o parâmetro 'projection' não for fornecido,
    # constrói a projeção a partir de um EPSG (padrão = 31983).
    if projection is None:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)     # Importa o EPSG
        projection = srs.ExportToWkt()
    dataset.SetProjection(projection)
    
    # 4) Escreve os dados da imagem em cada banda do dataset
    # Para cada canal (R, G, B, etc.), grava a matriz correspondente.
    for i in range(channels):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(image[:, :, i])
    
    # 5) Se houver GeoTransform fornecido (ex.: [origem_x, tamanho_px_x, rot, origem_y, rot, tamanho_px_y]),
    #    aplica ao dataset para posicionar corretamente no espaço.
    if geotransform:
        dataset.SetGeoTransform(geotransform)
    
    # 6) Salva (flush) e fecha o arquivo
    dataset.FlushCache()
    dataset = None
