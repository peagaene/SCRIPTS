import os
import warnings
import arcpy

#Remover os avisos
#warnings.filterwarnings('ignore')
output_rgb_pasta = r'\\192.168.2.28\g\5_ORTOMOSAICOS\SI_01\2_IR\1_GEOTIFF'
nome_area = 'SI_01_ORTOFOTO_IR' #MUDAR PRA IR SE NECESSÁRIO
nome_dir_prj_i = f'D:\esri_31983.prj'

#Criar um mosaic dataset usando o arcpy
dir_gdb = r'\\192.168.2.28\g\5_ORTOMOSAICOS\SI_01\2_IR\3_GBD'
arcpy.env.workspace = dir_gdb

#Criação de File Geodatabase (.gdb)
arcpy.CreateFileGDB_management(dir_gdb, f'{nome_area}.gdb')

#Criação de um mosaico dataset
prj = nome_dir_prj_i
arcpy.CreateMosaicDataset_management(f'{nome_area}.gdb/', f'Mosaic_{nome_area}', nome_dir_prj_i, "3", "8_BIT_UNSIGNED", "NONE", "")

#Adicionar imagens ao Mosaic Dataset
mdname = f"{nome_area}.gdb/Mosaic_{nome_area}"
rastype = "Raster Dataset"
inpath = output_rgb_pasta
updatecs = "UPDATE_CELL_SIZES"
updatebnd = "UPDATE_BOUNDARY"
updateovr = "NO_OVERVIEWS" 
maxlevel = "-1"
maxcs = "#"
maxdim = "#"
spatialref = nome_dir_prj_i
inputdatafilter = "*.tif"
subfolder = "NO_SUBFOLDERS"
duplicate = "EXCLUDE_DUPLICATES"
buildpy = "NO_PYRAMIDS" #"BUILD_PYRAMIDS"
calcstats = "NO_STATISTICS"
buildthumb = "NO_THUMBNAILS"
comments = "Add Raster Datasets"
forcesr = "#"

arcpy.AddRastersToMosaicDataset_management(
     mdname,  rastype, inpath, updatecs, updatebnd, updateovr,
     maxlevel, maxcs, maxdim, spatialref, inputdatafilter,
     subfolder, duplicate, buildpy, calcstats, 
     buildthumb, comments, forcesr)