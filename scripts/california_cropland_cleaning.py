# clean CROP_MAPPING_2014 to exclude non-crop landuses
# anaya hall
# updated aug. 13, 2019

import numpy as np
import os
from os.path import join as opj
import pandas as pd
import shapely as shp
import geopandas as gpd
import scipy as sp

# Set data directory -- CHANGE THIS FOR YOUR LOCAL DEVICE
DATA_DIR = "/Users/anayahall/projects/compopt/data" 

############################################################
# CROPLANDS
#############################################################
 
# Read in cropland data
cropmap = gpd.read_file(opj(DATA_DIR, 
  "raw/Crop__Mapping_2014/Crop__Mapping_2014.shp")) 

# Exclude non-crop uses
# non_crops = ["Managed Wetland", "Urban", "Idle", "Mixed Pasture"]	#Anaya's original categories to exclude
non_crops = ["NR | RIPARIAN VEGETATION", "U | URBAN", "V | VINEYARD"] #Caitlin's categories to exclued
# crops = cropmap[cropmap['Crop2014'].isin(non_crops)== False]	#Anaya's field
crops = cropmap[cropmap['DWR_Standa'].isin(non_crops)== False] # Caitlin's field is DWR_Standa


## Save as shapefile
out = r"clean/CropMap2014_clean.shp"
crops.to_file(driver='ESRI Shapefile', filename=opj(DATA_DIR, out))


# done! 