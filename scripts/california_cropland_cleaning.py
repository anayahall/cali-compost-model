# clean CROP_MAPPING_2014 to focus on high value crops
# anaya hall
# updated july 2020

from os.path import join as opj
import geopandas as gpd


# Set data directory -- CHANGE THIS FOR YOUR LOCAL DEVICE
# DATA_DIR = "/Users/anayahall/projects/cali-compost-model/data" 

############################################################
# CROPLANDS
#############################################################
def cleancropdata(cropdata):
	# Read in cropland data
	cropmap = gpd.read_file(cropdata) 

	# Exclude non-crop uses
	non_crops = ["Managed Wetland", "Urban", "Idle", "Mixed Pasture"]	#Anaya's original categories to exclude
	# non_crops = ["NR | RIPARIAN VEGETATION", "U | URBAN", "V | VINEYARD"] #Caitlin's categories to exclued
	# crops = cropmap[cropmap['Crop2014'].isin(non_crops)== False]	#Anaya's field
	# crops = cropmap[cropmap['DWR_Standa'].isin(non_crops)== False] # Caitlin's field is DWR_Standa

	# Keep HIGH VALUE CROPS ONLY
	 
	# by CROP2014
	highvaluecrops = ["Pomegranates", "Pears", "Dates", "Pomegranates", "Apples", "Cherries", 
					"Olives", "Plums, Prunes and Apricots", "Pistachios", "Avocados", "Walnuts", 
					"Citrus", "Almonds", "Grapes"]

	# OR by DWR_Standa
	# highvaluecrops = ["D | DECIDUOUS FRUITS AND NUT", "V | VINEYARD", "T | TRUCK NURSERY AND BERRY CROPS", "C | CITRUS AND SUBTROPICAL"]

	# croplands = cropmap[cropmap['DWR_Standa'].isin(highvaluecrops)==True]
	croplands = cropmap[cropmap['Crop2014'].isin(highvaluecrops)==True]

	## Save as shapefile
	# out = r"clean/CropMap2014_clean.shp"
	# croplands.to_file(driver='ESRI Shapefile', filename=opj(DATA_DIR, out))

	return croplands
# done! 

