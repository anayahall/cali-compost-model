# clean CROP_MAPPING_2014 to focus on high value crops
# anaya hall
# updated july 2020

from os.path import join as opj
import geopandas as gpd


DATA_DIR = "data"

cropdata_shapefile = "crops/Crop__Mapping_2014-shp/Crop__Mapping_2014.shp"

############################################################
# CROPLANDS
#############################################################
def cleancropdata(cropdata):
	# Read in cropland data
	cropmap = gpd.read_file(opj(DATA_DIR,
					  cropdata_shapefile))

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

	croplands['centroid'] = croplands['geometry'].centroid 
	# get cropland capacity: rated as 9t/ha for treecrops
	# 0.58 tons per cubic meter
	# 9 tons per hectare
	# m3 = acres * (ha/acres) * (t/ha) * (m3/t)
	croplands['capacity_m3'] = croplands['Acres'] * (0.404686) * (9) * (1/0.58)
	croplands.reset_index(inplace = True, drop = True)

		## Save as shapefile
	# out = r"clean/CropMap2014_clean.shp"
	# croplands.to_file(driver='ESRI Shapefile', filename="treecrops.shp")


	# gdf = croplands
	return croplands
# done! 

# croplands_dissolved = croplands.dissolve(by = 'County')


# x = gpd.GeoDataFrame(croplands.groupby('County')).dissolve('Crop2014')



