## THIS SCRIPT READS IN ALL OF THE DATA NEEDED TO RUN THE COMPOSTLP SCRIPT
# when running locally (primarily for testing) set LOCAL = TRUE
# this will subset the data to run more quickly
# when running on AWS set LOCAL = FALSE
# this will run the full dataset

#changing a thing and changing it back


import cvxpy as cp
import numpy as np
import os
import datetime
from os.path import join as opj
import json
import sys


import pandas as pd
import shapely as shp
import geopandas as gpd
import scipy as sp

from california_cropland_cleaning import cleancropdata
# from biomass_preprocessing import MergeInventoryAndCounty
#from swis_preprocessing import LoadAndCleanSWIS #TODO


############################################################
# run on full data set or small sample (for testing!!)
SUBSET = True 

# Change this to activate/decativate print statements throughout
DEBUG = True

# Change this for counties vs census tracts (true is muni, false is counties)
CENSUSTRACT = False

############################################################

# set data directories (relative)
print(" - dataload file - packages loaded, setting directories") if (DEBUG == True) else ()
DATA_DIR = "data"
RESULTS_DIR = "results"


# def loadmsw(msw_shapefile):
def Fetch(df, key_col, key, value):
	#counties['disposal'].loc[counties['COUNTY']=='San Diego'].values[0]
	return df[value].loc[df[key_col]==key].values[0]


######################################################################################
### LOAD IN DATA ###
#######################################################################################

# # bring in biomass data
# gbm_pts, tbm_pts = MergeInventoryAndCounty(
#     gross_inventory     = opj(DATA_DIR, "raw/biomass.inventory.csv"),
#     technical_inventory = opj(DATA_DIR, "raw/biomass.inventory.technical.csv"),
#     county_shapefile    = opj(DATA_DIR, "raw/CA_Counties/CA_Counties_TIGER2016.shp"),
#     counties_popcen     = opj(DATA_DIR, "counties/CenPop2010_Mean_CO06.txt")
# )

# mini gdfs of county wastes (tbm - location and MSW for 2014) 
# counties = gpd.read_file(opj(DATA_DIR, "clean/techbiomass_pts.shp"))
# counties = counties.to_crs(epsg=4326)
# counties = tbm_pts # could change to GBM


##### Municipal Solid Waste (points) #####
print("about to load MSW points") if (DEBUG == True) else ()
msw_shapefile = "msw_2020_v2/msw_2020.shp"

msw = gpd.read_file(opj(DATA_DIR,
				  msw_shapefile))
msw.rename(columns = {'COUNTY' : 'muni_ID'}, inplace=True)

# raise Exception('msw loaded')

# # filter to just keep food and green waste (subject of regulations)
# msw = msw[(msw['subtype'] == "MSWfd_wet_dryad_wetad") | (msw['subtype'] == "MSWgn_dry_dryad")]

# # MSW DATA NOTES: 
# # fog = Fats, Oils, Grease; lb = lumber; cd = cardboard; fd = food;
# # pp = paper, gn = green; ot = Other ; suffix describes what the 
# # waste is deemed suitable for

# # rename categories to be more intuitive
# msw['subtype'].replace({'MSWfd_wet_dryad_wetad': 'MSW_food', 
# 						'MSWgn_dry_dryad': 'MSW_green'}, inplace = True)

# toggle for using county levels
# if CENSUSTRACT == False:
# 	print("RUNNING ON COUNTIES (not census tracts)")
# 	# group by County and subtype
# 	grouped = msw.groupby(['County', 'subtype'], as_index=False).sum()

# 	# NEW COUNTY SHAPE
# 	counties = pd.read_csv(opj(DATA_DIR, 
# 				"counties/CenPop2010_Mean_CO06.txt")) # NEW - population weighted means!
# 	# rename lat and lon for easier plotting
# 	counties.rename(columns = {'LATITUDE': 'lat', 'LONGITUDE': 'lon', 'COUNAME': 'County'}, inplace=True)

# 	#merge and turn back into shapefile?
# 	# join counties and grouped
# 	df = grouped.merge(counties, on='County')

# 	# back to gdf
# 	msw = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
# 	msw.rename(columns = {'County' : 'muni_ID'}, inplace=True)
# else:
# 	msw.rename(columns = {'ID' : 'muni_ID'}, inplace=True)

# ADJUST VALUES ## FLAG!!!!!!!!!
print("MSW loaded - now adjust to get wt") if (DEBUG == True) else ()
# I'll be using 100% of the GREEN waste, so leave as is
# for FOOD WASTE, take off 2.5%, 
# then of the remainer divert 62.5% of generation to compost
# (assume that 25% goes straight to compost, 75% goes to AD, which reduces volume of material by half, 
# before being composted)
# equivlant to 0.609375
# create new array of values
# new_wt_values = msw[msw['subtype'] == 'MSW_food']['wt']*0.609375
# # replace these in place!
# msw.loc[msw['subtype'] == 'MSW_food', 'wt'] = new_wt_values
###################################
############################################################


############################################################
# COMPOSTING/PROCESSING FACILITIES
############################################################

print("about to load facility data") if (DEBUG == True) else ()
# Load facility info
facilities = gpd.read_file(opj(DATA_DIR, "swis/clean_swis.shp"))
facilities.rename(columns={'County':'COUNTY'}, inplace=True)
# facilities = facilities.to_crs(epsg=4326)

# facilities = facilities[['SwisNo', 'AcceptedWa', 'COUNTY', 'cap_m3', 'geometry']].copy()
print("facility data loaded") if (DEBUG == True) else ()

############################################################
# RANGELANDS (ORIGINAL FMMP DATAFILE)
############################################################
# Import rangelands
# print("about to import rangelands") if (DEBUG == True) else ()
rangelands = gpd.read_file(opj(DATA_DIR, "rangelands/FMMP/grazingland_county.shp"))
rangelands = rangelands.to_crs(epsg=4326) # make sure this is read in degrees (WGS84)

# Fix county names in RANGELANDS! 
countyIDs = pd.read_csv(opj(DATA_DIR, "counties/CA_FIPS_wcode.csv"), 
	names = ['FIPS', 'COUNTY', 'State', 'county_nam'])
countyIDs = countyIDs[['COUNTY', 'county_nam']]
rangelands = pd.merge(rangelands, countyIDs, on = 'county_nam')

# convert area capacity into volume capacity
rangelands['area_ha'] = rangelands['Shape_Area']/10000 # convert area in m2 to hectares
rangelands['capacity_m3'] = rangelands['area_ha'] * 63.5 # use this metric for m3 unit framework
# rangelands['capacity_ton'] = rangelands['area_ha'] * 37.1 # also calculated for tons unit frameworkcropl

# estimate centroid
# rangelands_proj = rangelands.to_crs(epsg=3310) # change to projected crs for getting centroid
rangelands['centroid'] = rangelands['geometry'].centroid 

# rangelands = rangelands.to_crs(epsg=4326) # make sure this is read in degrees (WGS84)
print("rangelands loaded") if (DEBUG == True) else ()


##############################################################
# NEW RANGELAND DATA #(CAL CONSERVATION DEPT)
# ##############################################################
# # # UPDATE 08032020 -- bring in new rangelands (and keep naming convention)
# print("bringing in second rangeland data file") if (DEBUG == True) else ()
# rangelands = gpd.read_file(opj(DATA_DIR, "rangelands/DOC/ds553_counties.shp"))
# rangelands = rangelands.to_crs(epsg=4326)

# rangelands['OBJECTID'] = rangelands.index
# rangelands['COUNTY'] = rangelands['NAME']

# # Each planning unit falls into one of 3 groups.
# # 0. Not included in priority areas.
# # 1. Important for rangeland goals - selected 3-7 times of 10.
# # 2. Critical for rangeland goals- selected 8-10 times


# # convert area capacity into volume capacity
# rangelands['area_ha'] = rangelands['Shape_area']/10000 # convert area in m2 to hectares
# rangelands['capacity_m3'] = rangelands['area_ha'] * 63.5 # use this metric for m3 unit framework
# # # estimate centroid
# # rangelands_temp = rangelands.to_crs(epsg=3310) # change to projected crs for getting centroid
# rangelands['centroid'] = rangelands['geometry'].centroid 
# # rangelands = rangelands.to_crs(epsg=4326) # make sure this is read in degrees (WGS84)



# # optional - omit non-priority land:
# rangelands = rangelands[rangelands['Priority'] != 1]

# # # run full model below and see if it changes numbers dramatically, 
# # # then think through how to rename these such that it fits with croplands too. 
# # # idea: rangelands have priority values, maybe crops could be assigned these?

############################################################
# CROPLANDS
#############################################################
 
# # # # Import croplands
# croplands = cleancropdata(opj(DATA_DIR, 
# 	"crops/Crop__Mapping_2014-shp/Crop__Mapping_2014.shp"))
# print("croplands loaded")

############################################################
# SEQUESTRATION / EMISSIONS REDUCTION RATES
#############################################################


perennial_file = "compostrates/perennial_CN_high.csv"
annual_file = "compostrates/annual_CN_high.csv"
grazed_file = "compostrates/grazedgrasslands_CN_high.csv"


value =  'GHG Emissions'

# print("reading perennial") if (DEBUG == True) else ()
# perennial_rates = pd.read_csv(opj(DATA_DIR,
# 						perennial_file))
# perennial_rates['seq_f'] = perennial_rates['GHG Emissions'] / 0.404686 / 63.5 / 0.001

# # for county in perennial_rates['County']:
# # 	rate = Fetch(perennial_rates, 'County', county, value)



# # print("reading annual")
# # annual_rates = pd.read_csv(opj(DATA_DIR,
# # 						annual_file))
# # annual_rates['seq_f'] = annual_rates['GHG Emissions'] / 0.404686 / 63.5 / 0.001

# # # for county in annual_rates['County']:
# # # 	rate = Fetch(annual_rates, 'County', county, value)

print("reading grazed seq rates")
grazed_rates = pd.read_csv(opj(DATA_DIR, 
	grazed_file))
grazed_rates['seq_f'] = grazed_rates['GHG Emissions'] / 0.404686 / 63.5 / 0.001

# for county in grazed_rates['County']:
# 	rate = Fetch(grazed_rates, 'County', county, value)
# 	print(rate)

#  - change rates in the file here, then load into the compostLP fucntions

# the rates above are MTCO2e/acre/year
# want to get to kgCO2e/m3/year
# conversion:
# MTCO2e/acre * (1acre/0.404686hectares) * (1 hectare / 63.5 cm3 compost) * (1 kg / 0.001 MT)
# seq_f = rate / 0.404686 / 63.5 / 0.001

############################################################
# SUBSET!! for testing functions
#############################################################
if SUBSET == True: 

	subset_size = 20

	print("* create SUBSET of data ( N=", subset_size, ") for testing locally *" )
	
	msw = msw[0:(2*subset_size)]
	facilities = facilities[0:subset_size]
	rangelands = rangelands[0:subset_size]
	# croplands = croplands[0:subset_size]

############################################################
# raise Exception("data loaded - pre optimization")
############################################################

print("all data loaded")

