

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
# Change this to subset the data easily for running locally
SUBSET = False

# Change this to activate/decativate print statements throughout
DEBUG = True

# Change this for counties vs census tracts (true is muni, false is counties)
CENSUSTRACT = False

# include crops?
CROPLAND = True
############################################################

# set data directories (relative)
print(" - dataload file - packages loaded, setting directories") if (DEBUG == True) else ()
DATA_DIR = "data"
RESULTS_DIR = "results"


# def loadmsw(msw_shapefile):

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
msw_shapefile = "msw_2020/msw_2020.shp"

msw = gpd.read_file(opj(DATA_DIR,
				  msw_shapefile))

# filter to just keep food and green waste (subject of regulations)
msw = msw[(msw['subtype'] == "MSWfd_wet_dryad_wetad") | (msw['subtype'] == "MSWgn_dry_dryad")]

# MSW DATA NOTES: 
# fog = Fats, Oils, Grease; lb = lumber; cd = cardboard; fd = food;
# pp = paper, gn = green; ot = Other ; suffix describes what the 
# waste is deemed suitable for

# rename categories to be more intuitive
msw['subtype'].replace({'MSWfd_wet_dryad_wetad': 'MSW_food', 
						'MSWgn_dry_dryad': 'MSW_green'}, inplace = True)

# toggle for using county levels
if CENSUSTRACT == False:
	print("RUNNING ON COUNTIES (not census tracts)")
	# group by County and subtype
	grouped = msw.groupby(['County', 'subtype'], as_index=False).sum()

	# NEW COUNTY SHAPE
	counties = pd.read_csv(opj(DATA_DIR, 
				"counties/CenPop2010_Mean_CO06.txt")) # NEW - population weighted means!
	# rename lat and lon for easier plotting
	counties.rename(columns = {'LATITUDE': 'lat', 'LONGITUDE': 'lon', 'COUNAME': 'County'}, inplace=True)

	#merge and turn back into shapefile?
	# join counties and grouped
	df = grouped.merge(counties, on='County')

	# back to gdf
	msw = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
	msw.rename(columns = {'County' : 'muni_ID'}, inplace=True)
else:
	msw.rename(columns = {'ID' : 'muni_ID'}, inplace=True)

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
facilities = gpd.read_file(opj(DATA_DIR, "clean/clean_swis.shp"))
facilities.rename(columns={'County':'COUNTY'}, inplace=True)
# facilities = facilities.to_crs(epsg=4326)

# facilities = facilities[['SwisNo', 'AcceptedWa', 'COUNTY', 'cap_m3', 'geometry']].copy()
print("facility data loaded") if (DEBUG == True) else ()

############################################################
# RANGELANDS (ORIGINAL FMMP DATAFILE)
############################################################
# Import rangelands
# print("about to import rangelands") if (DEBUG == True) else ()
rangelands = gpd.read_file(opj(DATA_DIR, "raw/CA_FMMP_G/gl_bycounty/grazingland_county.shp"))
rangelands = rangelands.to_crs(epsg=4326) # make sure this is read in degrees (WGS84)

# Fix county names in RANGELANDS! 
countyIDs = pd.read_csv(opj(DATA_DIR, "interim/CA_FIPS_wcode.csv"), 
	names = ['FIPS', 'COUNTY', 'State', 'county_nam'])
countyIDs = countyIDs[['COUNTY', 'county_nam']]
rangelands = pd.merge(rangelands, countyIDs, on = 'county_nam')

# convert area capacity into volume capacity
rangelands['area_ha'] = rangelands['Shape_Area']/10000 # convert area in m2 to hectares
rangelands['capacity_m3'] = rangelands['area_ha'] * 63.5 # use this metric for m3 unit framework
# rangelands['capacity_ton'] = rangelands['area_ha'] * 37.1 # also calculated for tons unit framework

# estimate centroid
rangelands['centroid'] = rangelands['geometry'].centroid 
print("rangelands loaded") if (DEBUG == True) else ()


##############################################################
# NEW RANGELAND DATA #(CAL CONSERVATION DEPT)
# ##############################################################
# # UPDATE 08032020 -- bring in new rangelands (and keep naming convention)
# print("bringing in second rangeland data file") if (DEBUG == True) else ()
# rangelands = gpd.read_file(opj(DATA_DIR, "rangelandareas/ds553.shp"))
# rangelands = rangelands.to_crs(epsg=4326)

# rangelands['OBJECTID'] = rangelands.index


# # Each planning unit falls into one of 3 groups.
# # 0. Not included in priority areas.
# # 1. Important for rangeland goals - selected 3-7 times of 10.
# # 2. Critical for rangeland goals- selected 8-10 times


# # convert area capacity into volume capacity
# rangelands['area_ha'] = rangelands['Shape_area']/10000 # convert area in m2 to hectares
# rangelands['capacity_m3'] = rangelands['area_ha'] * 63.5 # use this metric for m3 unit framework
# # # estimate centroid
# rangelands['centroid'] = rangelands['geometry'].centroid 


# # optional - omit non-priority land:
# rangelands = rangelands[rangelands['Priority'] != 0]

# # run full model below and see if it changes numbers dramatically, 
# # then think through how to rename these such that it fits with croplands too. 
# # idea: rangelands have priority values, maybe crops could be assigned these?





############################################################
# SUBSET!! for testing functions
#############################################################
if SUBSET == True: 
	print("* create SUBSET of data for testing locally *")
	subset_size = 10
	
	msw = msw[0:(2*subset_size)]
	facilities = facilities[0:subset_size]
	rangelands = rangelands[0:subset_size]

############################################################
# raise Exception("data loaded - pre optimization")
############################################################

############################################################
# CROPLANDS
#############################################################
 
# # # Import croplands
if CROPLAND == True:
	croplands = cleancropdata(opj(DATA_DIR, 
		"raw/Crop__Mapping_2014-shp/Crop__Mapping_2014.shp"))
