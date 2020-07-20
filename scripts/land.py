
import pandas as pd
import os
import numpy as np
import geopandas as gpd
from os.path import join as opj

DATA_DIR = "/Users/anayahall/projects/compopt/data"

###################################################################################
# ORIGINAL RUNS - import ind rangelands, fix county names and get centroid
# Import rangelands
rangelands = gpd.read_file(opj(DATA_DIR, "raw/CA_FMMP_G/gl_bycounty/grazingland_county.shp"))
rangelands = rangelands.to_crs(epsg=4326) # make sure this is read in degrees (WGS84)

# Fix county names! 
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

###################################################################################
# TEMP FIX- assigns ind rangeland area to county centroid
rangelands = gpd.read_file(opj(DATA_DIR, "raw/CA_FMMP_G/gl_bycounty/grazingland_county.shp"))
rangelands = rangelands.to_crs(epsg=4326) # make sure this is read in degrees (WGS84)
rangelands['area_ha'] = rangelands['Shape_Area']/10000 # convert area in m2 to hectares
rangelands['capacity_m3'] = rangelands['area_ha'] * 63.5 # use this metric for m3 unit framework

rangelands = rangelands[['county_nam', 'area_ha', 'capacity_m3']]

countyIDs = pd.read_csv(opj(DATA_DIR, "interim/CA_FIPS_wcode.csv"), 
    names = ['FIPS', 'COUNTY', 'State', 'county_nam'])
countyIDs = countyIDs[['COUNTY', 'county_nam']]
rangelands = pd.merge(rangelands, countyIDs, on = 'county_nam')

rangelands = rangelands[['COUNTY', 'area_ha', 'capacity_m3']]


CA = gpd.read_file(opj(DATA_DIR, "raw/CA_Counties/CA_Counties_TIGER2016.shp"))
CA= CA.to_crs(epsg=4326)
CA['county_centroid'] = CA['geometry'].centroid
CA['COUNTY'] = CA['NAME']
CA = CA[['COUNTY', 'county_centroid']]


rangelands = pd.merge(CA, rangelands, on = 'COUNTY')



###################################################################################
# ALTERNATE RANGELAND FILE
# rangelands = gpd.read_file(opj(DATA_DIR, "raw/CA_FMMP_G/grazingland_dis/CA_grazingland.shp"))

# convert area capacity into volume capacity
rangelands['area_ha'] = rangelands['Shape_Area']/10000 # convert area in m2 to hectares
rangelands['capacity_m3'] = rangelands['area_ha'] * 63.5 # use this metric for m3 unit framework

# estimate centroid
rangelands['centroid'] = rangelands['geometry'].centroid 


