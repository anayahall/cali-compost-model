# SANITY CHECK TRANSPORT DISTANCES- WHY ARE THEY SO EVEN???

import numpy as np
import os
import datetime
from os.path import join as opj

import pandas as pd
import shapely as shp
import geopandas as gpd
import scipy as sp
import shapely as shp
import json

from biomass_preprocessing import MergeInventoryAndCounty


DATA_DIR = "/Users/anayahall/projects/compopt/data"

############################################################
# FUNCTIONS USED IN THIS SCRIPT

def Haversine(lat1, lon1, lat2, lon2):
  """
  Calculate the Great Circle distance on Earth between two latitude-longitude
  points
  :param lat1 Latitude of Point 1 in degrees
  :param lon1 Longtiude of Point 1 in degrees
  :param lat2 Latitude of Point 2 in degrees
  :param lon2 Longtiude of Point 2 in degrees
  :returns Distance between the two points in kilometres
  """
  Rearth = 6371
  lat1   = np.radians(lat1)
  lon1   = np.radians(lon1)
  lat2   = np.radians(lat2)
  lon2   = np.radians(lon2)
  #Haversine formula 
  dlon = lon2 - lon1 
  dlat = lat2 - lat1 
  a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
  c = 2 * np.arcsin(np.sqrt(a)) 
  return Rearth*c


def Distance(loc1, loc2):
    # print(loc1.x, loc1.y, loc2.x, loc2.y)
    return Haversine(loc1.y, loc1.x, loc2.y, loc2.x)


def Fetch(df, key_col, key, value):
    #counties['disposal'].loc[counties['COUNTY']=='San Diego'].values[0]
    return df[value].loc[df[key_col]==key].values[0]

############################################################
# LOAD DATA

# bring in biomass data
gbm_pts, tbm_pts = MergeInventoryAndCounty(
    gross_inventory     = opj(DATA_DIR, "raw/biomass.inventory.csv"),
    technical_inventory = opj(DATA_DIR, "raw/biomass.inventory.technical.csv"),
    county_shapefile    = opj(DATA_DIR, "raw/CA_Counties/CA_Counties_TIGER2016.shp"),
    counties_popcen     = opj(DATA_DIR, "counties/CenPop2010_Mean_CO06.txt")
)

counties = tbm_pts # could change to GBM

# facilities
facilities = gpd.read_file(opj(DATA_DIR, "clean/clean_swis.shp"))
# facilities = facilities.to_crs(epsg=4326)
# facilities = facilities[0:10]

######################################################################
# RANGELAND FIX!
######################################################################

# Import rangelands
# (partially dissolved subset)
# rangelands = gpd.read_file(opj(DATA_DIR, "raw/CA_FMMP_G/grazingland_dis/CA_grazingland.shp"))


# (all individual rangelands)
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

# keep only needed attributes
# rangelands = rangelands[['COUNTY', 'geometry', 'capacity_m3', 'area_ha']]

# # Dissolve into single polygon per county
# rangelands = rangelands.dissolve(by='COUNTY')

# raise Exception("data loaded- pre distance calc --- test output of cloc = Fetch(counties, 'COUNTY', county, 'county_centroid')")


# gdf = geopandas.GeoDataFrame(
#     df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))



# estimate centroid
rangelands['centroid'] = rangelands['geometry'].centroid 

detour_factor = 1.4

c2f = {}
for county in counties['COUNTY']:
    c2f[county] = {}
    # print(county)
    cloc = Fetch(counties, 'COUNTY', county, 'county_centroid')
    # print(cloc)
    for facility in facilities['SwisNo']:
        floc = Fetch(facilities, 'SwisNo', facility, 'geometry')
        # print(facility)
        # print(floc)
        c2f[county][facility] = {}
        c2f[county][facility]['trans_dist'] = Distance(cloc,floc)*detour_factor
        # print(c2f[county][facility]['trans_dist'])

print("C2f distances calculated")

f2r = {}
for facility in facilities['SwisNo']:
    f2r[facility] = {}
    print(facility)
    floc = Fetch(facilities, 'SwisNo', facility, 'geometry')
    for rangeland in rangelands['OBJECTID']:
        r_string = str(rangeland)
        rloc = Fetch(rangelands, 'OBJECTID', rangeland, 'centroid')
        f2r[facility][r_string] = {}
        f2r[facility][r_string]['trans_dist'] = Distance(floc,rloc)*detour_factor
        print(f2r[facility][r_string]['trans_dist'])

print("f2r distances calculated")

avgDict_f2r = {}
for facility in f2r.keys():
    # print(facility)
    temp = 0
    avgDict_f2r[facility] = {}
    avgDict_f2r[facility]['SwisNo'] = facility
    for rangeland in f2r[facility].keys():
        r_string = str(rangeland)
        temp += f2r[facility][r_string]['trans_dist']
        # print(temp) 
    avgDict_f2r[facility]['avg_dist'] = temp*(1/116)

avgDict_c2f = {}
for county in c2f.keys():
    # print(county)
    temp = 0
    avgDict_c2f[county] = {}
    avgDict_c2f[county]['COUNTY'] = county
    for facility in c2f[county].keys():
        temp += c2f[county][facility]['trans_dist']
        # print(temp) 
    avgDict_c2f[county]['avg_dist'] = temp*(1/109)

# fp = "/Users/anayahall/projects/compopt/data"


with open('c2f_DIST.json', 'w') as fp:
    json.dump(c2f, fp)

with open('f2r_DIST.json', 'w') as fp:
    json.dump(f2r, fp)



################################




