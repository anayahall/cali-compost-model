import cvxpy as cp
import numpy as np
import os
from os.path import join as opj

import pandas as pd
import shapely as shp
import geopandas as gpd

import scipy as sp

from biomass_preprocessing import MergeInventoryAndCounty

DATA_DIR = "/Users/anayahall/projects/grapevine/data"


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
    return Haversine(loc1.y, loc1.x, loc2.y, loc2.x)


def Fetch(df, key_col, key, value):
    #counties['disposal.y'].loc[counties['COUNTY']=='San Diego'].values[0]
    return df[value].loc[df[key_col]==key].values[0]


gbm_pts, tbm_pts = MergeInventoryAndCounty(
	gross_inventory     = opj(DATA_DIR, "raw/biomass.inventory.csv"),
	technical_inventory = opj(DATA_DIR, "raw/biomass.inventory.technical.csv"),
	county_shapefile    = opj(DATA_DIR, "raw/CA_Counties/CA_Counties_TIGER2016.shp"),
	fips_data           = opj(DATA_DIR, "interim/CA_FIPS.csv")
)

# mini gdfs of county wastes (tbm - location and MSW for 2014) 
counties = gpd.read_file(opj(DATA_DIR, "clean/techbiomass_pts.shp"))

#counties = counties[(counties['biomass.ca'] == "organic fraction municipal solid waste") & (counties['year'] == 2014)].copy()
counties = counties[(counties['biomass.fe'] == "FOOD") & (counties['year'] == 2014)].copy()

counties = counties[['FIPS', 'COUNTY', 'disposal.y', 'geometry']]
# subset out four counties
counties = counties[(counties['COUNTY'] == "Los Angeles") | (counties['COUNTY'] == "San Diego")| (counties['COUNTY'] == "Orange")| (counties['COUNTY'] == "Imperial")]

####MAKE DICTIONARY HERE
cdict = dict(zip(counties['COUNTY'], counties['disposal.y']))

# Mini gdfs of facilites (location and capacity)
facilities = gpd.read_file(opj(DATA_DIR, "clean/clean_swis.shp"))
# facilities.head(8)

facilities = facilities[['SwisNo', 'AcceptedWa', 'County', 'cap_m3', 'geometry']].copy()

# subset out four counties
facilities = facilities[(facilities['County'] == "Los Angeles") | (facilities['County'] == "San Diego") | 
          (facilities['County'] == "Orange")| (facilities['County'] == "Imperial")].copy()
# too many, just select first 5
facilities = facilities[0:5]

# Import rangelands
rangelands = gpd.read_file(opj(DATA_DIR, "raw/CA_FMMP_G/gl_bycounty/grazingland_county.shp"))

# get area in kilometers
rangelands["area_km"] = rangelands['geometry'].area/ 10**6

# estimate centroid
rangelands['centroid'] = rangelands['geometry'].centroid


# OPTIMIZATION #################################

#Variables
# constants
landfill_ef = 1 #TODO
compost_ef = 0.5
kilometres_to_emissions = 1
spreader_ef = 1
seq_f = -40
conversion = 0.58

# decision variables
# proportion of county waste to send to a facility 
c2f = {}
for county in counties['COUNTY']:
    c2f[county] = {}
    cloc = Fetch(counties, 'COUNTY', county, 'geometry')
    for facility in facilities['SwisNo']:
        floc = Fetch(facilities, 'SwisNo', facility, 'geometry')
        c2f[county][facility] = {}
        c2f[county][facility]['prop'] = cp.Variable()
        c2f[county][facility]['trans_cost'] = Distance(cloc,floc)*kilometres_to_emissions

# proportion of compost to send to rangeland 
f2r = {}
for facility in facilities['SwisNo']:
    f2r[facility] = {}
    floc = Fetch(facilities, 'SwisNo', facility, 'geometry')
    for rangeland in rangelands['OBJECTID']:
        rloc = Fetch(rangelands, 'OBJECTID', rangeland, 'centroid')
        f2r[facility][rangeland] = {}
        f2r[facility][rangeland]['prop'] = cp.Variable()
        f2r[facility][rangeland]['trans_cost'] = Distance(floc,rloc)*kilometres_to_emissions

# intake of facility - sum of material a facility recieves 
intake = {}
for facility in facilities['SwisNo']:
    intake[facility] = cp.Variable()





#BUILD OBJECTIVE FUNCTION: we want to minimize emissions
obj = 0

# emissions due to waste remaining in county
for county in counties['COUNTY']:
    temp = 0
    for facility in facilities['SwisNo']:
        x    = c2f[county][facility]
        temp += x['prop']
#    temp = sum([c2f[county][facility]['prop'] for facilities in facilities['SwisNo']]) #Does the same thing
    obj += landfill_ef*(1 - temp)*Fetch(counties, 'COUNTY', county, 'disposal.y')

# emissions due to transport of waste from county to facility
for county in counties['COUNTY']:
    for facility in facilities['SwisNo']:
        x    = c2f[county][facility]
        obj += x['prop']*x['trans_cost']*Fetch(counties, 'COUNTY', county, 'disposal.y')

# emissions due to waste remaining in facility #TODO - hyperbolic constraint!
for facility in facilities['SwisNo']:
    temp = 0
    for rangeland in rangelands['OBJECTID']:
        x = f2r[facility][rangeland]
        temp += x['prop']
    obj += compost_ef*(1 - temp)*conversion*intake[facility]    
    #TODO - change capacity units

for facility in facilities['SwisNo']:
    for rangeland in rangelands['OBJECTID']:
        x = f2r[facility][rangeland]
        applied_amount = x['prop']*conversion*intake[facility]
        # emissions due to transport of compost from facility to rangelands
        obj += x['trans_cost']* applied_amount
        # emissions due to application of compost by manure spreader
        obj += spreader_ef * applied_amount
        #TODO - change capacity units
        # emissions due to sequestration of applied compost
        obj += seq_f * applied_amount

#Constraints
cons = []

#supply constraint
for county in counties['COUNTY']:
    temp = 0
    for facility in facilities['SwisNo']:
        x    = c2f[county][facility]
        temp += x['prop']
        cons += [0<=x['prop']]              #Each proportion must be >=0
    cons += [temp <= 1]                     #Sum of propotions for each county must be <=1

#
for facility in facilities['SwisNo']:
    temp = 0
    for rangeland in rangelands['OBJECTID']:
        x = f2r[facility][rangeland]
        temp += x['prop']
        cons += [0<=x['prop']]              #Each proportion must be >=0
    cons += [temp <= 1]  

# capacity constraint
for facility in facilities['SwisNo']:
    cons += [intake[facility] <= Fetch(facilities, 'SwisNo', facility, 'cap_m3')]

# intake constraint
for facility in facilities['SwisNo']:
    temp = 0
    for county in counties['COUNTY']:
        x    = c2f[county][facility]
        temp += x['prop']*Fetch(counties, 'COUNTY', county, 'disposal.y')             #Each proportion must be >=0
    cons += [temp == intake[facility]]         

prob = cp.Problem(cp.Minimize(obj), cons)
val = prob.solve(gp=True)
print("Optimal object value = {0}".format(val))


print("{0:15} {1:15} {2:15}".format("County","Facility","Amount"))
for county in counties['COUNTY']:
    for facility in facilities['SwisNo']:
        print("{0:15} {1:15} {2:15}".format(county,facility,c2f[county][facility]['prop'].value))

# print("{0:15} {1:15}".format("Rangeland","Amount"))
# for facility in facilities['SwisNo']:
#     for rangeland in rangelands['OBJECTID']:
#         print("{0:15} {1:15} {2:15}".format(facility,rangeland,f2r[facility][rangeland]['prop'].value))


