## compostoptimization.py

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

# from california_cropland_cleaning import cleancropdata
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
CROPLAND = False
############################################################

# set data directories (relative)
print(" - compostLP - packages loaded, setting directories") if (DEBUG == True) else ()
DATA_DIR = "data"
RESULTS_DIR = "results"


############################################################
# FUNCTIONS USED IN THIS SCRIPT

print(" - compostLP - defining functions used in script (haversine distance, etc)") if (DEBUG == True) else ()
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


def SaveModelVars(c2f, f2r):

	c2f_values = {}

	for muni in c2f.keys():
		# print("COUNTY: ", county)
		c2f_values[muni] = {}
		for facility in c2f[muni].keys():
			# print("FACILITY: ", facility)
			c2f_values[muni][facility] = {}
			x = c2f[muni][facility]['quantity'].value
			c2f_values[muni][facility] = (round(int(x)))


	f2r_values = {}

	for facility in f2r.keys():
		f2r_values[facility] = {}
		for rangeland in f2r[facility].keys():
			f2r_values[facility][rangeland] = {}
			x = f2r[facility][rangeland]['quantity'].value
			f2r_values[facility][rangeland] = (round(int(x)))

	return c2f_values, f2r_values


#######################################################################################
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
	print("RUNNING ON COUNTIES")
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
# RANGELANDS 
############################################################
# Import rangelands
print("about to import rangelands") if (DEBUG == True) else ()
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
# if CROPLAND == True:
# 	croplands = cleancropdata(opj(DATA_DIR, 
# 		  "raw/Crop__Mapping_2014-shp/Crop__Mapping_2014.shp"))


############################################################
# OPTIMIZATION MODEL       #################################
############################################################

print("about to define model") if (DEBUG == True) else ()

# Below is the core linear programming model, defined as a function
def SolveModel(scenario_name = None, 
	# set feedstock
	feedstock = 'food_and_green', 

	# data sources
	msw = msw, 
	landuse = rangelands, 
	facilities = facilities,
	
	# Scenario settings
	disposal_rate = 1,   # percent of waste to include in run
	fw_reduction = 0,    # food waste reduced/recovered pre-disposal #FLAG is this accounted for ELSEWHERE?
	ignore_capacity = False, # toggle to ignore facility capacity info
	capacity_multiplier = 1, # can inflate capacity 
	
	# KEY MODEL PARAMETERS
	landfill_ef = 315, #kg CO2e / m3 = avoided emissions from waste going to landfill
	kilometres_to_emissions = 0.37, # kg CO2e/ m3 - km for 35mph speed 
	# kilometres_to_emissions_10 = 1, # FLAG!
	spreader_ef = 1.854, # kg CO2e / m3 = emissions from spreading compost
	seq_f = -108, # kg CO2e / m3 = sequestration rate
	
	# soil_emis = 68, # ignore now, included in seq?
	process_emis = 11, # kg CO2e/ m3 = emisisons at facility from processing compost
	waste_to_compost = 0.58, #% volume change from waste to compost
	c2f_trans_cost = 0.412, #$/m3-km # transit costs (alt is 1.8)
	f2r_trans_cost = .206, #$/m3-km # transit costs
	spreader_cost = 5.8, #$/m3 # cost to spread
	detour_factor = 1.4, #chosen based on literature - multiplier on haversine distance
		):
	
	"""
	Solves linear programming model for msw to convert to compost and apply to rangelands
	
	:param scenario_name OPTIONAL name given to scenario and amended to model output
	:param feedstock type of feedstock to use in run (default is 'food_and_green', options are also 'food' or 'green')
	
	:param msw MSW data source
	:param landuse landarea data source
	:param facilities SWIS data source

	:param disposal_rate percent of waste to include in run (default is 1)
	:param fw_reduction food waste reduced/recovered pre-disposal (default is 0) 
	:param ignore_capacity toggle to ignore facility capacity info (default is FALSE)
	:param capacity_multiplier scalar multiplier by which to inflate capacity (default is 1)

	:param landfill_ef Landfill Emission Factor (kg CO2e / m3)
	:param kilometers_to_emisisons Roadway travel emissions factor for heavy duty trucks
	:param spreader_ef Manure spreader EF
	:param seq_f Sequestration factor 

	:param process_emis Processing emissions
	:param waste_to_compost volume change from waste to compost ( = 0.58)
	:param c2f_trans_cost Collection transit cost (default = 0.412, alt is 1.8)
	:param f2r_trans_cost Hauling transit cost (default is .206, $/m3-km 
	:param spreader_cost Cost to spread (= 5.8, $/m3 )
	:param detour_factor Detour Factor multiplier on haversine distance (default is 1.4)

	:returns c2f and f2r quantities, area/amount applied by rangeland, total cost, total mitigation, and abatement cost
	"""

	# #Variables
	print("--setting constant parameters") if (DEBUG == True) else ()

	print("-- setting feedstock and disposal") if (DEBUG == True) else ()
	# change supply constraint by feedstock selected
	if feedstock == 'food_and_green':
		# combine food and green waste (wet tons) and convert to cubic meters
		# first, adjust food waste tonnage by fw reduction factor
		msw.loc[(msw['subtype']=='MSW_food'),'wt'] = msw[msw['subtype'] == 'MSW_food']['wt']*(1-fw_reduction)
		# then combine (sum) and convert to cubic meters	   
		msw['disposal'] = msw.groupby(['muni_ID'])['wt'].transform('sum') / (1.30795*(1/2.24))
		msw = msw.drop_duplicates(subset = 'muni_ID')
		msw['subtype'].replace({'MSW_green':'food_and_green'}, inplace = True)

	elif feedstock == 'food':
		# subset just food waste and convert wet tons to cubic meters
		msw = msw[(msw['subtype'] == "MSW_food")]
		# msw['disposal'] = (1-fw_reduction)* counties['disposal_wm3']
		msw['disposal'] = (1-fw_reduction)* msw['wt'] / (1.30795*(1/2.24))

	elif feedstock == 'green':
		# make green!!
		msw = msw[(msw['subtype'] == "MSW_green")]
		msw['disposal'] = msw['wt'] / (1.30795*(1/2.24))

############################################################

	# decision variables
	print("--defining decision vars") if (DEBUG == True) else ()
	# amount of county waste to send to a facility 
	c2f = {}
	for muni in msw['muni_ID']:
		c2f[muni] = {}
		cloc = Fetch(msw, 'muni_ID', muni, 'geometry')
		for facility in facilities['SwisNo']:
			floc = Fetch(facilities, 'SwisNo', facility, 'geometry')
			c2f[muni][facility] = {}
			# this is what actually defines the decision variable
			c2f[muni][facility]['quantity'] = cp.Variable()
			# since already grabbing this relationship, might as well store distance and associated emis/cost
			dist = Distance(cloc,floc)
			c2f[muni][facility]['trans_emis'] = dist*detour_factor*kilometres_to_emissions
			c2f[muni][facility]['trans_cost'] = dist*detour_factor*c2f_trans_cost

	# amount of compost to send to rangeland 
	f2r = {}
	for facility in facilities['SwisNo']:
		f2r[facility] = {}
		floc = Fetch(facilities, 'SwisNo', facility, 'geometry')
		for rangeland in rangelands['OBJECTID']:
			rloc = Fetch(rangelands, 'OBJECTID', rangeland, 'centroid')
			f2r[facility][rangeland] = {}
			# define decision variable here
			f2r[facility][rangeland]['quantity'] = cp.Variable()
			# and again grab distance for associated emis/cost
			dist = Distance(floc,rloc)
			f2r[facility][rangeland]['trans_emis'] = dist*detour_factor*kilometres_to_emissions
			f2r[facility][rangeland]['trans_cost'] = dist*detour_factor*f2r_trans_cost

	############################################################

	#BUILD OBJECTIVE FUNCTION: we want to minimize emissions (same as maximizing mitigation)
	obj = 0

	print("--building objective function") if (DEBUG == True) else ()

	# EMISIONS FROM C TO F (at at Facility)
	count = 0
	# emissions due to waste remaining in muni
	for muni in msw['muni_ID']:
		count += 1
		print("muni ID: ", muni, " ## ", count,  "--AVOIDED LANDFILL EMISSIONS") if (DEBUG == True) else ()

		# total_waste = Fetch(counties, 'COUNTY', county, 'disposal_cap')
		temp = 0
		for facility in facilities['SwisNo']:
			print("c2f - facility: ", facility) if (DEBUG == True) else ()
			x    = c2f[muni][facility]
			temp += x['quantity']
			# emissions due to transport of waste from county to facility 
			obj += x['quantity']*x['trans_emis']
			# emissions due to processing compost at facility
			obj += x['quantity']*process_emis
	#    temp = sum([c2f[muni][facility]['quantity'] for facilities in facilities['SwisNo']]) #Does the same thing
		obj += landfill_ef*(0 - temp) #FLAG
		# obj += landfill_ef*(total_waste - temp)

	print("OBJ SIZE (C2f): ", sys.getsizeof(obj)) if (DEBUG == True) else ()

	# EMISSIONS FROM F TO R (and at Rangeland)
	for facility in facilities['SwisNo']:
		print("SW facility: ", facility, "--to RANGELAND") if (DEBUG == True) else ()
		for rangeland in rangelands['OBJECTID']:
			print('f2r - rangeland #: ', rangeland) if (DEBUG == True) else ()
			x = f2r[facility][rangeland]
			applied_amount = x['quantity']
			# emissions due to transport of compost from facility to rangelands
			obj += x['trans_emis']* applied_amount
			# emissions due to application of compost by manure spreader
			obj += spreader_ef * applied_amount
			# emissions due to sequestration of applied compost
			obj += seq_f * applied_amount

	print("OBJ (C2f + F2R) SIZE: ", sys.getsizeof(obj)) if (DEBUG == True) else ()

	############################################################


	# Set disposal cap for use in constraints
	msw['disposal_cap'] = (disposal_rate) * msw['disposal']

	#Constraints
	cons = []
	print("--subject to constraints") if (DEBUG == True) else ()
	now = datetime.datetime.now()
	print("Time starting constraints: ", str(now)) if (DEBUG == True) else ()

	#supply constraint
	for muni in msw['muni_ID']:
		temp = 0
		for facility in facilities['SwisNo']:
			print("supply constraints -- muni: ",muni, " to facility: ", facility) if (DEBUG == True) else ()
			x    = c2f[muni][facility]
			temp += x['quantity']
			cons += [0 <= x['quantity']]              #Quantity must be >=0
		cons += [temp <= Fetch(msw, 'muni_ID', muni, 'disposal_cap')]   #Sum for each county must be <= county production

	facilities['facility_capacity'] = capacity_multiplier * facilities['cap_m3']

	# for scenarios in which we want to ignore existing infrastructure limits on capacity
	if ignore_capacity == False:
		# otherwise, use usual demand constraints
		for facility in facilities['SwisNo']:
			temp = 0
			for rangeland in rangelands['OBJECTID']:
				x = f2r[facility][rangeland]
				temp += x['quantity']
				cons += [0 <= x['quantity']]              #Each quantity must be >=0
			cons += [temp <= Fetch(facilities, 'SwisNo', facility, 'facility_capacity')]  # sum of each facility must be less than capacity        

	# end-use  constraint capacity
	for rangeland in rangelands['OBJECTID']:
		print("rangeland constraints: ", rangeland) if (DEBUG == True) else ()
		temp = 0
		for facility in facilities['SwisNo']:
			x = f2r[facility][rangeland]
			temp += x['quantity']
			#TODO - is this constraint necessary - or repetitive of above
			cons += [0 <= x['quantity']]				# value must be >=0
		# rangeland capacity constraint (no more can be applied than 0.25 inches)
		cons += [temp <= Fetch(rangelands, 'OBJECTID', rangeland, 'capacity_m3')]


	# balance facility intake to facility output
	for facility in facilities['SwisNo']:
		print("balancing facility intake and outake for facility: ", facility) if (DEBUG == True) else ()
		temp_in = 0
		temp_out = 0
		for muni in msw['muni_ID']:
			print("muni: ", muni) if (DEBUG == True) else ()
			x = c2f[muni][facility]
			temp_in += x['quantity']	# sum of intake into facility from counties
		for rangeland in rangelands['OBJECTID']:
			print("rangeland: ", rangeland) if (DEBUG == True) else ()
			x = f2r[facility][rangeland]
			temp_out += x['quantity']	# sum of output from facilty to rangeland
		cons += [temp_out == waste_to_compost*temp_in]

	############################################################
	tzero = datetime.datetime.now()
	print("-solving...  time: ", tzero)
	print("*********************************************")

	# DEFINE PROBLEM --> to MINIMIZE OBJECTIVE FUNCTION 
	prob = cp.Problem(cp.Minimize(obj), cons)

	# SOLVE MODEL TO GET FINAL VALUE (which will be in terms of kg of CO2)
	val = prob.solve(gp=False)
	now = datetime.datetime.now()


	# translate to MMT
	CO2mit = -val/(10**9)
	
	print("TIME ELAPSED SOLVING: ", str(now - tzero))
	print("*********************************************")

	print("Optimal object value (Mt CO2eq) = {0}".format(CO2mit))
	print("*********************************************")

	############################################################
	# print("{0:15} {1:15}".format("Rangeland","Amount"))
	# for facility in facilities['SwisNo']:
	#     for rangeland in rangelands['OBJECTID']:
	#         print("{0:15} {1:15} {2:15}".format(facility,rangeland,f2r[facility][rangeland]['quantity'].value))
	############################################################

	# Rangeland area covered (ha) & applied amount by rangeland
	land_app = {}
	for rangeland in rangelands['OBJECTID']:
		print("Calculating rangeland area & amount applied for rangeland: ", rangeland) if (DEBUG == True) else ()
		r_string = str(rangeland)
		applied_volume = 0
		area = 0
		temp_transport_emis = 0
		temp_transport_cost = 0
		land_app[r_string] = {}
		land_app[r_string]['OBJECTID'] = r_string
		land_app[r_string]['COUNTY'] = Fetch(rangelands, 'OBJECTID', rangeland, 'COUNTY')
		for facility in facilities['SwisNo']:
			# print("from facility: ", facility)
			x = f2r[facility][rangeland]
			applied_volume += x['quantity'].value
			temp_transport_emis += applied_volume* x['trans_emis']
			temp_transport_cost += applied_volume *x['trans_cost']
			area += int(round(applied_volume * (1/63.5)))
		land_app[r_string]['area_treated'] = area
		land_app[r_string]['volume'] = int(round(applied_volume))
		land_app[r_string]['application_cost'] = int(round(applied_volume))*spreader_cost
		land_app[r_string]['application_emis'] = int(round(applied_volume))*spreader_ef
		land_app[r_string]['trans_emis'] = temp_transport_emis
		land_app[r_string]['trans_cost'] = temp_transport_cost
		land_app[r_string]['sequestration'] = applied_volume*seq_f



	# # # Quantity moved out of county
	# county_results = {}
	# # print("{0:15} {1:15} {2:15}".format("COUNTY","Facility","Amount"))
	# for muni in msw['muni_ID']:
	#     output = 0
	#     # temp_volume = 0
	#     temp_transport_emis = 0
	#     temp_transport_cost = 0
	#     county_results[county] = {}
	#     for facility in facilities['SwisNo']:
	#         x = c2f[county][facility]
	#         output += x['quantity'].value
	#         # temp_volume += x['quantity'].value
	#         temp_transport_emis += output * x['trans_emis']
	#         temp_transport_cost += output * x['trans_cost']
	#         # print("{0:15} {1:15} {2:15}".format(county,facility,output))
	#     county_results[county]['output'] = int(round(output))
	#     county_results[county]['ship_emis'] = int(round(temp_transport_emis))
	#     county_results[county]['TOTAL_emis'] = temp_transport_emis
	#     county_results[county]['ship_cost'] = int(round(temp_transport_cost))
	#     county_results[county]['TOTAL_cost'] = temp_transport_cost

	# # # Facility intake 
	# fac_intake = {}
	# for facility in facilities['SwisNo']:
	#     temp_volume = 0
	#     fac_intake[facility] = {}
	#     fac_intake[facility]['SwisNo'] = facility
	#     fac_intake[facility]['COUNTY'] = Fetch(facilities, 'SwisNo', facility, 'COUNTY')
	#     for county in counties['COUNTY']:
	#         x = c2f[county][facility]
	#         # t = c2f[county][facility]['quantity'].value
	#         temp_volume += x['quantity'].value
	#     fac_intake[facility]['intake'] = int(round(temp_volume))
	#     fac_intake[facility]['facility_emis'] = temp_volume*process_emis

	# ####################################
	# # print(county_results)
	# # county_results = {}
	# for k,v in land_app.items():
	#     county = v['COUNTY']
	#     # print('county', county)
	#     if county in county_results:

	#         if 'TOTAL_emis' in county_results[county].keys():
	#             county_results[v['COUNTY']]['TOTAL_emis'] = county_results[v['COUNTY']]['TOTAL_emis']
	#         else: 
	#             county_results[v['COUNTY']]['TOTAL_emis'] = 0

	#         # SUM VOLUME OF RANGELAND IN COUNTY
	#         if 'volume_applied' in county_results[county].keys():
	#             county_results[v['COUNTY']]['volume_applied'] = county_results[v['COUNTY']]['volume_applied'] + v['volume']
	#         else:
	#             county_results[county]['volume_applied'] = v['volume']
			
	#         # Sum of cost of applying compost in the county
	#         if 'application_cost' in county_results[county].keys():
	#             county_results[v['COUNTY']]['application_cost'] = county_results[v['COUNTY']]['application_cost'] + v['application_cost']
	#         else:
	#             county_results[county]['application_cost'] = v['application_cost']
			
	#         # sum of emissions from applying compost in county
	#         if 'application_emis' in county_results[county].keys():
	#             county_results[v['COUNTY']]['application_emis'] = county_results[v['COUNTY']]['application_emis'] + v['application_emis']
	#             county_results[v['COUNTY']]['TOTAL_emis'] = county_results[v['COUNTY']]['TOTAL_emis'] + v['application_emis']

	#         else:
	#             county_results[county]['application_emis'] = v['application_emis']
	#             county_results[v['COUNTY']]['TOTAL_emis'] =  v['application_emis']

			
	#         # sum of transportation emissions for hauling compost to county's rangelands
	#         if 'trans_emis' in county_results[county].keys():
	#             county_results[v['COUNTY']]['trans_emis'] = county_results[v['COUNTY']]['trans_emis'] + v['trans_emis']
	#             county_results[v['COUNTY']]['TOTAL_emis'] = county_results[v['COUNTY']]['TOTAL_emis'] + v['trans_emis']

	#         else:
	#             county_results[county]['trans_emis'] = v['trans_emis']
	#             county_results[v['COUNTY']]['TOTAL_emis'] = v['trans_emis']

			
	#         # sum of transportation costs for hauling compost to county's rangelands
	#         if 'trans_cost' in county_results[county].keys():
	#             county_results[v['COUNTY']]['trans_cost'] = county_results[v['COUNTY']]['trans_cost'] + v['trans_cost']
	#         else:
	#             county_results[county]['trans_cost'] = v['trans_cost']
			
	#         # total sequestration potential from applying compost in county
	#         if 'sequestration' in county_results[county].keys():
	#             county_results[v['COUNTY']]['sequestration'] = county_results[v['COUNTY']]['sequestration'] + v['sequestration']
	#             county_results[v['COUNTY']]['TOTAL_emis'] = county_results[v['COUNTY']]['TOTAL_emis'] - v['sequestration']

	#         else:
	#             county_results[county]['sequestration'] = v['sequestration']
	#             county_results[v['COUNTY']]['TOTAL_emis'] = v['sequestration']

	#     else:
	#         county_results[county] = {}
	#         county_results[county]['volume_applied'] = v['volume']
	#         county_results[county]['trans_cost'] = v['trans_cost']
	#         county_results[county]['trans_emis'] = v['trans_emis']
	#         county_results[county]['application_emis'] = v['application_cost']
	#         county_results[county]['application_emis'] = v['application_emis']
	#         county_results[county]['sequestration'] = v['sequestration']
	#         county_results[county]['TOTAL_emis'] = v['application_emis']+ v['trans_emis']

	# for k,v in fac_intake.items():
	#     # print('k: ', k)
	#     # print('v: ', v)
	#     # print('V.COUNTY: ', v['COUNTY'])
	#     county = v['COUNTY']
	#     # print('county', county)
	#     if county in county_results:
	#         if 'county_fac_intake' in county_results[county].keys(): 
	#             county_results[v['COUNTY']]['county_fac_intake'] = county_results[v['COUNTY']]['county_fac_intake'] + v['intake']
	#             # print('got in here...')
	#         else:
	#             county_results[county]['county_fac_intake'] = v['intake']
	#         if 'county_fac_emis' in county_results[county].keys(): 
	#             county_results[v['COUNTY']]['county_fac_emis'] = county_results[v['COUNTY']]['county_fac_emis'] + v['facility_emis']
	#             # print('got in here...')
	#         else:
	#             county_results[county]['county_fac_emis'] = v['facility_emis']
	#     else:
	#         county_results[county] = {}
	#         county_results[county]['county_fac_intake'] = v['intake']
	#         county_results[county]['county_fac_emis'] = v['facility_emis']
	#     # print(county_results)

#########################################

	#Calculate cost after solving!
	project_cost = 0

	print("Calculating PROJECT cost ") if (DEBUG == True) else ()
	cost_dict = {}
	# transport costs - county to facility
	for muni in msw['muni_ID']:
		print(" > Calculating c2f cost for muni: ", muni) if (DEBUG == True) else ()
		cost_dict[muni] = {}
		ship_cost = 0
		# cost_dict[county]['COUNTY'] = county
		for facility in facilities['SwisNo']:
			# print("c2f distance cost for facility: ", facility)
			x    = c2f[muni][facility]
			project_cost += x['quantity'].value*x['trans_cost']
			ship_cost += x['quantity'].value*x['trans_cost']
		cost_dict[muni]['cost'] = int(round(ship_cost))

	for facility in facilities['SwisNo']:
		print(" > calculating f2r distance cost for facility: ", facility) if (DEBUG == True) else ()
		for rangeland in rangelands['OBJECTID']:
			# print("f2r cost for rangeland: ", rangeland)
			x = f2r[facility][rangeland]
			applied_amount = x['quantity'].value
			# project_cost due to transport of compost from facility to rangelands
			project_cost += x['trans_cost']* applied_amount
			# project_cost due to application of compost by manure spreader
			project_cost += spreader_cost * applied_amount


	cost_millions = (project_cost/(10**6))    
	print("TOTAL COST (Millions $) : ", cost_millions)
	# val is in terms of kg CO2e 
	result = project_cost/val
	abatement_cost = (-result*1000)
	print("*********************************************")
	print("$/tCO2e MITIGATED: ", abatement_cost)
	print("*********************************************")


	c2f_values, f2r_values = SaveModelVars(c2f, f2r)


	return c2f_values, f2r_values, land_app, cost_millions, CO2mit, abatement_cost

# r = pd.merge(rangelands, rdf, on = "COUNTY")
# fac_df = pd.merge(facilities, fac_df, on = "SwisNo")


############################################################



