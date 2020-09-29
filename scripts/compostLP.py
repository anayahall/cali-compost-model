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

# from biomass_preprocessing import MergeInventoryAndCounty
#from swis_preprocessing import LoadAndCleanSWIS #TODO

############################################################
# Change this to activate/decativate print statements throughout
DEBUG = True
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
			if c2f[muni][facility]['quantity'].value is not None:
				v = c2f[muni][facility]['quantity'].value  
			else:
				v = 0.0
			c2f_values[muni][facility] = (round(int(v)))


	f2r_values = {}

	for facility in f2r.keys():
		f2r_values[facility] = {}
		for rangeland in f2r[facility].keys():
			f2r_values[facility][rangeland] = {}
			if f2r[facility][rangeland]['quantity'].value is not None:
				x = f2r[facility][rangeland]['quantity'].value
			else:
				x = 0.0
			f2r_values[facility][rangeland] = (round(int(x)))

	return c2f_values, f2r_values


# ############################################################
# ### LOAD IN DATA ###
# ############################################################
# LOAD ACTUAL DATA (LARGE)
# from dataload import msw, rangelands, facilities, croplands
# ^^ this script loads the data used in the analysis
# requires original shapefiles too large to host on github
# comment out if just testing linear programming model

print(" - compostLP - ")
# LOAD TOY DATA 
from toydata import msw, rangelands, facilities
# ^ this script generates a set of toy data that will allow users to 
# run the optimization model below without the original shapefiles
# uncomment if jsut testing linear programming model


############################################################


############################################################
# OPTIMIZATION MODEL       #################################
############################################################

print(" - compostLP - about to define model") if (DEBUG == True) else ()

# Below is the core linear programming model, defined as a function
def SolveModel(scenario_name = None, 
	# set feedstock
	feedstock = 'food_and_green', 

	# data sources
	msw = msw, 
	landuse = rangelands,
	# priority = 1,  
	facilities = facilities,
	
	# Scenario settings
	disposal_min = 0.00001,   # percent of waste to include in run (cannot be ZERO - will break solver #TODO)
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

	:param disposal_min percent of waste to include in run (default is 1)
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

	# # # Priority settng for rangelands 
	# critical only
	# if priority == 2:
	# 	landuse = landuse[landuse['Priority'] == 2]
	# # critical and semi critical
	# elif priority == 1: 
		# landuse = landuse[landuse['Priority'] != 0]
	# # not included in planning goals
	# elif priority == 0: 
	# 	landuse = landuse[{(landuse['Priority'] == 0)}]

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
		for land in landuse['OBJECTID']:
			rloc = Fetch(landuse, 'OBJECTID', land, 'centroid')
			f2r[facility][land] = {}
			# define decision variable here
			f2r[facility][land]['quantity'] = cp.Variable()
			# and again grab distance for associated emis/cost
			dist = Distance(floc,rloc)
			f2r[facility][land]['trans_emis'] = dist*detour_factor*kilometres_to_emissions
			f2r[facility][land]['trans_cost'] = dist*detour_factor*f2r_trans_cost

	############################################################

	#BUILD OBJECTIVE FUNCTION: we want to minimize emissions (same as maximizing mitigation)
	obj = 0

	print("--building objective function") if (DEBUG == True) else ()

	print(" -- Objective: MINIMIZE PROJECT cost ") if (DEBUG == True) else ()
	# cost_dict = {}
	# transport costs - county to facility
	for muni in msw['muni_ID']:
		print(" >  c2f cost for muni: ", muni) if (DEBUG == True) else ()
		# cost_dict[muni] = {}
		ship_cost = 0
		# cost_dict[county]['COUNTY'] = county
		for facility in facilities['SwisNo']:
			# print("c2f distance cost for facility: ", facility)
			x    = c2f[muni][facility]
			obj += x['quantity']*x['trans_cost']
		# cost_dict[muni]['cost'] = int(round(ship_cost))

	for facility in facilities['SwisNo']:
		print(" >  f2r  cost for facility and land: ", facility) if (DEBUG == True) else ()
		for land in landuse['OBJECTID']:
			x = f2r[facility][land]
			# project_cost due to transport of compost from facility to landuse
			obj += x['quantity'] * x['trans_cost']
			# project_cost due to application of compost by manure spreader
			obj += x['quantity'] * spreader_cost

	print("OBJ (C2f + F2R) SIZE: ", sys.getsizeof(obj)) if (DEBUG == True) else ()

	############################################################


	# Set disposal cap for use in constraints
	msw['disposal_minimum'] = (disposal_min) * msw['disposal']

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
		cons += [temp <= Fetch(msw, 'muni_ID', muni, 'disposal')]   #Sum for each county must be <= county production
		cons += [temp >= Fetch(msw, 'muni_ID', muni, 'disposal_minimum')]   #Sum for each county must be <= county production

	facilities['facility_capacity'] = capacity_multiplier * facilities['cap_m3']

	# for scenarios in which we want to ignore existing infrastructure limits on capacity
	if ignore_capacity == False:
		# otherwise, use usual demand constraints
		for facility in facilities['SwisNo']:
			temp = 0
			for land in landuse['OBJECTID']:
				x = f2r[facility][land]
				temp += x['quantity']
				cons += [0 <= x['quantity']]              #Each quantity must be >=0
			cons += [temp <= Fetch(facilities, 'SwisNo', facility, 'facility_capacity')]  # sum of each facility must be less than capacity        

	# end-use  constraint capacity
	for land in landuse['OBJECTID']:
		print("land constraints: ", land) if (DEBUG == True) else ()
		temp = 0
		for facility in facilities['SwisNo']:
			x = f2r[facility][land]
			temp += x['quantity']
			#TODO - is this constraint necessary - or repetitive of above
			cons += [0 <= x['quantity']]				# value must be >=0
		# land capacity constraint (no more can be applied than 0.25 inches)
		cons += [temp <= Fetch(landuse, 'OBJECTID', land, 'capacity_m3')]


	# balance facility intake to facility output
	for facility in facilities['SwisNo']:
		print("balancing facility intake and outake for facility: ", facility) if (DEBUG == True) else ()
		temp_in = 0
		temp_out = 0
		for muni in msw['muni_ID']:
			print("muni: ", muni) if (DEBUG == True) else ()
			x = c2f[muni][facility]
			temp_in += x['quantity']	# sum of intake into facility from counties
		for land in landuse['OBJECTID']:
			print("land: ", land) if (DEBUG == True) else ()
			x = f2r[facility][land]
			temp_out += x['quantity']	# sum of output from facilty to land
		cons += [temp_out == waste_to_compost*temp_in]

	############################################################
	tzero = datetime.datetime.now()
	print("-solving...  time: ", tzero)
	print("*********************************************")

	# DEFINE PROBLEM --> to MINIMIZE OBJECTIVE FUNCTION 
	prob = cp.Problem(cp.Minimize(obj), cons)

	# SOLVE MODEL TO GET FINAL VALUE (which will be in terms of kg of CO2)
	val = prob.solve(gp=False, verbose = True)
	now = datetime.datetime.now()
	
	project_cost = val

	print("TIME ELAPSED SOLVING: ", str(now - tzero))
	print("*********************************************")


	############################################################
	# print("{0:15} {1:15}".format("Rangeland","Amount"))
	# for facility in facilities['SwisNo']:
	#     for land in landuse['OBJECTID']:
	#         print("{0:15} {1:15} {2:15}".format(facility,land,f2r[facility][land]['quantity'].value))
	############################################################

	# Rangeland area covered (ha) & applied amount by land
	land_app = {}
	for land in landuse['OBJECTID']:
		print("Calculating land area & amount applied for land: ", land) if (DEBUG == True) else ()
		r_string = str(land)
		applied_volume = 0
		area = 0
		temp_transport_emis = 0
		temp_transport_cost = 0
		land_app[r_string] = {}
		land_app[r_string]['OBJECTID'] = r_string
		# toggle this on to collect County info. not in the second landuse dataset
		# land_app[r_string]['COUNTY'] = Fetch(landuse, 'OBJECTID', land, 'COUNTY') #FLAG!
		for facility in facilities['SwisNo']:
			# print("from facility: ", facility)
			x = f2r[facility][land]
			if x['quantity'].value is not None:
				v = x['quantity'].value  
			else:
				v = 0.0 
			applied_volume += v 
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

####### OLD OBJECTIVE FUNCTION --- swap to calculate AFTER SOLVING: 
#use c2f['muni']['facility']['quantity'].value
# or f2r['facility']['land']['quantity'].value

	total_emis = 0

	# EMISIONS FROM C TO F (at at Facility)
	count = 0 # for keeping track of the municipality count
	# emissions due to waste remaining in muni
	for muni in msw['muni_ID']:
		count += 1
		print("muni ID: ", muni, " ## ", count,  "-- (AVOIDED) LANDFILL EMISSIONS") if (DEBUG == True) else ()
		county_disposal = Fetch(msw, 'muni_ID', muni, 'disposal')
		temp = 0
		for facility in facilities['SwisNo']:
			print("c2f - facility: ", facility) if (DEBUG == True) else ()
			#grab quantity and sum for each county
			x    = c2f[muni][facility]
			if x['quantity'].value is not None:
				v = x['quantity'].value  
			else:
				v = 0.0
			temp += v
			# emissions due to transport of waste from county to facility 
			total_emis += v * x['trans_emis']
			# emissions due to processing compost at facility
			total_emis += v * process_emis
	#    temp = sum([c2f[muni][facility]['quantity'] for facilities in facilities['SwisNo']]) #Does the same thing
		total_emis += landfill_ef*(-temp) #AVOIDED Landfill emissions
		# obj += landfill_ef*(county_disposal - temp) #PENALTY for the waste stranded in county

	print("OBJ SIZE (C2f): ", sys.getsizeof(obj)) if (DEBUG == True) else ()

	# EMISSIONS FROM F TO R (and at Rangeland)
	for facility in facilities['SwisNo']:
		print("SW facility: ", facility, "--to LAND") if (DEBUG == True) else ()
		for land in landuse['OBJECTID']:
			print('f2r - land #: ', land) if (DEBUG == True) else ()
			x = f2r[facility][land]
			if x['quantity'].value is not None:
				applied_amount = x['quantity'].value  
			else:
				applied_amount = 0.0 
			# emissions due to transport of compost from facility to landuse
			total_emis += x['trans_emis']* applied_amount
			# emissions due to application of compost by manure spreader
			total_emis += spreader_ef * applied_amount
			# sequestration of applied compost
			total_emis += seq_f * applied_amount



#########################################

	cost_millions = (val/(10**6))    
	print("TOTAL COST (Millions $) : ", cost_millions)
	print("TOTAL EMISSIONS (kg CO2e) : ", total_emis)


	# translate to MMT
	CO2mit = -total_emis/(10**9)

	print("*********************************************")
	print("CO2 Mitigated (MMt CO2eq) = {0}".format(CO2mit))

	# val is in terms of dollars, total emis is in kg
	result = val/total_emis
	# result is in $ per kg
	#convert to $ per ton for abatement cost!!
	abatement_cost = (-result*1000)
	print("*********************************************")
	print("$/tCO2e MITIGATED: ", abatement_cost)
	print("*********************************************")


	c2f_values, f2r_values = SaveModelVars(c2f, f2r)


	return c2f_values, f2r_values, land_app, cost_millions, CO2mit, abatement_cost

# r = pd.merge(landuse, rdf, on = "COUNTY")
# fac_df = pd.merge(facilities, fac_df, on = "SwisNo")


############################################################



