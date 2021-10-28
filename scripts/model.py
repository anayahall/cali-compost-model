## compostoptimization.py

import cvxpy as cp
import numpy as np
import os
import datetime
from os.path import join as opj
import json
import sys
import gurobipy as gp
import pickle
import matplotlib.pyplot as plt

import pandas as pd
import shapely as shp
import geopandas as gpd
import scipy as sp


############################################################
# Change this to activate/decativate detailed print statements throughout
DEBUG = False
############################################################

# set data directories (relative)
print(" - model- packages loaded, setting directories") if (DEBUG == True) else ()


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
		c2f_values[muni] = {}
		for facility in c2f[muni].keys():
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

def LandApplication(
	landuse, 
	facilities,
	f2r): 
	"""
	takes solved decision vars and returns total area applied as well as total by county
	:param landuse land use in model
	:param facilities swis facilities
	:param f2r amount of compost moved from each facility to each rangeland
	"""

	total_area = 0 
	# create empty dictionary
	land_app = {}

	for land in landuse['OBJECTID']:
		print("Calculating land area & amount applied for land: ", land) if (DEBUG == True) else ()
		r_string = str(land)
		applied_volume = 0
		local_area = 0
		land_app[r_string] = {}
		land_app[r_string]['OBJECTID'] = r_string

		# print("SEQ F: ", seq_f)
		
		# toggle this on to collect County info. not in the second landuse dataset
		# land_app[r_string]['COUNTY'] = Fetch(landuse, 'OBJECTID', land, 'COUNTY') #FLAG!
		for facility in facilities['SwisNo']:
			land_app[r_string][facility] = {}
			# print("from facility: ", facility)
			x = f2r[facility][land]
			if x['quantity'].value is not None:
				v = x['quantity'].value  
			else:
				v = 0.0 
			applied_volume += v 
			local_area += int(round(v * (1/63.5)))
			land_app[r_string]['facility'] = facility
			total_area += local_area
		land_app[r_string]['area_treated'] = local_area
		land_app[r_string]['volume'] = int(round(applied_volume))

	return land_app, total_area


def TotalEmissions(
	msw, landuse, facilities, seq_factors, c2f, f2r
	):
	"""
	hoping to use this to calc emissions after running the full pareto model
	"""
	# emission factors 
	landfill_ef = 315.0, #kg CO2e / m3 = avoided emissions from waste going to landfill
	kilometres_to_emissions = 0.37, # kg CO2e/ m3 - km for 35mph speed 
	spreader_ef = 1.854, # kg CO2e / m3 = emissions from spreading compost
	process_emis = 11, # kg CO2e/ m3 = emisisons at facility from processing compost

	total_emis = 0
	# print("starting emissions calc")
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
		 # temp = sum([c2f[muni][facility]['quantity'] for facilities in facilities['SwisNo']]) #Does the same thing
		total_emis += 315.0 * (-temp) #AVOIDED Landfill emissions
		# obj += landfill_ef*(county_disposal - temp) #PENALTY for the waste stranded in county

	# EMISSIONS FROM F TO R (and at Rangeland)
	for facility in facilities['SwisNo']:
		print("SW facility: ", facility, "--to LAND") if (DEBUG == True) else ()
		for land in landuse['OBJECTID']:
			print('f2r - land #: ', land) if (DEBUG == True) else ()

			# pull county specific sequestration rate!!
			county = Fetch(landuse, 'OBJECTID' , land, 'COUNTY')
			# print("COUNTYYYYYYYYYYYYYY: ", county)
			seq_f = Fetch(seq_factors, 'County', county, 'seq_f')
			# print("SEQ F: ", seq_f)

			x = f2r[facility][land]
			if x['quantity'].value is not None:
				applied_amount = x['quantity'].value   
			else:
				applied_amount = 0.0 

			# emissions due to transport of compost from facility to landuse
			total_emis += x['trans_emis']* applied_amount
			# emissions due to application of compost by manure spreader
			
			total_emis += 1.854 * applied_amount
			# sequestration of applied compost
			
			total_emis += seq_f * applied_amount
	# print("****** DONE W EMISSIONS LOOP****" )
						
			# translate to MMT
	CO2mit = -total_emis/(10**9)

	return CO2mit, total_emis

def ProjectCost(
	msw, landuse,facilities, c2f, f2r):
	"""
	to sum project cost after solving
	"""
	# cost parameters
	# print("start calculating project cost")	
	c2f_trans_cost = 0.412, #$/m3-km # transit costs (alt is 1.8)
	f2r_trans_cost = .206, #$/m3-km # transit costs
	spreader_cost = 5.8, #$/m3 # cost to spread

	# 	PROJECT COST!
	project_cost = 0
	# transport costs - county to facility
	for muni in msw['muni_ID']:
		print(" >  c2f cost for muni: ", muni) if (DEBUG == True) else ()
		# cost_dict[muni] = {}
		ship_cost = 0
		# cost_dict[county]['COUNTY'] = county
		for facility in facilities['SwisNo']:
			# print("c2f distance cost for facility: ", facility)
			x    = c2f[muni][facility]
			if x['quantity'].value is not None:
				v = x['quantity'].value  
			else:
				v = 0.0
			project_cost += v * x['trans_cost']

		# cost_dict[muni]['cost'] = int(round(ship_cost))

	for facility in facilities['SwisNo']:
		print(" >  f2r  cost for facility and land: ", facility) if (DEBUG == True) else ()
		for land in landuse['OBJECTID']:
			x = f2r[facility][land]
			if x['quantity'].value is not None:
				applied_amount = x['quantity'].value   
			else:
				applied_amount = 0.0
			# project_cost due to transport of compost from facility to landuse
			project_cost += applied_amount * x['trans_cost'] 

			# project_cost due to application of compost by manure spreader
			project_cost += applied_amount * 5.8 

	# print("done calculating project cost")		
	return project_cost

	

# ############################################################
# ### LOAD IN DATA ###
# ############################################################
# LOAD ACTUAL DATA (LARGE)
from dataload import msw, rangelands, facilities, grazed_rates
# ^^ this script loads the data used in the analysis
# requires original shapefiles too large to host on github
# comment out if just testing linear programming model

# OR LOAD IN TOY DATA
# from toydata import msw, rangelands, facilities, seq_factors


############################################################
############################################################

def RunModel(
		# data sources
	msw = msw, 
	landuse = rangelands,
	facilities = facilities,
	seq_factors = grazed_rates, #alt is seq_f = -108
	feedstock = 'food_and_green',

	# scaling parameters 
	detour_factor = 1.4,
	capacity_multiplier = 1,
	fw_reduction = 0,
		
	# emission factors 
	landfill_ef = 315, #kg CO2e / m3 = avoided emissions from waste going to landfill
	kilometres_to_emissions = 0.37, # kg CO2e/ m3 - km for 35mph speed 
	spreader_ef = 1.854, # kg CO2e / m3 = emissions from spreading compost
	process_emis = 11, # kg CO2e/ m3 = emisisons at facility from processing compost
	waste_to_compost = 0.58, #% volume change from waste to compost
	# cost parameters
	c2f_trans_cost = 0.412, #$/m3-km # transit costs (alt is 1.8)
	f2r_trans_cost = .206, #$/m3-km # transit costs
	spreader_cost = 5.8, #$/m3 # cost to spread	
	
	# emission factors 
	# landfill_ef = cp.Parameter(pos=True, value=315), #315, kg CO2e / m3 = avoided emissions from waste going to landfill
	# kilometres_to_emissions = cp.Parameter(pos=True, value=0.37), #0.37, kg CO2e/ m3 - km for 35mph speed 
	# process_emis = cp.Parameter(pos=True, value=11), #11, #kg CO2e/ m3 = emisisons at facility from processing compost
	# waste_to_compost = cp.Parameter(pos=True, value=0.58), #0.58, % volume change from waste to compost
	# spreader_ef = cp.Parameter(pos=True, value=1.854), # kg CO2e / m3 = emissions from spreading compost


	# # cost parameters
	# c2f_trans_cost = cp.Parameter(pos=True, value=0.412), #0.412, $/m3-km # transit costs (alt is 1.8)
	# f2r_trans_cost = cp.Parameter(pos=True, value=0.206), #.206, $/m3-km # transit costs
	# spreader_cost = cp.Parameter(pos=True, value=5.8), #5.8, $/m3 # cost to spread
		
	# pareto tradeoff
	a = cp.Parameter(nonneg=True)): # minimizing on cost when a is 1, and on ghg when a is 0
		
	m=5

	print("-- setting feedstock and disposal")
	# change supply constraint by feedstock selected
	if feedstock == 'food_and_green':
		# combine food and green waste (wet tons) and convert to cubic meters
		msw_temp = msw.copy()
		msw_temp['disposal'] = msw_temp['fg_wt']/(1.30795/2.24)
		msw = msw_temp.copy()


	# elif feedstock == 'food':
	# 	msw_temp = msw.copy()
	# 	# subset just food waste and convert wet tons to cubic meters
	# 	msw_temp = msw_temp[(msw_temp['subtype'] == "MSW_food")]
	# 	# msw['disposal'] = (1-fw_reduction)* counties['disposal_wm3']
	# 	msw_temp['disposal'] = (1-fw_reduction)* msw_temp['wt'] / (1.30795*(1/2.24))
	# 	msw = msw_temp.copy()

	# elif feedstock == 'green':
	# 	msw_temp = msw.copy()
	# 	msw_temp = msw_temp[(msw_temp['subtype'] == "MSW_green")]
	# 	msw_temp['disposal'] = msw_temp['wt'] / (1.30795*(1/2.24))
	# 	msw = msw_temp.copy()



		
		############################################################
	# decision variables
	print("--defining decision vars") #if (DEBUG == True) else ()
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
			f2r[facility][land]['quantity'] = cp.Variable(nonneg=True)
			# and again grab distance for associated emis/cost
			dist = Distance(floc,rloc)
			f2r[facility][land]['trans_emis'] = dist*detour_factor*kilometres_to_emissions
			f2r[facility][land]['trans_cost'] = dist*detour_factor*f2r_trans_cost
		
		############################################################       
		#BUILD OBJECTIVE FUNCTION
	obj = 0

		# COSTS: collection cost, hauling cost, spreading cost
	print(" -- Objective: min [a*cost + (1-a)*emis] --") #if (DEBUG == True) else ()
	# transport costs - county to facility
	for muni in msw['muni_ID']:
		print(" >  c2f cost for muni: ", muni) if (DEBUG == True) else ()
		for facility in facilities['SwisNo']:
			# print("c2f distance cost for facility: ", facility)
			x    = c2f[muni][facility]
			obj += a * x['quantity']*x['trans_cost'] # new pareto analysis

	for facility in facilities['SwisNo']:
		print(" >  f2r  cost for facility and land: ", facility) if (DEBUG == True) else ()
		for land in landuse['OBJECTID']:
			x = f2r[facility][land]
						
			# project_cost due to transport of compost from facility to landuse
			obj += a * x['quantity'] * x['trans_cost'] # new pareto analysis

			# project_cost due to application of compost by manure spreader
			obj += a * x['quantity'] * spreader_cost # new pareto analysis

	# EMISIONS FROM C TO F (at at Facility)
	# Emissions: collection, processing, avoided landfill	
	count = 0 # for keeping track of the municipality count
	# emissions due to waste remaining in muni
	for muni in msw['muni_ID']:
		count += 1
		print("muni ID: ", muni, " ## ", count,  "-- EMISSIONS") if (DEBUG == True) else ()
		# county_disposal = Fetch(msw, 'muni_ID', muni, 'disposal')
		temp = 0
		for facility in facilities['SwisNo']:
			print("c2f - facility: ", facility) if (DEBUG == True) else ()
			#grab quantity and sum for each county
			x    = c2f[muni][facility]
			if x['quantity'] is not None:
				v = x['quantity']  
			else:
				v = 0.0
			temp += v
						
			# emissions due to transport of waste from county to facility 
			obj += (1-a) * v * x['trans_emis'] # pareto analysis

			# emissions due to processing compost at facility
			obj += (1-a) * v * process_emis # pareto analysis

	#    temp = sum([c2f[muni][facility]['quantity'] for facilities in facilities['SwisNo']]) #Does the same thing
		obj += (1-a) * landfill_ef*(-temp) #AVOIDED Landfill emissions ## pareto analysis

		# obj += landfill_ef*(county_disposal - temp) #PENALTY for the waste stranded in county

	# EMISSIONS FROM F TO R (and at Rangeland)
		# Emissions: hauling, spreading, sequestration
	for facility in facilities['SwisNo']:
		print("SW facility: ", facility, "--to LAND") if (DEBUG == True) else ()
		for land in landuse['OBJECTID']:
			print('f2r - land #: ', land) if (DEBUG == True) else ()
			

			# pull county specific sequestration rate!!
			county = Fetch(landuse, 'OBJECTID' , land, 'COUNTY')
			# print("COUNTYY: ", county)
			seq_f = Fetch(seq_factors, 'County', county, 'seq_f')
			# print("SEQ F: ", seq_f)

			x = f2r[facility][land]
			if x['quantity'] is not None:
				applied_amount = x['quantity']  
			else:
				applied_amount = 0.0 
								
			# emissions due to transport of compost from facility to landuse
			obj += (1-a) * x['trans_emis']* applied_amount # pareto analysis

			# emissions due to application of compost by manure spreader
			obj += (1-a) * spreader_ef * applied_amount # pareto analysis

			# sequestration of applied compost
			obj += (1-a) * (-seq_f) * applied_amount # pareto analysis
				
	############################################################
	#Constraints
	# supply constraint, processing capacity, land-use, throughput!
	# print("--subject to constraints") if (DEBUG == True) else ()

	cons = []
		
	#supply constraint (quantity can't exceed msw supply)
	for muni in msw['muni_ID']:
		temp = 0
		for facility in facilities['SwisNo']:
			print("supply constraints -- muni: ",muni, " to facility: ", facility) if (DEBUG == True) else ()
			x    = c2f[muni][facility]
			temp += x['quantity']
			cons += [0 <= x['quantity']]              #Quantity must be >=0
		cons += [temp <= Fetch(msw, 'muni_ID', muni, 'disposal')]   #Sum for each county must be <= county production

		# processing capacity constraint
	facilities['facility_capacity'] = capacity_multiplier * facilities['cap_m3']
		# default capacity_multiplier is 
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
	print("defining problem")

	# DEFINE PROBLEM --> to MINIMIZE OBJECTIVE FUNCTION 
	prob = cp.Problem(cp.Minimize(obj), cons)

	#check that parameterized properly? (if not, will run sensitivities in oustide loop!)
	# print("Is DPP? ", prob.is_dcp(dpp=True))
	# print("Is DCP? ", prob.is_dcp(dpp=False))

	tzero = datetime.datetime.now()
	# print("-solving with DEFAULT...  time: ", tzero)
	# print("*********************************************")
	
	############################################################

	# SOLVE MODEL (across range of pareto vals)
	# contruct trade-off curve

	ton_values = []
	acre_values = []
	e_values = []
	area_values = []

	# print("######A VLAUE #########", a.value)
	if a.value is None:
		alpha_vals = np.linspace(0.0, 1.0, num=m)
		# alpha_vals = [0.0]
		for alpha in alpha_vals:
			print("*********************************************")
			print(" >> ALPHA VAL = {0}".format(alpha))
			a.value = alpha
			val = prob.solve(verbose = False)
	 
			now = datetime.datetime.now()
			print("TIME ELAPSED SOLVING: ", str(now - tzero))
			#calculate land area applied, total emissions and project cost
			land_app_dict, area_treated = LandApplication(landuse = rangelands, facilities = facilities, f2r = f2r)
			CO2mit, total_emis = TotalEmissions(msw, rangelands, facilities, grazed_rates, c2f, f2r)
			project_cost = ProjectCost(msw, rangelands, facilities, c2f, f2r)
			# print results!
			
			# print("VAL: (sort of meaningless now, just checking for solve) ", val) 
			cost_millions = (project_cost/(10**6))
			print("CO2 Mitigated (MMt CO2eq) = {0}".format(CO2mit))    
			print("TOTAL COST (Millions $) : ", cost_millions)
			print("Total_area (acres) = {}".format(area_treated))
			# print("TOTAL EMISSIONS (kg CO2e) : ", total_emis)
			# print("*********************************************")
			
			ton_price = (-project_cost/total_emis)*1000
			acre_price = project_cost/area_treated

			print("PRICE ($/tCO2) = {0}".format(ton_price))
			print("PRICE ($/acre) = {}".format(acre_price))

			ton_values.append(ton_price)
			acre_values.append(acre_price)
			e_values.append(CO2mit)
			area_values.append(area_treated)
			print("*********************************************")
			
			if alpha == 0.75: 
			# save output dicts!  
				c2f_values, f2r_values = SaveModelVars(c2f, f2r)
	else:
		# print(" SOLVING FOR USER SET ALPHA: ", a.value)
		val = prob.solve(verbose = False)
	 
		now = datetime.datetime.now()
		# print("TIME ELAPSED SOLVING: ", str(now - tzero))
		#calculate land area applied, total emissions and project cost
		land_app_dict, area_treated = LandApplication(landuse = rangelands, facilities = facilities, f2r = f2r)
		CO2mit, total_emis = TotalEmissions(msw, rangelands, facilities, grazed_rates, c2f, f2r)
		project_cost = ProjectCost(msw, rangelands, facilities, c2f, f2r)
		# print results!
		
		# print("VAL: (sort of meaningless now, just checking for solve) ", val) 
		cost_millions = (project_cost/(10**6))
		print("CO2 Mitigated (MMt CO2eq) = {0}".format(CO2mit))    
		print("TOTAL COST (Millions $) : ", cost_millions)
		print("Total_area (acres) = {}".format(area_treated))
		# print("TOTAL EMISSIONS (kg CO2e) : ", total_emis)
		# print("*********************************************")
		
		ton_price = (-project_cost/total_emis)*1000
		acre_price = project_cost/area_treated

		print("PRICE ($/tCO2) = {0}".format(ton_price))
		print("PRICE ($/acre) = {}".format(acre_price))
		
		c2f_values, f2r_values = SaveModelVars(c2f, f2r)
		print("*********************************************")

	return c2f_values, f2r_values, ton_values, acre_values, e_values, area_values #, land_app, cost_millions, CO2mit, abatement_cost

# Now run the beast!
# c2f, f2r , t, acres, e, area = RunModel()

# save c and f (from alpha =0.75) for plotting in  FLOW FIG
# see flowCNI.py for that code
# with open('out/c2f_a75.p', 'wb') as f:
# 	pickle.dump(c2f, f)
# with open('out/f2r_a75.p', 'wb') as f:
# 	pickle.dump(f2r, f)

# ###########################################################
# # PARETO FRONTIER PLOT
# ###########################################################
# raise Exception('pause here to plot')
# # pop off last element (bc skews plot)
# t1 = t[0:][:-1]
# # a1 = a[:-1]
# e1 = e[0:][:-1]
# area1 = area[:-1]
# acres1 = acres[:-1]


# print("starting PARETO PLOTS!!!")
# plt.rc('font', family='serif')

# plt.figure(figsize=(10,6))
# plt.subplot(121)
# # plot co2 sequestered by cost per ton
# plt.plot(e1[0:],t1[0:],marker='o', color ='green')
# # plt.xlabel('Incresing Cost Weighting In Objective Function', fontsize=12)
# plt.xlabel('Total MMT CO2eq', fontsize=12)
# plt.ylabel(r'$/tCO2', fontsize=14)
# plt.title('Pareto Frontier of Price per Ton CO2', fontsize=16)

# plt.subplot(122)
# #plot total area treated by cost per acre
# plt.plot(area1, acres1 ,marker='*', color = 'black')
# plt.xlabel('Acres Treated with Compost', fontsize=12)
# # plt.xlabel('Total MMT CO2eq', fontsize=12)
# plt.ylabel(r'$/Acre', fontsize=14)
# plt.title('Pareto Frontier of Price per Acre', fontsize=16)
# plt.show()



############################################################
############################################################
#					 MINIMIZE COST 
############################################################
############################################################

def RunModel_MinCost(
		# data sources
	msw = msw, 
	landuse = rangelands,
	facilities = facilities,
	seq_factors = grazed_rates, #alt is 
	# seq_f = -108,
	feedstock = 'food_and_green',

		# scaling parameters 
	detour_factor = 1.4,
	capacity_multiplier = 1,
		
		# emission factors 
	landfill_ef = 315, #kg CO2e / m3 = avoided emissions from waste going to landfill
	kilometres_to_emissions = 0.37, # kg CO2e/ m3 - km for 35mph speed 
	spreader_ef = 1.854, # kg CO2e / m3 = emissions from spreading compost
	process_emis = 11, # kg CO2e/ m3 = emisisons at facility from processing compost
	waste_to_compost = 0.58, #% volume change from waste to compost
	# cost parameters
	c2f_trans_cost = 0.412, #$/m3-km # transit costs (alt is 1.8)
	f2r_trans_cost = .206, #$/m3-km # transit costs
	spreader_cost = 5.8, #$/m3 # cost to spread
		
		#additional constraints
	# g = cp.Parameter(nonneg = True),
	d = cp.Parameter(nonneg=True)
		
	):  # minimizing on cost when a is 1, and on ghg when a is 0
	 

		# something about food/green waste here? or else earlier !

	print("-- setting feedstock and disposal") #if (DEBUG == True) else ()
	# change supply constraint by feedstock selected
	if feedstock == 'food_and_green':
		# combine food and green waste (wet tons) and convert to cubic meters
		# first, adjust food waste tonnage by fw reduction factor
		# print('feedstock food and green')
		msw_temp = msw.copy()
		# msw_temp.loc[(msw_temp['subtype']=='MSW_food'),'wt'] = msw_temp[msw_temp['subtype'] == 'MSW_food']['wt']*(1-fw_reduction)
		# then combine (sum) and convert to cubic meters	
		# print('new disposal')   
		msw_temp['disposal'] = msw_temp.groupby(['muni_ID'])['wt'].transform('sum') / (1.30795*(1/2.24))
		msw_temp = msw_temp.drop_duplicates(subset = 'muni_ID')
		# print('replacing')
		msw_temp['subtype'].replace({'MSW_green':'food_and_green'}, inplace = True)
		msw_temp['disposal_constraint'] = (d) * msw_temp['disposal'] 
		msw = msw_temp.copy()

	elif feedstock == 'food':
		msw_temp = msw.copy()
		# subset just food waste and convert wet tons to cubic meters
		msw_temp = msw_temp[(msw_temp['subtype'] == "MSW_food")]
		# msw['disposal'] = (1-fw_reduction)* counties['disposal_wm3']
		msw_temp['disposal'] = (1-fw_reduction)* msw_temp['wt'] / (1.30795*(1/2.24))
		msw = msw_temp.copy()

	elif feedstock == 'green':
		msw_temp = msw.copy()
		msw_temp = msw_temp[(msw_temp['subtype'] == "MSW_green")]
		msw_temp['disposal'] = msw_temp['wt'] / (1.30795*(1/2.24))
		msw = msw_temp.copy()

		# msw['disposal_constraint'] = (d) * msw['disposal'] 

		############################################################
		# decision variables
	print("--defining decision vars") #if (DEBUG == True) else ()
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
		#BUILD OBJECTIVE FUNCTION
	obj = 0

		# COSTS: collection cost, hauling cost, spreading cost
	print(" -- Objective: min(cost)--") #if (DEBUG == True) else ()
	# transport costs - county to facility
	for muni in msw['muni_ID']:
		print(" >  c2f cost for muni: ", muni) if (DEBUG == True) else ()
		for facility in facilities['SwisNo']:
			# print("c2f distance cost for facility: ", facility)
			x = c2f[muni][facility]
			if x['quantity'] is not None:
				v = x['quantity']
			else:
				v = 0.0
			obj += v *x['trans_cost'] # original cost opt 


	for facility in facilities['SwisNo']:
		print(" >  f2r  cost for facility and land: ", facility) if (DEBUG == True) else ()
		for land in landuse['OBJECTID']:
			x = f2r[facility][land]
			if x['quantity'] is not None:
				v = x['quantity']
			else:
				v = 0.0
			# project_cost due to transport of compost from facility to landuse
			obj += x['quantity'] * x['trans_cost'] # original cost opt
			# project_cost due to application of compost by manure spreader
			obj += x['quantity'] * spreader_cost # original cost opt

	
						
	############################################################
	#Constraints
		# supply constraint, processing capacity, land-use, throughput!
		# print("--subject to constraints") if (DEBUG == True) else ()

	cons = []
		
	#supply constraint (quantity can't exceed msw supply)
	for muni in msw['muni_ID']:
		temp = 0
		for facility in facilities['SwisNo']:
			print("supply constraints -- muni: ",muni, " to facility: ", facility) if (DEBUG == True) else ()
			x    = c2f[muni][facility]
			temp += x['quantity']
			cons += [0 <= x['quantity']]              #Quantity must be >=0
		cons += [temp <= Fetch(msw, 'muni_ID', muni, 'disposal')] #Sum for each county must be <= county production
		cons += [temp >= Fetch(msw, 'muni_ID', muni, 'disposal_constraint')]   

		# processing capacity constraint
	facilities['facility_capacity'] = capacity_multiplier * facilities['cap_m3']
		# default capacity_multiplier is 
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

		# Emissions: collection, processing, avoided landfill
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
			# if x['quantity'] is not None:
			# 	v = x['quantity']  
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
			# if x['quantity'] is not None:
			# 	applied_amount = x['quantity']  
			else:
				applied_amount = 0.0 
			# emissions due to transport of compost from facility to landuse
			total_emis += x['trans_emis']* applied_amount
			# emissions due to application of compost by manure spreader
			total_emis += spreader_ef * applied_amount
			# sequestration of applied compost
			total_emis += seq_f * applied_amount

	# Trying emissions constraints to force some cost!		
	# cons += [-total_emis >= g]

	# cost constraint?
	# cons += [obj >= g]


		############################################################
	print("defining problem")

	# DEFINE PROBLEM --> to MINIMIZE OBJECTIVE FUNCTION 
	prob = cp.Problem(cp.Minimize(obj), cons)

	tzero = datetime.datetime.now()
	print("-solving with DEFAULT...  time: ", tzero)
	print("*********************************************")

	# SOLVE MODEL TO GET FINAL VALUE (which will be in terms of kg of CO2)
		#solver = cp.GUROBI,
	delta_vals = np.array([0.01, 0.05, 0.1])
	for delta in delta_vals:
		print("*********************************************")
		print(" >> DELTA (DISPOSAL CONSTRAINT) VAL = {0}".format(delta))
		d.value = delta
		val = prob.solve(solver = cp.MOSEK, verbose = True)
	 
		now = datetime.datetime.now()
		print("TIME ELAPSED SOLVING: ", str(now - tzero))
		print("*********************************************")

		land_app_dict, area_treated = LandApplication(landuse = rangelands, facilities = facilities, f2r = f2r)

		print("AREA TREATED", area_treated)

		############################################################
		print("VAL: ", val) 
		print("TOTAL EMIS: ", total_emis)
		cost_millions = (val/(10**6))    
		print("TOTAL COST (Millions $) : ", cost_millions)
		# print("TOTAL EMISSIONS (kg CO2e) : ", total_emis)
		# print("*********************************************")
		# print("CO2 Mitigated (MMt CO2eq) = {0}".format(CO2mit))
			
		c2f_values, f2r_values = SaveModelVars(c2f, f2r)


	return c2f_values, f2r_values #, land_app, cost_millions, CO2mit, abatement_cost

# c, f = RunModel_MinCost()

## ADD PARAMS!
# # Initialize parameter with a value.
# rho = cp.Parameter(nonneg=True, value=2)
############################################################
############################################################
#					 MINIMIZE EMISSIONS 
############################################################
############################################################

def RunModel_MinEmis(
		# data sources
	msw = msw, 
	landuse = rangelands,
	facilities = facilities,
	seq_factors = grazed_rates, 
		# seq_f = -108,
	feedstock = 'food_and_green',

		# scaling parameters 
	detour_factor = 1.4,
	capacity_multiplier = 1,
	fw_reduction = 0,
		
		# emission factors 
	landfill_ef = 315, #kg CO2e / m3 = avoided emissions from waste going to landfill
	kilometres_to_emissions = 0.37, # kg CO2e/ m3 - km for 35mph speed 
	spreader_ef = 1.854, # kg CO2e / m3 = emissions from spreading compost
	process_emis = 11, # kg CO2e/ m3 = emisisons at facility from processing compost
	waste_to_compost = 0.58, #% volume change from waste to compost
	# cost parameters
	c2f_trans_cost = 0.412, #$/m3-km # transit costs (alt is 1.8)
	f2r_trans_cost = .206, #$/m3-km # transit costs
	spreader_cost = 5.8 #$/m3 # cost to spread
		):  # minimizing on cost when a is 1, and on ghg when a is 0
		
		# something about food/green waste here? or else earlier !

	print("-- setting feedstock and disposal") #if (DEBUG == True) else ()
	# change supply constraint by feedstock selected
	if feedstock == 'food_and_green':
		# combine food and green waste (wet tons) and convert to cubic meters
		# first, adjust food waste tonnage by fw reduction factor
		# print('feedstock food and green')
		msw_temp = msw.copy()
		# msw_temp.loc[(msw_temp['subtype']=='MSW_food'),'wt'] = msw_temp[msw_temp['subtype'] == 'MSW_food']['wt']*(1-fw_reduction)
		# then combine (sum) and convert to cubic meters	
		# print('new disposal')   
		msw_temp['disposal'] = msw_temp.groupby(['muni_ID'])['wt'].transform('sum') / (1.30795*(1/2.24))
		msw_temp = msw_temp.drop_duplicates(subset = 'muni_ID')
		# print('replacing')
		msw_temp['subtype'].replace({'MSW_green':'food_and_green'}, inplace = True)
		msw = msw_temp.copy()

		# for m in msw['muni_ID']:
		# 	d = Fetch(msw, 'muni_ID', m, 'disposal')
		# 	print("DISPOSAL: ", d)

	elif feedstock == 'food':
		msw_temp = msw.copy()
		# subset just food waste and convert wet tons to cubic meters
		msw_temp = msw_temp[(msw_temp['subtype'] == "MSW_food")]
		# msw['disposal'] = (1-fw_reduction)* counties['disposal_wm3']
		msw_temp['disposal'] = (1-fw_reduction)* msw_temp['wt'] / (1.30795*(1/2.24))
		msw = msw_temp.copy()

	elif feedstock == 'green':
		msw_temp = msw.copy()
		msw_temp = msw_temp[(msw_temp['subtype'] == "MSW_green")]
		msw_temp['disposal'] = msw_temp['wt'] / (1.30795*(1/2.24))
		msw = msw_temp.copy()

		
		############################################################
		# decision variables
	print("--defining decision vars") #if (DEBUG == True) else ()
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
		#BUILD OBJECTIVE FUNCTION
	obj = 0

	print(" -- Objective: min(emissions)--") #if (DEBUG == True) else ()

	# EMISIONS FROM C TO F (AND at at Facility)
		# Emissions: collection, processing, avoided landfill
		
	count = 0 # for keeping track of the municipality count
	# emissions due to waste remaining in muni
	for muni in msw['muni_ID']:
		count += 1
		print("muni ID: ", muni, " ## ", count,  "-- EMISSIONS") if (DEBUG == True) else ()
		# county_disposal = Fetch(msw, 'muni_ID', muni, 'disposal')
		temp = 0
		for facility in facilities['SwisNo']:
			print("c2f - facility: ", facility) if (DEBUG == True) else ()
			#grab quantity and sum for each county
			x    = c2f[muni][facility]
			if x['quantity'] is not None:
				v = x['quantity']  
			else:
				v = 0.0
			temp += v
						
			# emissions due to transport of waste from county to facility 
			obj += v * x['trans_emis'] # for use as constraint in cost opt

			# emissions due to processing compost at facility
			obj += v * process_emis # for use as constraint in cost opt

		 # temp = sum([c2f[muni][facility]['quantity'] for facilities in facilities['SwisNo']]) #Does the same thing
		obj += landfill_ef*(-temp) #AVOIDED Landfill emissionsb # # for use as constraint in cost opt
		# obj += (1-a) * landfill_ef*(-temp) #AVOIDED Landfill emissions ## pareto analysis

		# obj += landfill_ef*(county_disposal - temp) #PENALTY for the waste stranded in county
	


	# EMISSIONS FROM F TO R (AND ON Rangeland)
		# Emissions: hauling, spreading, sequestration
	for facility in facilities['SwisNo']:
		print("SW facility: ", facility, "--to LAND") if (DEBUG == True) else ()
		for land in landuse['OBJECTID']:
			print('f2r - land #: ', land) if (DEBUG == True) else ()

			# pull county specific sequestration rate!!
			county = Fetch(landuse, 'OBJECTID' , land, 'COUNTY')
			# print("COUNTYY: ", county)
			seq_f = Fetch(seq_factors, 'County', county, 'seq_f')
			# print("SEQ F: ", seq_f)

			x = f2r[facility][land] 
			if x['quantity'] is not None:
				applied_amount = x['quantity']  
			else:
				applied_amount = 0.0 
								
			# emissions due to transport of compost from facility to landuse
			obj += x['trans_emis']* applied_amount # # for use as constraint in cost opt
			# obj += (1-a) * x['trans_emis']* applied_amount # pareto analysis

			# emissions due to application of compost by manure spreader
			obj += spreader_ef * applied_amount # # for use as constraint in cost opt
			# obj += (1-a) * spreader_ef * applied_amount # pareto analysis

			# sequestration of applied compost
			obj += seq_f * applied_amount # # for use as constraint in cost opt
			# obj += (1-a) * (-seq_f) * applied_amount # pareto analysis
						
						
	# EMISSIONS FROM F TO R (AND ON Rangeland)
	# for land in landuse['OBJECTID']:
	# 	print("LAND #", land)
				# pull county specific sequestration rate!!
	# 	# county = Fetch(landuse, 'OBJECTID' , land, 'COUNTY')
	# 	# print("COUNTYYYYYYYYYYYYYY: ", county)
	# 	# seq_f = Fetch(seq_factors, 'County', county, 'seq_f')
	# 	# print("SEQ F: ", seq_f)
	# 	# seq_f = 108

	# 	for facility in facilities['SwisNo']:
	# 		print('SW facility', facility)
	# 		x = f2r[facility][land]
	# 		if x['quantity'] is not None:
	# 			applied_amount = x['quantity']  
	# 		else:
	# 			applied_amount = 0.0 
								
	# 		# emissions due to transport of compost from facility to landuse
	# 		obj += x['trans_emis']* applied_amount # # for use as constraint in cost opt
	# 		# obj += (1-a) * x['trans_emis']* applied_amount # pareto analysis

	# 		# emissions due to application of compost by manure spreader
	# 		obj += spreader_ef * applied_amount # # for use as constraint in cost opt
	# 		# obj += (1-a) * spreader_ef * applied_amount # pareto analysis

	# 		# sequestration of applied compost
	# 		obj += seq_f * applied_amount # # for use as constraint in cost opt
	# 		# obj += (1-a) * (-seq_f) * applied_amount # pareto analysis
						
	############################################################
	#Constraints
		# supply constraint, processing capacity, land-use, throughput!
		# print("--subject to constraints") if (DEBUG == True) else ()

	cons = []
		
	# supply constraint (quantity can't exceed msw supply)
	for muni in msw['muni_ID']:
		temp = 0
		for facility in facilities['SwisNo']:
			print("supply constraints -- muni: ",muni, " to facility: ", facility) if (DEBUG == True) else ()
			x    = c2f[muni][facility]
			temp += x['quantity']
			cons += [0 <= x['quantity']]              #Quantity must be >=0
		cons += [temp <= Fetch(msw, 'muni_ID', muni, 'disposal')]   #Sum for each county must be <= county production

		# processing capacity constraint
	facilities['facility_capacity'] = capacity_multiplier * facilities['cap_m3']
		# default capacity_multiplier is 
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
	print("defining problem")

	# DEFINE PROBLEM --> to MINIMIZE OBJECTIVE FUNCTION 
	prob = cp.Problem(cp.Minimize(obj), cons)

	tzero = datetime.datetime.now()
	print("-solving with DEFAULT...  time: ", tzero)
	print("*********************************************")

	# SOLVE MODEL TO GET FINAL VALUE (which will be in terms of kg of CO2)
	val = prob.solve(verbose = False)
 
	now = datetime.datetime.now()
	print("TIME ELAPSED SOLVING: ", str(now - tzero))
	print("*********************************************")


	#calc other important values after solve
	land_app_dict, area_treated = LandApplication(landuse = rangelands, facilities = facilities, f2r = f2r)
	project_cost = ProjectCost(msw, rangelands, facilities, c2f, f2r)
	# CO2mit2 = TotalEmissions(msw, rangelands, facilities, grazed_rates, c2f, f2r)
	print("AREA TREATED", area_treated)

	############################################################
	# print("VAL: ", val) 
	cost_millions = (project_cost/(10**6))    
	print("TOTAL COST (Millions $) : ", cost_millions)
	print("TOTAL EMISSIONS (kg CO2e) : ", val)
	CO2mit = -val/(10**9)
	# print("*********************************************")
	print("CO2 Mitigated (MMt CO2eq) = {0}".format(CO2mit))

	print("PRICE ($/tCO2) = {0}".format((-project_cost/val)*1000))
	print("PRICE ($/acre) = {}".format(project_cost/area_treated))
		
	c2f_values, f2r_values = SaveModelVars(c2f, f2r)

	return c2f_values, f2r_values, land_app_dict, #cost_millions, CO2mit, abatement_cost


# c, f, l = RunModel_MinEmis()

# SEPARATE FXNS TO CALC COST AND EMISSIONS AFTERWARD!!








