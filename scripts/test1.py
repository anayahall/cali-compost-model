# test1.py
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
		
		# pareto tradeoff
	a = cp.Parameter(nonneg=True)): # minimizing on cost when a is 1, and on ghg when a is 0
		
	m=1
	# something about food/green waste here? or else earlier !

	print("-- setting feedstock and disposal")
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
			# obj += x['quantity']*x['trans_cost'] # original cost opt
			obj += a * x['quantity']*x['trans_cost'] # new pareto analysis

	for facility in facilities['SwisNo']:
		print(" >  f2r  cost for facility and land: ", facility) if (DEBUG == True) else ()
		for land in landuse['OBJECTID']:
			x = f2r[facility][land]
						
			# project_cost due to transport of compost from facility to landuse
			# obj += x['quantity'] * x['trans_cost'] # original cost opt
			obj += a * x['quantity'] * x['trans_cost'] # new pareto analysis

			# project_cost due to application of compost by manure spreader
			# obj += x['quantity'] * spreader_cost # original cost opt
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
			# total_emis += v * x['trans_emis'] # for use as constraint in cost opt
			obj += (1-a) * v * x['trans_emis'] # pareto analysis

			# emissions due to processing compost at facility
			# total_emis += v * process_emis # for use as constraint in cost opt
			obj += (1-a) * v * process_emis # pareto analysis

	#    temp = sum([c2f[muni][facility]['quantity'] for facilities in facilities['SwisNo']]) #Does the same thing
		# total_emis += landfill_ef*(-temp) #AVOIDED Landfill emissionsb # # for use as constraint in cost opt
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
			# total_emis += x['trans_emis']* applied_amount # # for use as constraint in cost opt
			obj += (1-a) * x['trans_emis']* applied_amount # pareto analysis

			# emissions due to application of compost by manure spreader
			# total_emis += spreader_ef * applied_amount # # for use as constraint in cost opt
			obj += (1-a) * spreader_ef * applied_amount # pareto analysis

			# sequestration of applied compost
			# total_emis += seq_f * applied_amount # # for use as constraint in cost opt
			obj += (1-a) * (-seq_f) * applied_amount # pareto analysis
	
	### REDO OTHER WAY?
	# for land in landuse['OBJECTID']:
	# 	print("LAND #", land) if (DEBUG == True) else ()
 #        # pull county specific sequestration rate!!
	# 	county = Fetch(landuse, 'OBJECTID' , land, 'COUNTY')
	# 	# print("COUNTYYYYYYYYYYYYYY: ", county)
	# 	seq_f = Fetch(seq_factors, 'County', county, 'seq_f')
	# 	# print("SEQ F: ", seq_f)
	# 	# seq_f = 108

	# 	for facility in facilities['SwisNo']:
	# 		print('SW facility', facility) if (DEBUG == True) else ()
	# 		x = f2r[facility][land]
	# 		if x['quantity'] is not None:
	# 			applied_amount = x['quantity']  
	# 		else:
	# 			applied_amount = 0.0 
								
	# 		# emissions due to transport of compost from facility to landuse
	# 		# total_emis += x['trans_emis']* applied_amount # # for use as constraint in cost opt
	# 		obj += (1-a) * x['trans_emis']* applied_amount # pareto analysis

	# 		# emissions due to application of compost by manure spreader
	# 		# total_emis += spreader_ef * applied_amount # # for use as constraint in cost opt
	# 		obj += (1-a) * spreader_ef * applied_amount # pareto analysis

	# 		# sequestration of applied compost
	# 		# total_emis += seq_f * applied_amount # # for use as constraint in cost opt
	# 		obj += (1-a) * (-seq_f) * applied_amount # pareto analysis
						
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

	tzero = datetime.datetime.now()
	# print("-solving with DEFAULT...  time: ", tzero)
	print("*********************************************")
	
	############################################################

	# SOLVE MODEL (across range of pareto vals)
	# contruct trade-off curve

	ton_values = []
	acre_values = []
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
		# print("*********************************************")

		# save output dicts!  
		c2f_values, f2r_values = SaveModelVars(c2f, f2r)
		print("*********************************************")
	
	# print("starting to PLOT?!?!?!!")

	# # plt.rc('text', usetex=True)
	# plt.rc('font', family='serif')

	# plt.figure(figsize=(6,6))
	# plt.plot(a,t[0:])
	# 	# a[i], t[i][0]
	# plt.xlabel(r'\alpha', fontsize=16)
	# plt.ylabel(r'$/tCO2', fontsize=16)
	# plt.title('TEST PLOT', fontsize=16)
	# plt.show()

	# # print(alpha_vals)
	# # print(ton_values)

	return c2f_values, f2r_values, alpha_vals, ton_values #, land_app, cost_millions, CO2mit, abatement_cost

c, f , a, t, = RunModel()
