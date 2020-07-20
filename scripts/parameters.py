# parameters

## ARGUMENTS
waste_type = # select out of biomass inventory



# SWIS
#swf_type 	# solid waste facility type
#opc			# operating cost
tc			# transportation cost


# LAND
# cl_rate
# rl_rate


### emission factors

#### transportation


#### processing


### conversion factors

# bdt waste to waste

fw_mc = .7 #food waste moisture content, source: Breunig SI pg S13
gw_mc = .5 # green waste moisture content

# general formula 
# wt = bdt * (1+mc)  # wet tonnes = bone dry tonnes*(1+mc)
fw_wt = fw_bdt * (1 + fw_mc)
gw_wt = gw_bdt * (1 + gw_mc)

# waste to compost (input to output)
wtc = .5

# 

