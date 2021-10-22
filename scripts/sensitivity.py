# sensitivity.py
import numpy as np
import os
import cvxpy as cp
# import datetime
# from os.path import join as opj
# import json
# import sys
# import gurobipy as gp
# import pickle
# import matplotlib.pyplot as plt

import pandas as pd
import shapely as shp
import geopandas as gpd
import scipy as sp

from model import RunModel


############################################################
# SENSITIVITY ANALYSIS 
############################################################

	# landfill_ef = 315, #kg CO2e / m3 = avoided emissions from waste going to landfill
	# kilometres_to_emissions = 0.37, # kg CO2e/ m3 - km for 35mph speed 
	# spreader_ef = 1.854, # kg CO2e / m3 = emissions from spreading compost
	# process_emis = 11, # kg CO2e/ m3 = emisisons at facility from processing compost
	# waste_to_compost = 0.58, #% volume change from waste to compost
	# # cost parameters
	# c2f_trans_cost = 0.412, #$/m3-km # transit costs (alt is 1.8)
	# f2r_trans_cost = .206, #$/m3-km # transit costs
	# spreader_cost = 5.8, #$/m3 # cost to spread	

# raise Exception ("pre-senstivities")

## Landfill emission factors (l)
lvals = [182.5, 315, 472.5]
## Transportation emission factors (t)
tvals = [0.185, 0.37, 0.55]
## Compost processing emission factors (y)
yvals = [0.29, 0.58, 0.87]

## Collection cost (d)
dvals = [0.026, 0.412, 0.618]
## Hauling cost (e)
evals = [0.103, 0.206, 0.309]
## Spreading cost (h)
hvals = [2.9, 5.8, 8.7]

print(" >> LANDFILL EF SENSIVITIY <<")
for li in lvals:
	print("Landfill EF: ", li)
	c2f, f2r , t, acres, e, area = RunModel(landfill_ef = li, 
		a = cp.Parameter(value=0.75))

print("*********************************************")
print("*********************************************")
print(" >> TRANSPORTATION EF SENSIVITIY <<")
for ti in tvals:
	print(" >>>>> Transportation EF: ", ti)
	c2f, f2r , t, acres, e, area = RunModel(kilometres_to_emissions = ti, 
		a = cp.Parameter(value=0.75))

print("*********************************************")
print("*********************************************")
print(" >> PROCCESSING EF SENSIVITIY <<")
for yi in yvals:
	print(" >>>>> Processing EF: ", yi)
	c2f, f2r , t, acres, e, area = RunModel(process_emis = yi, 
		a = cp.Parameter(value=0.75))

print("*********************************************")
print("*********************************************")
print(" >> COLLECTION COST SENSIVITIY <<")
for di in dvals:
	print(" >>>>> Collection Cost: ", di)
	c2f, f2r , t, acres, e, area = RunModel(c2f_trans_cost = di, 
		a = cp.Parameter(value=0.75))

print("*********************************************")
print("*********************************************")
print(" >> HAULING COST SENSIVITIY <<")
for ei in evals:
	print(" >>>>> Hauling Cost: ", ei)
	c2f, f2r , t, acres, e, area = RunModel(f2r_trans_cost = ei, 
		a = cp.Parameter(value=0.75))

print("*********************************************")
print("*********************************************")
print(" >> SPREADING COST SENSIVITIY <<")
for hi in hvals:
	print(" >>>>> Hauling Cost: ", hi)
	c2f, f2r , t, acres, e, area = RunModel(spreader_cost = hi, 
		a = cp.Parameter(value=0.75))
