# THIS IS THE MAIN SCRIPT FROM WHICH THE COMPOSTLP MODEL IS CALLED
# edited for running on slurm where I mostly care about getting the abatement curve set
# will do one run at midlevel afterwards to get the flows!

############################################################
# Load packages
import cvxpy as cp
import numpy as np
import os
from os.path import join as opj

import pandas as pd
import shapely as shp
import geopandas as gpd
import scipy as sp
import pickle
import time
# import yagmail


############################################################
# Settings
# Change this to activate/decativate print statements throughout
DEBUG = True

# run on crops? 
CROPLANDS = False

############################################################
# Load necessary other scripts (CompostLP and dataload)
print(" - main - packages loaded - import compost LP script now") if (DEBUG == True) else ()

from compostLP import Haversine, Distance, Fetch, SolveModel, SaveModelVars

print(" - main - starting solves!!! ") if (DEBUG == True) else ()


from dataload import msw, rangelands, facilities, grazed_rates

############################################################
### RUN SCENARIOS! #########################################
############################################################

# this loop is just to build out the abatement cost curve
A_levels = np.arange(0, 1, 0.1)

resultsarray = np.zeros([len(A_levels),2])


c = 0
for i in A_levels:
    print("Count: ", c) if (DEBUG == True) else ()
    run_name = str("run_"+str(i))
    print("RUNNING: ", run_name) if (DEBUG == True) else ()
    # RUN THE MODEL!!!
    c2f_val, f2r_val, land_app, cost_millions, CO2mit, abatement_cost = SolveModel(scenario_name = run_name,
        a = i,
        msw = msw,
        landuse = rangelands,
        facilities = facilities,
        feedstock = "food_and_green")

    # Send EMAIL w results
#     PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)
    resultsarray[c,0] = CO2mit
    resultsarray[c,1] = abatement_cost
    c += 1
    print("Run #", i, "done!!") if (DEBUG == True) else ()

    
with open('out/resultsarray.p', 'wb') as f:
    pickle.dump(resultsarray, f)

print(resultsarray)
############################################################
# now run at midlevel and save flows
# print("---about to run at midlevel ish---")
# run_name = 'ec1'

# c2f_val, f2r_val, land_app, cost_millions, CO2mit, abatement_cost = SolveModel(scenario_name = run_name,
#         emissions_constraint = 1,
#         msw = msw,
#         landuse = rangelands,
#         facilities = facilities,
#         feedstock = "food_and_green")

# #     # SAVE RESULTS C2F
# # with open('results/latest_c2f.p', 'wb') as f:
# with open('out/c2f_ec1.p', 'wb') as f:
#     pickle.dump(c2f_val, f)

# # SAVE RESULTS F2R
# with open('out/f2r_ec1.p', 'wb') as f:
#     pickle.dump(f2r_val, f)

# # SAVE LAND APPLICATION DICT
# with open('out/landapp_ec1.p', 'wb') as f:
#     pickle.dump(land_app, f)
############################################################

print("EXITING PROGRAM") if (DEBUG == True) else ()

exit()