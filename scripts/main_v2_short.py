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
print(" - main - packages loaded - import compost LP function now") if (DEBUG == True) else ()

from compostLP import Haversine, Distance, Fetch, SolveModel, SaveModelVars

print(" - main - starting solves!!! ") if (DEBUG == True) else ()


from dataload import msw, rangelands, facilities, grazed_rates, perennial_rates

############################################################
### RUN SCENARIOS! #########################################
############################################################
 
############################################################
# now run at midlevel and save flows
print("---about to run at midlevel ish---")
run_name = 'RL_a05'

c2f_val, f2r_val, land_app, cost_millions, CO2mit, abatement_cost = SolveModel(scenario_name = run_name,
        a = 0.5,
        msw = msw,
        landuse = rangelands,
        facilities = facilities,
        seq_factors = grazed_rates,
        feedstock = "food_and_green")

#     # SAVE RESULTS C2F
# with open('results/latest_c2f.p', 'wb') as f:
with open('out/c2f_RL5.p', 'wb') as f:
    pickle.dump(c2f_val, f)

# SAVE RESULTS F2R
with open('out/f2r_RL5.p', 'wb') as f:
    pickle.dump(f2r_val, f)

# SAVE LAND APPLICATION DICT
with open('out/landapp_RL5.p', 'wb') as f:
    pickle.dump(land_app, f)


raise Exception("SINGLE RUN - RANGELANDS")    
#### ##### ##### ##### #### #### #### #### ##### 
run_name = 'CL_a05'

c2f_val, f2r_val, land_app, cost_millions, CO2mit, abatement_cost = SolveModel(scenario_name = run_name,
        a = 0.5,
        msw = msw,
        landuse = croplands,
        facilities = facilities,
        seq_factors = perennial_rates,
        feedstock = "food_and_green")

#     # SAVE RESULTS C2F
# with open('results/latest_c2f.p', 'wb') as f:
with open('out/c2f_CL5.p', 'wb') as f:
    pickle.dump(c2f_val, f)

# SAVE RESULTS F2R
with open('out/f2r_CL5.p', 'wb') as f:
    pickle.dump(f2r_val, f)

# SAVE LAND APPLICATION DICT
with open('out/landapp_CL5.p', 'wb') as f:
    pickle.dump(land_app, f)
    
############################################################

print("EXITING PROGRAM") if (DEBUG == True) else ()

exit()