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



############################################################
# Settings
# Change this to activate/decativate print statements throughout
DEBUG = True

# run on crops? 
CROPLANDS = False

############################################################
# Load necessary other scripts (CompostLP and dataload)
print(" - main - packages loaded - import compost LP script now") if (DEBUG == True) else ()

from model import Haversine, Distance, Fetch, SolveModel, SaveModelVars

print(" - main - starting solves!!! ") if (DEBUG == True) else ()


from dataload import msw, rangelands, facilities, grazed_rates

