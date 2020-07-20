#biomass_polygons_prep.py

import cvxpy as cp
import numpy as np
import os
import datetime
from os.path import join as opj
import json


import pandas as pd
import shapely as shp
import geopandas as gpd
import scipy as sp

from biomass_preprocessing import MergeInventoryAndCounty

DATA_DIR = "/Users/anayahall/projects/compopt/data"
RESULTS_DIR = "/Users/anayahall/projects/compopt/results"

gbm_shp, tbm_shp = MergeInventoryAndCounty(
    gross_inventory     = opj(DATA_DIR, "raw/biomass.inventory.csv"),
    technical_inventory = opj(DATA_DIR, "raw/biomass.inventory.technical.csv"),
    county_shapefile    = opj(DATA_DIR, "raw/CA_Counties/CA_Counties_TIGER2016.shp"),
    fips_data           = opj(DATA_DIR, "interim/CA_FIPS.csv")
)

g = gbm_shp[gbm_shp['category'] == 'organic fraction municipal solid waste']

out = r"clean/gbm_polys.shp"
g.to_file(driver='ESRI Shapefile', filename=opj(DATA_DIR, out))

# gbm_shp.to_file(driver='ESRI Shapefile', filename=out)

# print("p BIOMASS SHAPEFILE SAVED RUNNING")