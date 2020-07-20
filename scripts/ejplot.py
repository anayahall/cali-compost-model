#ejplot.py
#plot calenviroscreen data with compost facilities

import pandas as pd
import os
import numpy as np
import shapely as sp
from os.path import join as opj

import matplotlib.pyplot as plt
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
# from mpl_toolkits.basemap import Basemap
#CALIFORNIA BOUNDING BOX westlimit=-124.48; southlimit=32.53; eastlimit=-114.13; northlimit=42.01
# get_ipython().magic('matplotlib inline')

# from biomass_preprocessing import MergeInventoryAndCounty

DATA_DIR = "/Users/anayahall/projects/compopt/data"
OUT_DIR = "/Users/anayahall/projects/compopt/maps"

CA = gpd.read_file(opj(DATA_DIR, "raw/CA_Counties/CA_Counties_TIGER2016.shp"))
CA = CA.to_crs(epsg=4326)
CA['COUNTY'] = CA['NAME']
# print("Ca_proj crs: ", CA_proj.crs)
CA.head()

# LOAD BIOMASS DATA
# gbm_pts, tbm_pts = MergeInventoryAndCounty(
#     gross_inventory     = opj(DATA_DIR, "raw/biomass.inventory.csv"),
#     technical_inventory = opj(DATA_DIR, "raw/biomass.inventory.technical.csv"),
#     county_shapefile    = opj(DATA_DIR, "raw/CA_Counties/CA_Counties_TIGER2016.shp"),
#     fips_data           = opj(DATA_DIR, "interim/CA_FIPS.csv")
# )
# counties = tbm_pts[tbm_pts['year'] == 2014] # could change to GBM

ej = gpd.read_file(opj(DATA_DIR, "calenviroscreen/CESJune2018Update_SHP/CES3June2018Update.shp"))


# rangelands = gpd.read_file(opj(DATA_DIR, "raw/CA_FMMP_G/gl_bycounty/grazingland_county.shp"))
# # rangelands = gpd.read_file(opj(DATA_DIR, "raw/CA_FMMP_G/grazingland_dis/CA_grazingland.shp"))
# rangelands = rangelands.to_crs(epsg=4326)

swis =  gpd.read_file(opj(DATA_DIR, "clean/clean_swis.shp"))

f, ax = plt.subplots()
CA.plot(ax = ax, color = "white", figsize = (10,10), linewidth=0.2, edgecolor = "grey")
ej.plot(ax= ax, color = "reds", alpha = .5)
ax.axis('off')
# ax.set_title('Rangelands', fontdict={'fontsize': '12', 'fontweight' : '3'})
# plt.savefig(opj(OUT_DIR, "rangelands.png"), dpi=300)