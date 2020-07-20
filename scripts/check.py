#checking gross v tech biomass
# First, load packages
import pandas as pd
import os
from os.path import join as opj
import numpy as np
import fiona
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
import plotly.plotly as py

from biomass_preprocessing import MergeInventoryAndCounty

#change wd
DATA_DIR = "/Users/anayahall/projects/compopt/data"

# from fxns import epsg_meters
##################################################################
# tbm = pd.read_csv("data/raw/biomass.inventory.technical.csv")


# GOALS FOR FRIDAY AFTERNOON
# function that takes gdf and plots choropleth maps of a set of variables 
# (facility emissions to plot with EJ ; abatement cost ; quantity out, area applied, etc)


# TUESDAY NIGHT PLOTS
# testf2rdf = pd.DataFrame.from_dict(testf2r)
# testf2rdf['sum'] = testf2rdf.sum(axis = 1, skipna = True) 

# merge with rangelands, then plot

rangelands = gpd.read_file(opj(DATA_DIR, "raw/CA_FMMP_G/gl_bycounty/grazingland_county.shp"))
# rangelands = gpd.read_file(opj(DATA_DIR, "raw/CA_FMMP_G/grazingland_dis/CA_grazingland.shp"))
rangelands = rangelands.to_crs(epsg=4326)
