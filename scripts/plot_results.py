import pandas as pd
import os
import numpy as np
import shapely as sp
from os.path import join as opj

import matplotlib.pyplot as plt

import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
import json

DATA_DIR = "/Users/anayahall/projects/compopt/data"
OUT_DIR = "/Users/anayahall/projects/compopt/maps"


# temp df for testing PlotCountyResults function below
x = gpd.read_file("/Users/anayahall/projects/compopt/test_fgbaseresults.shp")


def MakeCountyGDF(dict):
    county_results = dict
    
    print("cleaning dict")
    # print("TESTTTTTS")
    for county in county_results.keys():
        c = county_results[county]
        c['netGHG'] = c['TOTAL_emis']-315*c['output']
        c['abcost'] = -(c['TOTAL_cost']/c['netGHG'])*1000
        c['COUNTY'] = county
    print("to df")
    countyresults = pd.DataFrame.from_dict(county_results, orient = 'index')
    countyresults = countyresults.fillna(0)
    print("merging with CA shapefile")
    # read in data
    CA = gpd.read_file(opj(DATA_DIR, "raw/CA_Counties/CA_Counties_TIGER2016.shp"))
    CA = CA.to_crs(epsg=4326)
    CA['COUNTY'] = CA['NAME']
    CA = CA[['COUNTY', 'geometry']]
    results_shp = pd.merge(CA, countyresults, on = 'COUNTY')

    return results_shp


def PlotCountyResults(gdf, plotvar, cmap = 'viridis', log = False):

    results_shp = gdf

        # CA = gpd.read_file(opj(DATA_DIR, "raw/CA_Counties/CA_Counties_TIGER2016.shp"))
        # CA = CA.to_crs(epsg=4326)

    print("Plotting: ", plotvar)
    if log == True:
        column = np.log(gdf[plotvar])
    else:
        column = plotvar

    # first plot
    f, ax = plt.subplots()
    # CA.plot(ax = ax, color = "white", figsize = (10,10), linewidth=0.1, edgecolor = "grey")
    results_shp.plot(ax= ax, column = column, cmap = cmap, alpha = .5, legend = True)
    ax.axis('off')
    ax.set_title(str(column), fontdict={'fontsize': '12', 'fontweight' : '3'})
    plt.savefig(opj(OUT_DIR, str(column) + "_TEST.png"), dpi=300)

    print("done")
    return 

## OTHER RESULTS
# try to plot network, first try with raw data, then try to merge c2f and f2r vals and plot those

def PrepMovedVals(c2fvals, f2rvals):
    # will need to clean - turn into df? then add lat lon from original data
    return ("incomplete function")

# facilities = gpd.read_file(opj(DATA_DIR, "clean/clean_swis.shp"))
# facilities.rename(columns={'County':'COUNTY'}, inplace=True)
# facilities = facilities.to_crs(epsg=4326)


# attempting to label CA polygons
# CA['coords'] = CA['geometry'].apply(lambda x: x.representative_point().coords[:])
# CA['coords'] = [coords[0] for coords in CA['coords']]

# # second plot
# f, ax = plt.subplots()
# CA.plot()
# for idx, row in CA.iterrows():
#     plt.annotate(s=row['NAME'], xy=row['coords'],
#                  horizontalalignment='center')
# plt.savefig(opj(OUT_DIR, "CA_Counties.png"), dpi=900)
