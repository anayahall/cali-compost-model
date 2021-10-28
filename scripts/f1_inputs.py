# f1_inputs.py

# figure #1 - inputs

# PLOTS - inputs

# import modules
import pandas as pd
import numpy as np
import shapely as shp
import geopandas as gpd
from os.path import join as opj
import matplotlib.pyplot as plt
import os
import pickle

from region import add_region_variable
from california_cropland_cleaning import cleancropdata
# import plotly.graph_objects as go
import plotly


# utilize Fetch function
def Fetch(df, key_col, key, value):
	#counties['disposal'].loc[counties['COUNTY']=='San Diego'].values[0]
	return df[value].loc[df[key_col]==key].values[0]

# set data path
DATA_DIR = "/Users/anayahall/projects_v2/cali-compost-model/data"

#### LOAD DATA 
print("LOADING DATA")

from dataload import msw, rangelands, facilities, grazed_rates

# # read in data
# # rangeland polygons
# rangelands = gpd.read_file(opj(DATA_DIR, "raw/CA_FMMP_G/gl_bycounty/grazingland_county.shp"))
# rangelands = rangelands.to_crs(epsg=4326)
# rangelands['centroid'] = rangelands['geometry'].centroid 
# rl_lon, rl_lat = rangelands.centroid.x, rangelands.centroid.y
# print("rangelands loaded")

# # bring in clean crop data too in case want to include in plot
# croplands = cleancropdata(opj(DATA_DIR, 
# 	"raw/Crop__Mapping_2014-shp/Crop__Mapping_2014.shp"))



# # county polygons
# county_shape = gpd.read_file(opj(DATA_DIR, 
#         "counties/CA_Counties_TIGER2016.shp")) #  shape
# counties_popcen = pd.read_csv(opj(DATA_DIR, 
#         "counties/CenPop2010_Mean_CO06.txt")) # population weighted means!
# counties_popcen.rename(columns = {'LATITUDE': 'lat', 
#         'LONGITUDE': 'lon', 'COUNAME': 'COUNTY'}, inplace=True)

# county_shape = county_shape.to_crs(epsg=4326)
# county_shape.rename(columns = {'COUNTYFP': 'COUNTY'}, inplace=True)
# county_shape['county_centroid'] = county_shape['geometry'].centroid
# print("county shapes and pop cen loaded")


# # # solid waste inventory data (CLEANED)
# swis =  gpd.read_file(opj(DATA_DIR, "clean/clean_swis.shp"))


# # Minimize geodataframe to dataframe with just fields of interest
# swis_df = swis[['SwisNo', 'Name', 'Latitude', 'Longitude', 'cap_m3', 'AcceptedWa']]

# # rename lat and lon for easier plotting
# swis_df.rename(columns = {'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)

# # may just want foodwaste for adding to the plot
# foodwaste_facilities = swis_df[swis_df['AcceptedWa'].str.contains("Food", na=False)]
# print("swis loaded")

print("add region variable to swis")
# add region variable to swis and group
swis = facilities
swis.rename(columns = {'COUNTY': 'County'}, inplace=True)
swis_county_region = add_region_variable(swis, 'County')
swis_region = swis_county_region.groupby('Region', as_index = False).sum()

# doesn't have MOUNTAIN NORTH -- need to add manually
new_row = {'Region':'MountainNorth', 'Latitude' : 0, 'Longitude' : 0, 'Throughput' : 0, 
            'Capacity': 0, 'Acreage' : 0, 'cap_tons' : 0}

#append row to the dataframe
swis_region = swis_region.append(new_row, ignore_index=True)

# # add tons back in for plotting?
# swis_region['cap_tons'] = swis_region['cap_m3'] * (1.30795*(1/2.24))


# ##### Municipal Solid Waste (points) #####
msw.rename(columns = {'muni_ID': 'County'}, inplace=True)

print("add region variable to msw_county")
# adding region variable to msw_county for dot plot
msw_county_region = add_region_variable(msw, 'County')

print("start dot plot....")
#dot plot by region

msw_region = msw_county_region.groupby(['Region'], as_index=False).sum()
msw_region.sort_values('fg_wt', inplace=True)

# Regions
X = []
# food waste values
F = []
# swis capacity
S = []

for r in msw_region['Region']:
	X.append(r)
	f = Fetch(msw_region, 'Region', r, 'FOOD')
	s = Fetch(swis_region, 'Region', str(r), 'cap_tons')
	S.append(s)
	F.append(f)
	print(r, "--food waste -- ", f)

F = np.true_divide(F, 10**6)
S = np.true_divide(S, 10**6)

# green waste
G = []
for r in msw_region['Region']:
	g = Fetch(msw_region, 'Region', r, 'GREEN')
	G.append((g*0.95))
	print(r, "--green waste -- ", g)

G = np.true_divide(G, 10**6)

# to match colors from QGIS
# food waste: #1f78b4
fw_col = '#1f78b4'
# green waste: #33a02c
gw_col = '#33a02c'
# composters: #9b41d7
swis_col = '#9b41d7' 

#axes
# x = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0])

# 
# Draw plot
plt.rc('font', family='serif')
fig, ax = plt.subplots(figsize=(5,3), dpi= 300)
ax.hlines(y=X, xmin=-0.1, xmax=2.0, color='gray', alpha=0.7, linewidth=1, linestyles='dashdot')
ax.scatter(y=X, x=F, s=150, color=fw_col, alpha=0.7)
ax.scatter(y=X, x=G, s=150, color=gw_col, alpha=0.7)
ax.scatter(y=X, x=S, marker = 'D', color = swis_col, alpha = 0.9)

# Title, Label, Ticks and Ylim
ax.set_title('Regional Inventory & Capacity (Tons)', fontdict={'size':14})
ax.set_xlabel('Tons of Material')
ax.set_xlim(-0.1, 1.5)
ax.set_yticks(X)
# ax.set_xticks(x)
# plt.xticks(x)

# plt.xticks(np.arange())
# ax.set_yticklabels(X.title(), fontdict={'horizontalalignment': 'right'})
plt.subplots_adjust(left=0.32, bottom=0.18)
# plt.show()


# save figure
plt.savefig('plots/Figure1_inset', transparent=True)



