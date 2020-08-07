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
DATA_DIR = "/Users/anayahall/projects/cali-compost-model/data"

#### LOAD DATA 
print("LOADING DATA")

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




# # solid waste inventory data (CLEANED)
swis =  gpd.read_file(opj(DATA_DIR, "clean/clean_swis.shp"))


# Minimize geodataframe to dataframe with just fields of interest
swis_df = swis[['SwisNo', 'Name', 'Latitude', 'Longitude', 'cap_m3', 'AcceptedWa']]

# rename lat and lon for easier plotting
swis_df.rename(columns = {'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)

# may just want foodwaste for adding to the plot
foodwaste_facilities = swis_df[swis_df['AcceptedWa'].str.contains("Food", na=False)]
print("swis loaded")

print("add region variable to swis")
# add region variable to swis and group
swis_county_region = add_region_variable(swis, 'County')
swis_region = swis_county_region.groupby('Region', as_index = False).sum()

# doesn't have MOUNTAIN NORTH -- need to add manually
new_row = {'Region':'MountainNorth', 'Latitude' : 0, 'Longitude' : 0, 'Throughput' : 0, 
            'Capacity': 0, 'Acreage' : 0, 'cap_m3' : 0}
#append row to the dataframe
swis_region = swis_region.append(new_row, ignore_index=True)

# add tons back in for plotting?
swis_region['cap_tons'] = swis_region['cap_m3'] * (1.30795*(1/2.24))






##### Municipal Solid Waste (points) #####
print("about to load MSW points") 
msw_shapefile = "msw_2020/msw_2020.shp"

msw = gpd.read_file(opj(DATA_DIR,
				  msw_shapefile))

# filter to just keep food and green waste (subject of regulations)
msw = msw[(msw['subtype'] == "MSWfd_wet_dryad_wetad") | (msw['subtype'] == "MSWgn_dry_dryad")]

# MSW DATA NOTES: 
# fog = Fats, Oils, Grease; lb = lumber; cd = cardboard; fd = food;
# pp = paper, gn = green; ot = Other ; suffix describes what the 
# waste is deemed suitable for

# rename categories to be more intuitive
msw['subtype'].replace({'MSWfd_wet_dryad_wetad': 'MSW_food', 
						'MSWgn_dry_dryad': 'MSW_green'}, inplace = True)


### Create GROUPED County Data
grouped = msw.groupby(['County', 'subtype'], as_index=False).sum()

# NEW COUNTY SHAPE
counties = pd.read_csv(opj(DATA_DIR, 
			"counties/CenPop2010_Mean_CO06.txt")) # NEW - population weighted means!
# rename lat and lon for easier plotting
counties.rename(columns = {'LATITUDE': 'lat', 'LONGITUDE': 'lon', 'COUNAME': 'County'}, inplace=True)

#merge and turn back into shapefile?
# join counties and grouped
df = grouped.merge(counties, on='County')

# back to gdf
msw_county = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
# gdf.rename(columns = {'County' : 'muni_ID'}, inplace=True)



# output county msw for plotting in QGIS
# print("creating new msw county shape")
# gdf.to_file(driver='ESRI Shapefile', filename="msw_counties")

print("add region variable to msw_county")
# adding region variable to msw_county for dot plot
msw_county_region = add_region_variable(df, 'County')

print("start dot plot....")
#dot plot by region

msw_region = msw_county_region.groupby(['Region', 'subtype'], as_index=False).sum()
msw_region.sort_values('wt', inplace=True)

# Regions
X = []
# food waste values
F = []
# swis capacity
S = []

for r in msw_region[msw_region['subtype']=='MSW_food']['Region']:
    X.append(r)
    f = Fetch(msw_region[msw_region['subtype']=='MSW_food'], 'Region', r, 'wt')
    s = Fetch(swis_region, 'Region', str(r), 'cap_tons')
    S.append(s)
    F.append(f)
    print(r, "--food waste -- ", f)

# green waste
G = []
for r in msw_region[msw_region['subtype']=='MSW_green']['Region']:
    g = Fetch(msw_region[msw_region['subtype']=='MSW_green'], 'Region', r, 'wt')
    G.append((g*0.95))
    print(r, "--green waste -- ", g)

# to match colors from QGIS
# food waste: #1f78b4
fw_col = '#1f78b4'
# green waste: #33a02c
gw_col = '#33a02c'
# composters: #9b41d7
swis_col = '#9b41d7' 


# Draw plot
fig, ax = plt.subplots(figsize=(6,5), dpi= 300)
ax.hlines(y=X, xmin=0, xmax=2750000, color='gray', alpha=0.7, linewidth=1, linestyles='dashdot')
ax.scatter(y=X, x=F, s=150, color=fw_col, alpha=0.7)
ax.scatter(y=X, x=G, s=150, color=gw_col, alpha=0.7)
ax.scatter(y=X, x=S, marker = 'x', color = swis_col, alpha = 0.9)

# Title, Label, Ticks and Ylim
# ax.set_title('Regional Biomass Inventory & Compost Capacity', fontdict={'size':22})
ax.set_xlabel('Tonnes of Organic Material')
ax.set_yticks(X)
ax.set_yticklabels(X.title(), fontdict={'horizontalalignment': 'right'})
plt.subplots_adjust(left=0.28)
# ax.set_xlim(10, 27)
# plt.show()


# save figure
plt.savefig('plots/Figure1_inset')



