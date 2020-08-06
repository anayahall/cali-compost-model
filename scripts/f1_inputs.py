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
# swis =  gpd.read_file(opj(DATA_DIR, "clean/clean_swis.shp"))


# # Minimize geodataframe to dataframe with just fields of interest
# swis_df = swis[['SwisNo', 'Name', 'Latitude', 'Longitude', 'cap_m3', 'AcceptedWa']]

# # rename lat and lon for easier plotting
# swis_df.rename(columns = {'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)

# # may just want foodwaste for adding to the plot
# foodwaste_facilities = swis_df[swis_df['AcceptedWa'].str.contains("Food", na=False)]
# print("swis loaded")






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

###########Example plot

# schools = ["Brown", "NYU", "Notre Dame", "Cornell", "Tufts", "Yale",
#            "Dartmouth", "Chicago", "Columbia", "Duke", "Georgetown",
#            "Princeton", "U.Penn", "Stanford", "MIT", "Harvard"]

# fig = plt.subplots()
# plt.scatter(
#     x=[72, 67, 73, 80, 76, 79, 84, 78, 86, 93, 94, 90, 92, 96, 94, 112],
#     y=schools,
#     marker=dict(color="crimson", size=12),
#     mode="markers",
#     name="Women",
# )

# plt.scatter(
#     x=[92, 94, 100, 107, 112, 114, 114, 118, 119, 124, 131, 137, 141, 151, 152, 165],
#     y=schools,
#     marker=dict(color="gold", size=12),
#     mode="markers",
#     name="Men",
# )

# plt.update_layout(title="Gender Earnings Disparity",
#                   xaxis_title="Annual Salary (in thousands)",
#                   yaxis_title="School")

# plt.show()

# plot.show()

