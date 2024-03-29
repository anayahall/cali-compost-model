## flowfigure.py

# map flow of material from urban centers to facilities and then facilities to rangelands
# import model output for a few scenarios
# plot!

# import modules
import pandas as pd
import numpy as np
import shapely as shp
import geopandas as gpd
from os.path import join as opj
import matplotlib.pyplot as plt
import os
import pickle

# suppress warnings in jupyter notebook!
import warnings
warnings.simplefilter('ignore')

def Fetch(df, key_col, key, value):
	#counties['disposal'].loc[counties['COUNTY']=='San Diego'].values[0]
	return df[value].loc[df[key_col]==key].values[0]

# set data path
DATA_DIR = "/Users/anayahall/projects_v2/cali-compost-model/data"

print("LOADING DATA")
#load data from main file
from dataload import msw, rangelands, facilities, grazed_rates

#rename swis facilities for use here
swis = facilities
# Minimize geodataframe to dataframe with just fields of interest
swis_df = swis[['SwisNo', 'Name', 'Latitude', 'Longitude', 'cap_m3', 'AcceptedWa']]
# rename lat and lon for easier plotting
swis_df.rename(columns = {'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)

# load county polygon shapes and rename columns for use here
county_shape = gpd.read_file(opj(DATA_DIR, 
		"counties/CA_Counties_TIGER2016.shp")) # OLD- raw shape
counties_popcen = pd.read_csv(opj(DATA_DIR, 
		"counties/CenPop2010_Mean_CO06.txt")) # NEW - population weighted means!
counties_popcen.rename(columns = {'LATITUDE': 'lat', 
		'LONGITUDE': 'lon', 'COUNAME': 'COUNTY'}, inplace=True)
county_shape = county_shape.to_crs(epsg=4326)
county_shape['county_centroid'] = county_shape['geometry'].centroid

#add lat lon to rangelands
rl_lon, rl_lat = rangelands.centroid.x, rangelands.centroid.y


#####  - generate random data--- need to load real matrix of quantities here!!!
# testmatrix = np.random.randint(500000, size=(58, 109))
# c2f_dict = testmatrix

# with open('out/c2f_a1.p', 'wb') as f:
# 	pickle.dump(c2f, f)
# with open('out/f24_a1.p', 'wb') as f:
# 	pickle.dump(f2r, f)


## LOAD DATA FROM ALPHA=0.08 RUN 
# will be a dictionary 

# Couty to Facility
with open('out/c2f_a75.p', 'rb') as f:
	c2f_quant = pickle.load(f) 
c2f_dict = c2f_quant

dictlist = []
for k, v in c2f_dict.items():
	temp = v
	for l,m in temp.items():
		if m > 11:
			dictlist.append(m)
np.quantile(dictlist, [.1, .25, .5 , .75, .95])
# array([  1674.8,  13181. ,  39980. ,  94910. , 25683.8])

# Facilty to Rangeland
with open('out/f2r_a75.p', 'rb') as f:
	f2r_quant = pickle.load(f) 
f2r_dict = f2r_quant

dictlist = []
for k, v in f2r_dict.items():
	temp = v
	for l,m in temp.items():
		if m > 10:
			dictlist.append(m)
# array([  1155.55,  1155.25,  34252.  ,  76149.5 , 212371.2 ])



# raise Exception("data loaded ; pre-plot")

print("Starting Plot")
# plt.rc('font', family='serif')

# PLOT
fig, ax = plt.subplots(figsize = (10,10))

# FIRST COUNTIES
county_shape.plot(ax = ax, color = "silver", alpha = 0.4, linewidth=1, edgecolor = "white")

# PLOT RANGELANDS
rangelands.plot(ax = ax, color = 'white', alpha = 0.8, label = 'Rangeland')

# composters
ax.scatter(swis_df['lon'], swis_df['lat'], s = swis_df['cap_m3']/2000, 
	color = 'dimgray', marker='D')

# label = 'Compost Facility', 

# PLOT LINES FROM COUNTY TO FACILITY
for i in counties_popcen.index:
	county_name = counties_popcen['COUNTY'].iloc[counties_popcen.index == i].values[0]
	c_lon = counties_popcen['lon'].iloc[counties_popcen.index == i].values[0]
	c_lat = counties_popcen['lat'].iloc[counties_popcen.index == i].values[0]
	_ = ax.plot(c_lon, c_lat, c= 'gray', marker='o', markersize = 5)
	# print("C: ", c_lon, c_lat)
	# print('*************')
	# print(county_name)
	for j in swis_df.index:
		f_no = swis_df['SwisNo'].iloc[swis_df.index == j].values[0]
		f_lon = swis_df['lon'].loc[swis_df.index == j].values[0]
		f_lat = swis_df['lat'].loc[swis_df.index == j].values[0]
		# print("F: ",f_lon, f_lat)
		# THIS IS HOW REAL DATA WILL BE FORMATTED - AS DICT
		q = c2f_dict[county_name][f_no]*.58
		# print(q)
		# q = c2f_dict.loc[c2f_dict.index == f_no, county_name].values[0]
		# q = c2f_dict[i, j] # USE THIS FOR TEST MATRIX ONLY
		# _ = ax.arrow(x=c_lon, y=c_lat, dx= (f_lon-c_lon), dy= (f_lat - c_lat), length_includes_head=True,
  #         head_width=0.08, head_length=0.002, alpha = 0.6)
		if q > 500000:
			_ = ax.arrow(x=c_lon, y=c_lat, dx= (f_lon-c_lon), dy= (f_lat - c_lat), color='c', length_includes_head=True,
          head_width=0.08, head_length=0.08, alpha = 0.6, linewidth=5.5)
		elif q > 50000:
			_ = ax.arrow(x=c_lon, y=c_lat, dx= (f_lon-c_lon), dy= (f_lat - c_lat), color='c', length_includes_head=True,
          head_width=0.08, head_length=0.08, alpha = 0.6, linewidth=3.5)
		elif q > 25000:
			_ = ax.arrow(x=c_lon, y=c_lat, dx= (f_lon-c_lon), dy= (f_lat - c_lat), color='c', length_includes_head=True,
          head_width=0.08, head_length=0.08, alpha = 0.6, linewidth=2.5)
		elif q > 5000:
			_ = ax.arrow(x=c_lon, y=c_lat, dx= (f_lon-c_lon), dy= (f_lat - c_lat), color='c', length_includes_head=True,
          head_width=0.08, head_length=0.08, alpha = 0.6, linewidth=1.5)
		elif q > 1000: 
			_ = ax.arrow(x=c_lon, y=c_lat, dx= (f_lon-c_lon), dy= (f_lat - c_lat), color='c', length_includes_head=True,
          head_width=0.08, head_length=0.08, alpha = 0.6, linewidth=0.75)
		elif q > 200: 
			_ = ax.arrow(x=c_lon, y=c_lat, dx= (f_lon-c_lon), dy= (f_lat - c_lat), color='c', length_includes_head=True,
          head_width=0.08, head_length=0.08, alpha = 0.6, linewidth=0.08)

# PLOT COMPOST --- FACILITY TO RANGELANDS
for j in swis_df.index:
	f_no = swis_df['SwisNo'].iloc[swis_df.index == j].values[0]
	f_lon = swis_df['lon'].iloc[swis_df.index == j].values[0]
	f_lat = swis_df['lat'].iloc[swis_df.index == j].values[0]
	# print('*************')
	# print(f_no)
	for r in rangelands.index:
		r_no = rangelands['OBJECTID'].iloc[rangelands.index == r].values[0]
		r_lon = rl_lon.loc[rl_lon.index == r].values[0]
		r_lat = rl_lat.loc[rl_lat.index == r].values[0]        # THIS IS HOW REAL DATA WILL BE FORMATTED - AS DICT
		if r_no in f2r_dict[f_no].keys():
			q = f2r_dict[f_no][r_no]*0.58
			# print(q)
		# q = c2fmatrix.loc[c2fmatrix.index == f_no, county_name].values[0]
		# q = c2fmatrix[i, j] # USE THIS FOR TEST MATRIX ONLY
			if q > 50000:
				_ = ax.arrow(x=f_lon, y=f_lat, dx= (r_lon-f_lon), dy= (r_lat - f_lat), color='m', 
					length_includes_head=True, head_width=0.08, head_length=0.08, alpha = 0.6, linewidth=4.5)
			elif q > 25000:
				_ = ax.arrow(x=f_lon, y=f_lat, dx= (r_lon-f_lon), dy= (r_lat - f_lat), color='m', 
					length_includes_head=True, head_width=0.08, head_length=0.08, alpha = 0.6, linewidth=3.5)
			elif q > 10000:
				_ = ax.arrow(x=f_lon, y=f_lat, dx= (r_lon-f_lon), dy= (r_lat - f_lat), color='m', 
					length_includes_head=True, head_width=0.08, head_length=0.08,  alpha = 0.6, linewidth=2.5)
			elif q > 5000:
				_ = ax.arrow(x=f_lon, y=f_lat, dx= (r_lon-f_lon), dy= (r_lat - f_lat), color='m', 
					length_includes_head=True, head_width=0.08, head_length=0.08,  alpha = 0.6, linewidth=1.5)
			elif q > 1000: 
				_ = ax.arrow(x=f_lon, y=f_lat, dx= (r_lon-f_lon), dy= (r_lat - f_lat), color='m', 
					length_includes_head=True, head_width=0.08, head_length=0.08,  alpha = 0.6, linewidth=0.75)
			elif q > 100: 
				_ = ax.arrow(x=f_lon, y=f_lat, dx= (r_lon-f_lon), dy= (r_lat - f_lat), color='m', 
					length_includes_head=True, head_width=0.08, head_length=0.08, alpha = 0.6, linewidth=0.08)


# LEGEND COUNTY CENTROID
ax.plot([], [], 'o', color = 'gray', markersize = 5, label = 'County Centroid')
ax.plot([], [], 'D', color = 'dimgrey', markersize = 5, label = 'Compost Facility')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
# ax[0].set_title("Feedstock Flow from County Centroid to Facility")

ax.axis('off')


## FIX__ MAKE SURE MATCH ABOVE!!

# LEGEND COMPOST Flow by Quantity
ax.plot([], [], 'm:',  label = 'Compost Flow (tons)')
ax.plot([], [], 'm-', alpha = 0.6, linewidth=0.08, label = '0 - 2000')
ax.plot([], [], 'm-', alpha = 0.6, linewidth=0.75, label = '2000 - 20000')
ax.plot([], [], 'm-', alpha = 0.6, linewidth=1.5, label = '20000 - 50000')
ax.plot([], [], 'm-', alpha = 0.6, linewidth=2.5, label = '20000 - 50000')
ax.plot([], [], 'm-', alpha = 0.6, linewidth=3.5, label = '50000 - 200000')
ax.plot([], [], 'm-', alpha = 0.6, linewidth=5.5, label = '200000 - 1000000')

# LEGEND FLOW - by size
ax.plot([],[], 'c:', label = "Feedstock Flow (tons)")
ax.plot([], [], 'c-', alpha = 0.6, linewidth=0.08, label = '0 - 2000')
ax.plot([], [], 'c-', alpha = 0.6, linewidth=0.75, label = '2000 - 10000')
ax.plot([], [], 'c-', alpha = 0.6, linewidth=1.5, label = '10000 - 25000')
ax.plot([], [], 'c-', alpha = 0.6, linewidth=2.5, label = '25000 - 60000')
ax.plot([], [], 'c-', alpha = 0.6, linewidth=3.5, label = '60000 - 100000')
ax.plot([], [], 'c-', alpha = 0.6, linewidth=4.5, label = '100000+ ')

ax.legend()

print("Figure done plotting")
fig.savefig('Maps/flowfigure_v2.png')       
# plt.show()

#################################################################################



