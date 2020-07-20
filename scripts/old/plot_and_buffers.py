
# coding: utf-8

# In[25]:

# making buffers ! - first use geometric, later base on transport distance?

# First, load packages
import pandas as pd
import os
import numpy as np
import shapely as sp

import matplotlib.pyplot as plt
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
# only for jupyter nb to show plots inline
# %matplotlib inline 


# Function for proj into common Statewide Projection (#26911)
def epsg_meters(gdf, proj=26911):
    g = gdf.copy()
    g = g.to_crs(epsg=proj)
    return g


# In[31]:

#change wd
os.chdir("/Users/anayahall/projects/grapevine")

gdf_proj =  gpd.read_file("data/clean/clean_swis.shp")
# load census tract shapefile
# CA = gpd.read_file("data/raw/tl_2018_06_tract/tl_2018_06_tract.shp")
# LOAD COUNTY SHAPEFILE 
CA_proj = gpd.read_file("data/raw/CA_Counties/CA_Counties_TIGER2016.shp")
print(CA_proj.crs)
CA_proj.head()


# In[32]:

CA = epsg_meters(CA_proj)

CA.head()


# try plotting by size of facility
# gdf.head()
# gdf.cap_m3
# gdf.plot(marker = 'o', markersize = gdf.cap_m3/1000)


# In[34]:

# print("swis gdf crs: ",gdf.crs)

gdf_proj.head()
# CA.crs
# re-project
# CA = CA.to_crs({'init': 'epsg:4326'})


gdf = epsg_meters(gdf_proj)

gdf.plot()
gdf.head()


# In[40]:

CA.plot()


# In[49]:

# add buffers - should be in meters now.....
gdf['buffers'] = gdf.buffer(15000)
# one degree is about 85km 
# buf = gdf.buffer(250)

gdf.set_geometry('buffers').plot()
# gdf.head()


# In[51]:

# # gdf.plot(s = 'buffers') 
f, ax = plt.subplots(1)
CA.plot(ax = ax, cmap='Set3', linewidth=0.1)
gdf.set_geometry('buffers').plot(ax = ax, color="blue", alpha="0.1")
gdf.set_geometry('geometry').plot(ax = ax, color="black", marker = '*', markersize= 1)
ax.axis('off')
ax.set_title('Composting Permits in CA', fontdict={'fontsize': '12', 'fontweight' : '3'})
plt.savefig("maps/CAwbuffers.png", dpi=300)


# In[52]:

# attempt to nest plotting
gdf.plot(ax=CA.plot(cmap='Set3', figsize=(10, 6)), marker='o', markersize=15)
# ax.axis('off')
# ax.set_title('Composting Permits in CA', fontdict={'fontsize': '12', 'fontweight' : '3'})
# plt.savefig("maps/map_export.png", dpi=300)


# In[55]:

# better plot, with title
f, ax = plt.subplots(1)
CA.plot(ax = ax, cmap='Set3', figsize = (10,6), linewidth=0.1)
gdf.plot(ax = ax, markersize = gdf.cap_m3/10000, marker = 'o', color = 'black', alpha=.7, linewidth=0)
ax.axis('off')
ax.set_title('Composting Permits in CA', fontdict={'fontsize': '12', 'fontweight' : '3'})
plt.savefig("maps/FacilitiesbyCapacity.png", dpi=300)


# In[ ]:



