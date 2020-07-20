

# Test plotting with basemap
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

from biomass_preprocessing import MergeInventoryAndCounty

DATA_DIR = "/Users/anayahall/projects/compopt/data"
OUT_DIR = "/Users/anayahall/projects/compopt/maps"

CA = gpd.read_file(opj(DATA_DIR, "raw/CA_Counties/CA_Counties_TIGER2016.shp"))
CA = CA.to_crs(epsg=4326)
CA['COUNTY'] = CA['NAME']
# print("Ca_proj crs: ", CA_proj.crs)
CA.head()

# LOAD BIOMASS DATA
gbm_pts, tbm_pts = MergeInventoryAndCounty(
    gross_inventory     = opj(DATA_DIR, "raw/biomass.inventory.csv"),
    technical_inventory = opj(DATA_DIR, "raw/biomass.inventory.technical.csv"),
    county_shapefile    = opj(DATA_DIR, "raw/CA_Counties/CA_Counties_TIGER2016.shp"),
    fips_data           = opj(DATA_DIR, "interim/CA_FIPS.csv")
)
counties = tbm_pts[tbm_pts['year'] == 2014] # could change to GBM

ej = gpd.read_file(opj(DATA_DIR, "calenviroscreen/CESJune2018Update_SHP/CES3June2018Update.shp"))


rangelands = gpd.read_file(opj(DATA_DIR, "raw/CA_FMMP_G/gl_bycounty/grazingland_county.shp"))
# rangelands = gpd.read_file(opj(DATA_DIR, "raw/CA_FMMP_G/grazingland_dis/CA_grazingland.shp"))
rangelands = rangelands.to_crs(epsg=4326)

swis =  gpd.read_file(opj(DATA_DIR, "clean/clean_swis.shp"))




f, ax = plt.subplots()
CA.plot(ax = ax, color = "white", figsize = (10,10), linewidth=0.2, edgecolor = "grey")
rangelands.plot(ax= ax, color = "green", alpha = .5)
ax.axis('off')
ax.set_title('Rangelands', fontdict={'fontsize': '12', 'fontweight' : '3'})
plt.savefig(opj(OUT_DIR, "rangelands.png"), dpi=300)




f, ax = plt.subplots()
CA.plot(ax = ax, color = "white", figsize = (10,10), linewidth=0.1, edgecolor = "grey")
swis.plot(ax = ax, markersize = swis.cap_m3/3000, marker = 'o', color = 'black', alpha=.4, linewidth=0)
ax.axis('off')
ax.set_title('Active Composting Facilities in CA', fontdict={'fontsize': '12', 'fontweight' : '3'})
# plt.savefig("maps/FacilitiesbyCapacity.png", dpi=300)



plotvar = swis['cap_m3']*1.30795

# before plotting prep legend
c = []
for i in [10, 25, 50, 75]:
    c.append(int(round(np.percentile(plotvar, i), -3)))

# Map Capacity by County
f, ax = plt.subplots(1)
CA.plot(ax = ax, color = "white", figsize = (10,10), linewidth=0.2, edgecolor = "black")
# capmap.plot(ax = ax, column = plotvar, cmap = "Oranges", legend = True)
swis.set_geometry('geometry').plot(ax = ax, markersize = plotvar/5000, marker = 'o', 
                                  legend = True, color = 'black', alpha=.6, linewidth=0)
ax.axis('off')
ax.set_title('Active Composting Facilities  (Cubic Yards)', fontdict={'fontsize': '12', 'fontweight' : '3'})

l1 = plt.scatter([],[], s=c[0]/1000, edgecolors='none', color = "black")
l2 = plt.scatter([],[], s=c[1]/1000, edgecolors='none', color = "black")
l3 = plt.scatter([],[], s=c[2]/1000, edgecolors='none', color = "black")
l4 = plt.scatter([],[], s=c[3]/1000, edgecolors='none', color = "black")

labels = [str(c[0]), str(c[1]), str(c[2]), str(c[3])]
labels = ["5,000", "15,000", "50,000", "100,000"]

leg = plt.legend([l1, l2, l3, l4], labels, ncol = 1, frameon=False, fontsize=10,
handlelength=2, loc = 1, borderpad = 1,
handletextpad=1, title='Facility Size', scatterpoints = 1)
plt.savefig(opj(OUT_DIR, "FacilitiesbyCapacity.png"), dpi=300)



# four plots: municipal food waste, green waste, agricultural residue and manure



# Clean ag residue for plotting

counties.category.value_counts()

counties['AgResidue'] = False

# #make category of Ag Residue
n = len(counties.index)
for i,row in counties.iterrows():
#     print(counties.category[i])
    if counties.category[i]=="row residue": 
        counties.at[i, 'AgResidue']=True
    elif counties.category[i]=="row culls": 
        counties.at[i, 'AgResidue']=True
    elif counties.category[i]=="orchard vineyard residue": 
        counties.at[i, 'AgResidue']=True
    elif counties.category[i]=="orchard vineyard culls": 
#         counties.AgResidue[i]=True
        counties.at[i, 'AgResidue']=True
    elif counties.category[i]=="field residue": 
        counties.at[i, 'AgResidue']=True
    else:
        counties.at[i, 'AgResidue']=False

agres = counties[counties['AgResidue'] == True].groupby(['COUNTY'])['disposal_BDT'].sum()

agres = pd.DataFrame(agres)
agres['COUNTY'] = agres.index


CA_shp = CA[['COUNTY', 'geometry']]

agres = pd.merge(CA_shp, agres, on = 'COUNTY')
agres.head()
# counties.head()




# plotvar = agres['disposal_BDT']


# Map Capacity by County
f, ax = plt.subplots()
CA.plot(ax = ax, color = "white", figsize = (10,10), linewidth=0.2, edgecolor = "black")
agres.plot(ax = ax, column = 'disposal_BDT', cmap = "Greens", legend = True)
ax.axis('off')
ax.set_title('Agricultural Residue (Annual BDT)', fontdict={'fontsize': '12', 'fontweight' : '3'})
plt.savefig(opj(OUT_DIR, "AgResidue.png"), dpi=300)




# manure dataset for plotting
manure = counties[counties['category'] == "manure"]
manure = manure[['COUNTY', 'disposal_BDT', 'wettons']]
manure = pd.merge(CA_shp, manure, on = 'COUNTY')

# manure.head()




f, ax = plt.subplots()
CA.plot(ax = ax, color = "white", figsize = (10,10), linewidth=0.2, edgecolor = "black")
manure.plot(ax = ax, column = 'disposal_BDT', cmap = "Blues", legend = True)
ax.axis('off')
ax.set_title('Manure (Annual BDT)', fontdict={'fontsize': '12', 'fontweight' : '3'})
plt.savefig(opj(OUT_DIR, "Manure.png"), dpi=300)




# food and green waste
ofmsw = counties[counties['category'] == "organic fraction municipal solid waste"]

foodwaste = ofmsw[ofmsw['feedstock'] == "FOOD"]
greenwaste = ofmsw[ofmsw['feedstock'] == "GREEN"]

foodwaste = foodwaste[['COUNTY', 'disposal_BDT', 'wettons', 'county_centroid']]
greenwaste = greenwaste[['COUNTY', 'disposal_BDT', 'wettons', 'county_centroid']]


foodwaste.head()




# before plotting prep legend
plotvar = foodwaste['disposal_BDT']

c = []
for i in [10, 45, 65, 85]:
    c.append(int(round(np.percentile(plotvar, i), -3)))

f, ax = plt.subplots()
CA.plot(ax = ax, color = "white", figsize = (10,10), linewidth=0.2, edgecolor = "black")
foodwaste.plot(ax = ax, markersize = foodwaste['disposal_BDT']/500, marker = 'o', 
               color = "Orange", alpha = 0.5, legend = True)
ax.axis('off')
ax.set_title('Food Waste (Annual BDT)', fontdict={'fontsize': '12', 'fontweight' : '3'})


l1 = plt.scatter([],[], s=c[0]/100, edgecolors='none', color = "black")
l2 = plt.scatter([],[], s=c[1]/100, edgecolors='none', color = "black")
l3 = plt.scatter([],[], s=c[2]/100, edgecolors='none', color = "black")
l4 = plt.scatter([],[], s=c[3]/100, edgecolors='none', color = "black")

# labels = [str(c[0]), str(c[1]), str(c[2]), str(c[3])]
labels = ["0", "5,000", "10,000", "30,000"]

leg = plt.legend([l1, l2, l3, l4], labels, ncol = 1, frameon=False, fontsize=10,
handlelength=2, loc = 1, borderpad = 1,
handletextpad=1, title='', scatterpoints = 1)

plt.savefig(opj(OUT_DIR, "FoodWaste.png"), dpi=300)





# before plotting prep legend
plotvar = greenwaste['disposal_BDT']

c = []
for i in [10, 45, 65, 85]:
    c.append(int(round(np.percentile(plotvar, i), -3)))


f, ax = plt.subplots()
CA.plot(ax = ax, color = "white", figsize = (10,10), linewidth=0.2, edgecolor = "black")
greenwaste.plot(ax = ax, markersize = plotvar/500, marker = 'o', 
               color = "Green", alpha = 0.5, legend = True)
ax.axis('off')
ax.set_title('Green Waste (Annual BDT)', fontdict={'fontsize': '12', 'fontweight' : '3'})


###########

l1 = plt.scatter([],[], s=c[0]/200, edgecolors='none', color = "black")
l2 = plt.scatter([],[], s=c[1]/200, edgecolors='none', color = "black")
l3 = plt.scatter([],[], s=c[2]/200, edgecolors='none', color = "black")
l4 = plt.scatter([],[], s=c[3]/200, edgecolors='none', color = "black")

# labels = [str(c[0]), str(c[1]), str(c[2]), str(c[3])]
labels = ["1,000", "10,000", "20,000", "65,000"]

leg = plt.legend([l1, l2, l3, l4], labels, ncol = 1, frameon=False, fontsize=10,
handlelength=2, loc = 1, borderpad = 1,
handletextpad=1, title='', scatterpoints = 1)

plt.savefig(opj(OUT_DIR, "GreenWaste.png"), dpi=300)



