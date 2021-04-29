# Script to clean capacity values of SWIS compost sites and make spatial

# GOAL IS TO CHANGE THIS TO TURN INTO A FUNCTION!

# First, load packages
import pandas as pd
import os
import shapely as sp
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
 

#change wd
# os.chdir("/Users/anayahall/projects/compopt")

#######################################################################
# Starting from INTERIM data (somewhat preprocessed in R - may come back to)
#######################################################################
# from R: 
# orginal data cleaned from excel: "data/swis_clean.csv"
# #filter to composting (may come back to this to select all other sites as well)
# comp_swis <- swis_clean %>% filter(str_detect(Activity, "Compost") | str_detect(Activity, "Chip") & OperationalStatus != "Closed")

# TODO: MOVE ALL ^^ TO PYTHON
# activities: compost processors and maybe transfers stations?


#read in compost facilities csv
df = pd.read_csv("data/interim/swis_compost.csv")

df.County.value_counts(dropna=False).head()


df['CapacityUnits'].value_counts(dropna=False)


# Identify and recode oddly labeled capacity units (those lacking time unit)
# first: Tons
df[df.CapacityUnits=="Tons"]

n = len(df.index)
for i in range(n):
#     print("index: ", i)
    if df.SwisNo[i] == "13-AA-0095": 
        df.at[i, 'CapacityUnits'] = "Tons/year"
    if df.SwisNo[i]=="49-AA-0422": 
        df.at[i, 'CapacityUnits'] = "Tons/year"
    if df.SwisNo[i]=="36-AA-0456": 
        df.at[i, 'CapacityUnits'] = "Tons/year"


#df[df.CapacityUnits=="Cubic Yards"]


n = len(df.index)
for i in range(n):
    #print("index: ", i)
    if df.SwisNo[i]=="12-AA-0113": 
        df.at[i, 'CapacityUnits']="Cu Yards/month"
    if df.SwisNo[i]=="44-AA-0013": 
        df.at[i, 'CapacityUnits']="Cu Yards/month"
    if df.SwisNo[i]=="28-AA-0037": 
        df.at[i, 'CapacityUnits']="Cu Yards/month"
    if df.SwisNo[i]=="37-AA-0992": 
         df.at[i, 'CapacityUnits']="Cu Yards/year"
    if df.SwisNo[i]=="37-AB-0011": 
         df.at[i, 'CapacityUnits']="Cu Yards/year"
    if df.SwisNo[i]=="43-AA-0015": 
         df.at[i, 'CapacityUnits']="Cu Yards/year"
    if df.SwisNo[i]=="11-AA-0039": 
         df.at[i, 'CapacityUnits']="Cu Yards/year" 
    if df.SwisNo[i]=="28-AA-0002": 
         df.at[i, 'CapacityUnits']="Cu Yards/year" 
    if df.SwisNo[i]=="19-AR-1226": 
         df.at[i, 'CapacityUnits']="Cu Yards/month" 
    if df.SwisNo[i]=="54-AA-0059": 
         df.at[i, 'CapacityUnits']="Cu Yards/year"
    if df.SwisNo[i]=="23-AA-0052": 
         df.at[i, 'CapacityUnits']="Cu Yards/year"            
            
        
df[df.CapacityUnits=="Cubic Yards"]


# first filter out all th
df = df[df['Capacity'].notnull()]

df.reset_index(inplace=True)

df['cap_m3'] = 0.0


# write function to convert all capacity units into cubic meters/month!

#how to assign values:
# df.at[i, 'CapacityUnits'] = "Tons/year"

print("CLEANING CAPACITY - CONVERT TO CUBIC METERS / YEAR")
n = len(df.index)
for i in range(n):
    #print("index: ", i)
    if df.CapacityUnits[i] == "Tons/year":
        # print("tons/year")
        # tons/year * cu yards/ton * cu meters/cu yards  
        #df.cap_m3[i] = df.Capacity[i] * 2.24 * 0.764555 
        df.at[i, 'cap_m3'] = df.Capacity[i] * 2.24 * 0.764555
        # print(df.cap_m3[i])
    elif df.CapacityUnits[i] == "Cu Yards/year":
        # print("cu yrds/year")
        # cu yards/year * cu meters/cu yards 
        df.at[i, 'cap_m3'] = df.Capacity[i] * 0.764555 
        # print(df.cap_m3[i])
    elif df.CapacityUnits[i] == "Cubic Yards":
        print("index: ", i ," - cu yrds --- NEED TO DISENTANGLE STILL!")
        # print(df.cap_m3[i])
    elif df.CapacityUnits[i] == "Tons":
        print("tons") #there should be none of these
        # print(df.cap_m3[i])
    elif df.CapacityUnits[i] == "Tons/day":
        # tons/day * cu yards/ton * cu meters/cu yards * days/year 
        df.at[i, 'cap_m3'] = df.Capacity[i] * 2.24 * 0.764555 * (365/1) 
        # print("tons/day")
        # print(df.cap_m3[i])
    elif df.CapacityUnits[i] == "Cu Yards/month":
        # cu yards/month * cu meters/cu yards * months/year
        df.at[i, 'cap_m3'] = df.Capacity[i] * 0.764555 * 12
        # print("cu yrds/month")
        # print(df.cap_m3[i])
    elif df.CapacityUnits[i] == "Tires/day":
        print("index: ", i ," - tires/day - delete? now set capacity at: ", df.cap_m3[i])
        df.at[i, 'cap_m3'] = 0.0
    else:
        print("none of the above")

# will also need a function to convert waste volume into compost volume

# df.columns
# m3 * yd3/m3 * tons/yd3
#cap_m3 * (1/0.764555) * (1/2.24)

sum(df.cap_m3) * (1/0.764555) * (1/2.24)


raise Exception("pre-geo")


# Use geopandas to make spatial
# from: https://geohackweek.github.io/vector/04-geopandas-intro/
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf = GeoDataFrame(df, geometry=geometry)

# check length to make sure it matches df
len(geometry)


# In[14]:

# gdf.plot(marker='*', color='green', markersize=50, figsize=(3, 3))


# In[15]:

df.head()


# In[16]:

gdf.crs = {'init' :'epsg:4326'}


# gdf.head()
print("exporting shapefile")
out = r"/Users/anayahall/projects/compopt/data/clean/clean_swis.shp"

# type(gdf)

gdf.to_file(driver='ESRI Shapefile', filename=out)

print("p SWIS PRE_PROCESSING DONE RUNNING")

# return gdf







