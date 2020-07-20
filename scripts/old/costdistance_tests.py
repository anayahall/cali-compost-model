
# coding: utf-8

# In[4]:

# MINI TEST
import pandas as pd
import os
import numpy as np
import shapely as shp
import geopandas as gpd

import pulp
import scipy as sp
from geopy.distance import great_circle
from geopy.distance import geodesic
from shapely.geometry import Point

from scipy.spatial import distance_matrix

# import plotly.plotly as py
# import plotly.graph_objs as go
# from matplotlib import rc

os.chdir("/Users/anayahall/projects/grapevine")


# In[126]:

# mini gdfs of county wastes (tbm - location and MSW for 2014) 
c = gpd.read_file("data/clean/techbiomass_pts.shp")
#c = c[(c['biomass.ca'] == "organic fraction municipal solid waste") & (c['year'] == 2014)].copy()
c = c[(c['biomass.fe'] == "FOOD") & (c['year'] == 2014)].copy()

c = c[['FIPS', 'COUNTY', 'disposal.y', 'geometry']]
# subset out four counties
csub = c[(c['COUNTY'] == "Los Angeles") | (c['COUNTY'] == "San Diego")| 
         (c['COUNTY'] == "Orange")| (c['COUNTY'] == "Imperial")].copy()
# csub = c[(c['COUNTY'] == "San Diego")| (c['COUNTY'] == "Imperial")].copy()



# In[97]:

# Mini gdfs of facilites (location and capacity)
f = gpd.read_file("data/clean/clean_swis.shp")
f = f[['SwisNo', 'AcceptedWa', 'County', 'cap_m3', 'geometry']].copy()

# subset out four counties
fsub = f[(f['County'] == "Los Angeles") | (f['County'] == "San Diego") | 
          (f['County'] == "Orange")| (f['County'] == "Imperial")].copy()
# too many, just select first 5
fsub = fsub[0:4].copy()
fsub


# In[127]:

def geo_to_coords(df):
    df['coord'] = ""
    for index, row in df.iterrows():
    #      print('index', index)
    #      print('row', row)
         for pt in list(row['geometry'].coords): 
    #         print(pt)
            df.at[index,'coord'] = np.asarray(pt)


# In[128]:

geo_to_coords(csub)
geo_to_coords(fsub)


# In[129]:




# In[156]:

# t = [[0,0],[0,1]]

C = list(csub.coord)
F = list(fsub.coord)

print("type C: ", type(C))
# print("type t: ", type(t))

#  pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)
mycosts = pd.DataFrame(distance_matrix(C, F), index = csub.COUNTY, columns = fsub.SwisNo)
## HORRAY THIS WORKS!! TRYING AGAIN TO GET INTO DIFFERNT FORMAT.... list of lists 

# distance_matrix(C,F)

costs = [   #Fac
         #2 3 6 10 11
         [2,4,5,2,1],#Imp   Counties
         [3,1,3,2,3], #LA
         [6,1,8,2,5], #ORANGE
         [3,4,1,5,3] #SD
        ]

type(costs)


# In[160]:

test1 = list(distance_matrix(C,F))

print("type test1: ", type(test1))
test1


# In[161]:

# from sklearn.metrics.pairwise import paired_distances >> FAILED
# C = np.asarray(csub.coord)
# F = np.asarray(fsub.coord)
# paired_distances(C,F)


# In[ ]:




# In[ ]:

# calc euclidian distance between them --> get Dij 
# (multiply by 1.4 for 'real' distance, and again by '1.6' to proxy Lij)

# for i in range(len(csub.index)):
#     print(csub.geometry[i])
#     for j in range(len(fsub.index)):
#         print(j)

x = range(1,5)
y = range(6,10)

# arr = [[]]

# for i in : 
#     arr[]
D = csub.distance(fsub)
D

# arr == [[]]

# print('fsub', fsub)
# print(csub)

# print(fsub)
    
# {'col1': [1, 2], 'col2': [3, 4]}
test = {}

# for csubElem in enumerate(csub):
count = 0
for key, value in csub.items():    
    #print('*** KEY', key)
    #print('&&& VALUE', value)
    currentCount = 'i' + str(count)
    test[currentCount] = []
    count += 1
    for key1, value1 in fsub.items():
        point = csub.geometry
#         point1 = fsub.geometry
#         foo = point.distance(point1)
        #print("%% FOO")
        test[currentCount].append('point(s)')
        
# #     print(csub.geometry[i])
#     for j in enumerate(fsub):
#         print(i, j)
df = pd.DataFrame(data = test).T #this will transpose it
# df

