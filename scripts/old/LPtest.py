
# coding: utf-8

# In[11]:

"""
MINI TEST OF LINEAR PROGRAMMING PULP MODEL
Mini-dicts of counties and facilities,
currently based on 'beer distribution model'
>> need to rewrite CORRECTLY
"""

import pandas as pd
import os
import numpy as np
import shapely as shp
import geopandas as gpd

import pulp
import scipy as sp
from scipy.spatial import distance_matrix

# import plotly.plotly as py
# import plotly.graph_objs as go
# from matplotlib import rc

os.chdir("/Users/anayahall/projects/grapevine")


# In[21]:

############################################################
# FUNCTIONS USED IN THIS SCRIPT

# define function to convert into proj in meters 
# (use epsg=26911 for statewide California, can change for others)
# alt is 3857 or 3310
def epsg_meters(gdf, proj=3310):
    g = gdf.copy()
    g = g.to_crs(epsg=proj)
    return g

# define function to get dictionary names for LP
def get_dict_names(dict):
    names = []
    for key, value in dict.items():
    #     print(key)
        names.append(key)
    return names

# function to make array of coordinates to calculate matrices
def geo_to_coords(df):
    df['coord'] = ""
    for index, row in df.iterrows():
    #      print('index', index)
    #      print('row', row)
         for pt in list(row['geometry'].coords): 
    #         print(pt)
            df.at[index,'coord'] = np.asarray(pt)


# In[26]:

# mini gdfs of county wastes (tbm - location and MSW for 2014) 
c_proj = gpd.read_file("data/clean/techbiomass_pts.shp")

#CONVERT TO METERS!!
c = epsg_meters(c_proj)

#c = c[(c['biomass.ca'] == "organic fraction municipal solid waste") & (c['year'] == 2014)].copy()
c = c[(c['biomass.fe'] == "FOOD") & (c['year'] == 2014)].copy()

c = c[['FIPS', 'COUNTY', 'disposal.y', 'geometry']]
# subset out four counties
csub = c[(c['COUNTY'] == "Los Angeles") | (c['COUNTY'] == "San Diego")| 
         (c['COUNTY'] == "Orange")| (c['COUNTY'] == "Imperial")].copy()
# csub = c[(c['COUNTY'] == "San Diego")| (c['COUNTY'] == "Imperial")].copy()
# csub.head()


# In[14]:

####MAKE DICTIONARY HERE
cdict = dict(zip(csub['COUNTY'], csub['disposal.y']))

# grab dict names for LP solver
conames = get_dict_names(cdict)


# In[24]:

# Mini gdfs of facilites (location and capacity)
f_proj = gpd.read_file("data/clean/clean_swis.shp")
# f.head(8)

f = epsg_meters(f_proj)

f = f[['SwisNo', 'AcceptedWa', 'County', 'cap_m3', 'geometry']].copy()

# subset out four counties
fsub = f[(f['County'] == "Los Angeles") | (f['County'] == "San Diego") | 
          (f['County'] == "Orange")| (f['County'] == "Imperial")].copy()
# too many, just select first 5
fsub = fsub[0:5].copy()

# make into dictionary for use in LP solver
fdict = dict(zip(fsub['SwisNo'], fsub['cap_m3'])) 

# grab dict names for LP solver
facnames = get_dict_names(fdict)


# In[16]:

# CREATE DISTANCE MATRIX #################################

# RUN ON subset gdfs
geo_to_coords(csub)
geo_to_coords(fsub)

# Make coords into list for cost-distance matrix
C = list(csub.coord)
F = list(fsub.coord)

# print("type C: ", type(C))

# Test distance_matrix function as dataframe
# test1 = pd.DataFrame(distance_matrix(C, F), index = csub.COUNTY, columns = fsub.SwisNo)
# Actually make it as LIST for use in LP
cost_distance = list(distance_matrix(C,F))

# Multiply each calculated distance (m) by 1.4 for DETOUR 
cost_distance = [d * 1.4 for d in cost_distance]

# cost_distance


# In[17]:

# CHECK distances to make sure they're in the right place - this test STILL IN DEGREES!
# import math
# C1 = [-118.57205925,   34.14803912]
# F3 = [-117.1805,   32.8622]
# distance = math.sqrt( ((C1[0]-F3[0])**2)+((C1[1]-F3[1])**2) )
# print("distance between cty 1(LA) and fac 3(37AB): ", distance)
# Checks out! 


# In[18]:

#FIRST RUN TEST - BASED ON BEER DISTRIBUTION EXAMPLE
# Import PuLP modeler functions
from pulp import *

# Creates a list of all the supply nodes
Counties  = conames

# Creates a dictionary for the number of units of supply for each supply node
waste = cdict

# Creates a list of all demand nodes
Facilities = facnames

# Creates a dictionary for the number of units of demand for each demand node
compost = fdict

# Creates a list of costs of each transportation path
costs = cost_distance

# The cost data is made into a dictionary
costs = makeDict([Counties, Facilities],costs,0)

emfac = 1.8

# Creates the 'prob' variable to contain the problem data
prob = LpProblem("Compost Distribution Problem",LpMaximize)

# Creates a list of tuples containing all the possible routes for transport
Routes = [(c,f) for c in Counties for f in Facilities]

# A dictionary called 'Vars' is created to contain the referenced variables(the routes)
vars = LpVariable.dicts("Route",(Counties,Facilities),0,None,LpInteger)

# The objective function is added to 'prob' first
prob += lpSum([vars[c][f]*costs[c][f] for (c,f) in Routes]) + lpSum([vars[c][f]*emfac for (c,f) in Routes]), "Sum_of_Transporting_Costs"

# # The supply maximum constraints are added to prob for each supply node (warehouse)
for c in Counties:
    prob += lpSum([vars[c][f] for f in Facilities]) <= waste[c], "Sum_of_waste_out_of_Counties_%s"%c

# The demand minimum constraints are added to prob for each demand node (bar)
for f in Facilities:
    prob += lpSum([vars[c][f] for c in Counties])<=compost[f], "Sum_of_compost_into_Facilities_%s"%f

# vars 
# The problem data is written to an .lp file
prob.writeLP("CompostDistributionProblem.lp")

# The problem is solved using PuLP's choice of Solver
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print(v.name, "=", v.varValue)

# The optimised objective function value is printed to the screen    
print("Total Cost of Transportation = ", value(prob.objective))


# In[19]:

print(value(prob.objective)/10000000)


# In[ ]:




# In[20]:

# """
# The American Steel Problem for the PuLP Modeller
# Authors: Antony Phillips, Dr Stuart Mitchell  2007
# """

# # Import PuLP modeller functions
# from pulp import *

# # List of all the nodes
# Nodes = ["Youngstown",
#          "Pittsburgh",
#          "Cincinatti",
#          "Kansas City",
#          "Chicago",
#          "Albany",
#          "Houston",
#          "Tempe",
#          "Gary"]

# nodeData = {# NODE        Supply Demand
#          "Youngstown":    [10000,0],
#          "Pittsburgh":    [15000,0],
#          "Cincinatti":    [0,0],
#          "Kansas City":   [0,0],
#          "Chicago":       [0,0],
#          "Albany":        [0,3000],
#          "Houston":       [0,7000],
#          "Tempe":         [0,4000],
#          "Gary":          [0,6000]}

# # List of all the arcs
# Arcs = [("Youngstown","Albany"),
#         ("Youngstown","Cincinatti"),
#         ("Youngstown","Kansas City"),
#         ("Youngstown","Chicago"),
#         ("Pittsburgh","Cincinatti"),
#         ("Pittsburgh","Kansas City"),
#         ("Pittsburgh","Chicago"),
#         ("Pittsburgh","Gary"),
#         ("Cincinatti","Albany"),
#         ("Cincinatti","Houston"),
#         ("Kansas City","Houston"),
#         ("Kansas City","Tempe"),
#         ("Chicago","Tempe"),
#         ("Chicago","Gary")]

# arcData = { #      ARC                Cost Min Max
#         ("Youngstown","Albany"):      [0.5,0,1000],
#         ("Youngstown","Cincinatti"):  [0.35,0,3000],
#         ("Youngstown","Kansas City"): [0.45,1000,5000],
#         ("Youngstown","Chicago"):     [0.375,0,5000],
#         ("Pittsburgh","Cincinatti"):  [0.35,0,2000],
#         ("Pittsburgh","Kansas City"): [0.45,2000,3000],
#         ("Pittsburgh","Chicago"):     [0.4,0,4000],
#         ("Pittsburgh","Gary"):        [0.45,0,2000],
#         ("Cincinatti","Albany"):      [0.35,1000,5000],
#         ("Cincinatti","Houston"):     [0.55,0,6000],
#         ("Kansas City","Houston"):    [0.375,0,4000],
#         ("Kansas City","Tempe"):      [0.65,0,4000],
#         ("Chicago","Tempe"):          [0.6,0,2000],
#         ("Chicago","Gary"):           [0.12,0,4000]}

# # Splits the dictionaries to be more understandable
# (supply, demand) = splitDict(nodeData)
# (costs, mins, maxs) = splitDict(arcData)

# # Creates the boundless Variables as Integers
# vars = LpVariable.dicts("Route",Arcs,None,None,LpInteger)

# # Creates the upper and lower bounds on the variables
# for a in Arcs:
#     vars[a].bounds(mins[a], maxs[a])

# # Creates the 'prob' variable to contain the problem data    
# prob = LpProblem("American Steel Problem",LpMinimize)

# # Creates the objective function
# prob += lpSum([vars[a]* costs[a] for a in Arcs]), "Total Cost of Transport"

# # Creates all problem constraints - this ensures the amount going into each node is at least equal to the amount leaving
# for n in Nodes:
#     prob += (supply[n]+ lpSum([vars[(i,j)] for (i,j) in Arcs if j == n]) >=
#              demand[n]+ lpSum([vars[(i,j)] for (i,j) in Arcs if i == n])), "Steel Flow Conservation in Node %s"%n

# # The problem data is written to an .lp file
# prob.writeLP("AmericanSteelProblem.lp")

# # The problem is solved using PuLP's choice of Solver
# prob.solve()

# # The status of the solution is printed to the screen
# print("Status:", LpStatus[prob.status])

# # Each of the variables is printed with it's resolved optimum value
# for v in prob.variables():
#     print(v.name, "=", v.varValue)

# # The optimised objective function value is printed to the screen    
# print("Total Cost of Transportation = ", value(prob.objective))


# In[ ]:




# In[ ]:



