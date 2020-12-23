# script to create toy data for model
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon


print("starting toy data creation")
# simulating random points:

# number of events to simulate
N = 20

# # AREA to cast random numbers in 
# # 10 degrees from -125
# lon = np.random.random(N) * 10 - 125
# # 10 degrees from 32
# lat = np.random.random(N) * 10 + 32

# munis = MultiPoint(np.vstack((lon, lat)).T)


# df = pd.DataFrame(np.random.randint(0,100,size=(15, 4)), columns=list('ABCD'))
d = []
for i in np.arange(N):
    # # AREA to cast random numbers in 
    # # 10 degrees from -125
    lon = np.random.random(1) * 10 - 125
    # # 10 degrees from 32
    lat = np.random.random(1) * 10 + 32
    pt = Point(lon, lat)
    d.append(
        {
            'muni_ID': ("city" + str(np.random.randint(0,60))),
            'subtype': 'MSW_food',
            'wt': np.random.randint(60,100),
            'geometry':  pt
            }  
    )
    # print(i)

msw = gpd.GeoDataFrame(d)


r = []
for i in np.arange(N):
    lon = np.random.random(1) * 10 - 125
    # # 10 degrees from 32
    lat = np.random.random(1) * 10 + 32
    pt = Point(lon, lat)
    r.append(
        {
            'OBJECTID' : np.random.randint(0,100),
            'capacity_m3' : np.random.randint(600000,700000),
            'centroid' : pt
        })

rangelands = gpd.GeoDataFrame(r)


f = []
for i in np.arange(N):
    lon = np.random.random(1) * 10 - 125
    # # 10 degrees from 32
    lat = np.random.random(1) * 10 + 32
    pt = Point(lon, lat)
    f.append(
        {
        'SwisNo' : ("facility" + str(np.random.randint(0,1000))),
        'cap_m3' : np.random.randint(3000,100000),
        'geometry' : pt
        })

facilities = gpd.GeoDataFrame(f)


s = []
for i in np.arange(N):
    f.append(
        {
        'County' : ("county" + str(np.random.randint(0,1000))),
        'seq_f' : np.random.randint(3000,100000),
        })

seq_factors =pd.DataFrame(s)



