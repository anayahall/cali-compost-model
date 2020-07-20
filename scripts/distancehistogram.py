# distancehistogram.py

import numpy as np
import matplotlib.pyplot as plt
import json


## LOAD COUNTY TO FACILITY DISTANCES
with open('c2f_DIST.json', 'r') as fp:
	c2f_dist = json.load(fp)

# STORE ALL PAIRWISE DISTANCES
c2f_distance_array = []

# JUST store nearest facility DISTANCES
c2f_min_dist_array = []

# LOOP THROUGH DICT TO STORE AS ARRAY FOR PLOTTING
for county in c2f_dist.keys(): 
	temp = []
	for facility in c2f_dist[county].keys():
		x = c2f_dist[county][facility]['trans_dist']
		c2f_distance_array.append(x)
		temp.append(x)
	c2f_min_dist_array.append(np.min(temp))



## FACILITY TO RANGELAND DISTANCES
with open('f2r_DIST.json', 'r') as fp:
	f2r_dist = json.load(fp)

# ALL PAIRWISE DISTANCES
f2r_distance_array = []

# store nearest facility in  array
f2r_min_dist_array = []

# LOOP THROUGH DICT TO STORE AS ARRAY FOR PLOTTING
for facility in f2r_dist.keys(): 
	temp = []
	for rangeland in f2r_dist[facility].keys():
		x = f2r_dist[facility][rangeland]['trans_dist']
		f2r_distance_array.append(x)
		temp.append(x)
	f2r_min_dist_array.append(np.min(temp))

### PLOT HISTOGRAMS

fig, ax = plt.subplots(2,2, figsize = (10,10))
ax[0,0].hist(c2f_distance_array, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
ax[0,0].set_title("C2F: All pairwise distances")
ax[0,1].hist(f2r_distance_array, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
ax[0,1].set_title("F2R: All pairwise distances")
ax[1,0].hist(c2f_min_dist_array, bins= 30, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
ax[1, 0].set_title("C2F: Distance to Nearest Facility")
ax[1,1].hist(f2r_min_dist_array, bins=30, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
ax[1, 1].set_title("F2R: Distance to Nearest Rangeland")
for i in (range(2)):
    for j in (range(2)):
        ax[i,j].set_xlabel("Distance (km)")
        ax[i,j].set_ylabel("Count")

plt.show()


