# countycosts.py
import json

DATA_DIR = "/Users/anayahall/projects/compopt/data"
RESULTS_DIR = "/Users/anayahall/projects/compopt/results"

c2f_trans_cost = 0.412, #$/m3-km # transit costs (alt is 1.8)
f2r_trans_cost = .206, #$/m3-km # transit costs
spreader_cost = 5.8, #$/m3 # cost to spread


# read in results
# scenarios = ['food', 'food_50', 'food_100', etc]
scenarios = ['food_50d']

with open("c2f_dist.json", "r") as read_file:
	dist_c2f = json.load(read_file)

with open("f2r_dist.json", "r") as read_file:
	dist_f2r = json.load(read_file)

for s in scenarios:
	landapp = opj(DATA_DIR, str(scenario)+"_LandApp.csv")
	land = pd.read_csv(landapp, 
		names = ['County', 'land_area', 'volume_applied'],
		header = 0)

raise Exception("LOAD DIST")

with open(opj(RESULTS_DIR, str(scenario)+'_c2f_values.json'), 'r') as read_file:
    c2f = json.load(read_file)

# save quantities moved
with open(opj(RESULTS_DIR, str(scenario)+'_f2r_values.json'), 'r') as read_file:
    f2r = json.load(read_file)

costs = {}
for county in dist_c2f.keys():
	cost[county]
	print(county)
	ship_cost = 0
	for facility in dist_c2f[county].keys()
		x = dist_c2f[county][facility]
		ship_cost += x['trans_dist']*c2f_trans_cost
