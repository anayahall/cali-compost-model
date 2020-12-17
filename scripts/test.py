# test.py

import pandas as pd
from os.path import join as opj

DATA_DIR = "data"
def Fetch(df, key_col, key, value):
	#counties['disposal'].loc[counties['COUNTY']=='San Diego'].values[0]
	return df[value].loc[df[key_col]==key].values[0]

perennial_file = "compostrates/perennial_CN_high.csv"
annual_file = "compostrates/annual_CN_high.csv"
grazed_file = "compostrates/grazedgrasslands_CN_high.csv"


value =  'GHG Emissions'


print("perennial")
perennial_rates = pd.read_csv(opj(DATA_DIR,
						perennial_file))

# for county in perennial_rates['County']:
# 	rate = Fetch(perennial_rates, 'County', county, value)

# 	print(rate)


print("annual")
annual_rates = pd.read_csv(opj(DATA_DIR,
						annual_file))

# for county in annual_rates['County']:
# 	rate = Fetch(annual_rates, 'County', county, value)
# 	print(rate)

print("grazed")
grazed_rates = pd.read_csv(opj(DATA_DIR, 
	grazed_file))

# for county in grazed_rates['County']:
# 	rate = Fetch(grazed_rates, 'County', county, value)
# 	print(rate)

# the rates above are MTCO2e/acre/year
# want to get to kgCO2e/m3/year
# conversion:
# MTCO2e/acre * (1acre/0.404686hectares) * (1 hectare / 63.5 cm3 compost) * (1 kg / 0.001 MT)
# seq_f = rate / 0.404686 / 63.5 / 0.001