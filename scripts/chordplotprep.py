# chordplotprep.py

import pandas as pd
import pickle
import os
import geopandas as gpd

from region import add_region_variable

# read in data:
# os.chdir('../results')

print("loading data")
with open('results/c2f_FG_50p.p', 'rb') as f:
    c2f = pickle.load(f)
    
with open('results/f2r_FG_50p.p', 'rb') as f:
    f2r = pickle.load(f)    

print("converting to df")
# convert to dataframe
c2fdf_swisno = pd.DataFrame.from_dict(c2f)
# reset index
c2fdf_swisno = c2fdf_swisno.reset_index()
# change column name to match 
c2fdf_swisno.rename(columns={'index':'SwisNo'},inplace=True)

# # grab names of counties (from column names, want as DF)
names = c2fdf_swisno.columns[1:].tolist()
names = pd.DataFrame(names, columns = ['County'])
# names = c2fdf.iloc[:,0] # old way of doing

# load swis data, keep only swis id number and county
swis = gpd.read_file("data/clean/clean_swis.shp")
swis = swis[['SwisNo', 'County']]

# merge swis to get county where each facility is located
c2fdf = swis.merge(c2fdf_swisno, on = 'SwisNo')

# merge with names so that there are rows for all
# c2fdf = c2fdf.merge(names, on = 'County', how = 'outer')
# need to sort and replace NaN with 0
# # c2fdf.fillna(0, inplace = True)
# c2fdf.sort_values(by = ['County'], inplace = True)
# c2fdf.reset_index(inplace=True, drop = True)


# rename to clarify that each row is where the swis facility is located
c2fdf.rename(columns={'County': 'SWIS_County'}, inplace = True)

# group by county! 
c2fdf = c2fdf.groupby(['SWIS_County'], as_index = False).sum()


print("reshaping")
# # reshape!
c2fdf_melted = pd.melt(c2fdf, id_vars=['SWIS_County'],var_name = 'MSW_County', value_name='value')
# c2fdf

c2fdf = c2fdf_melted.groupby(['MSW_County', 'SWIS_County'], as_index = False).sum()

# # c2fdf['value_scaled'] = round(c2fdf['value'] * 1/1000,-1)
c2fdf = c2fdf[c2fdf['value']!=0]
c2fdf.reset_index(inplace=True)

c2fdf.rename(columns={'MSW_County': 'from', 'SWIS_County': 'to'}, inplace=True)

# #then subset c2fdf to prepare to turn into matrix
c2fdf = c2fdf.iloc[:, 1:]

#save county to county flow
c2fdf.to_csv('results/chord/C2F_FG_50.csv')


flow_reg = add_region_variable(c2fdf, 'from')
flow_reg.rename(columns={'Region':'from_region'}, inplace = True)


flow_reg = add_region_variable(flow_reg, 'to')
flow_reg.rename(columns={'Region':'to_region'}, inplace = True)



flow_reg_group = flow_reg.groupby(['from_region', 'to_region'], as_index = False).sum()



# SAVE REGIONAL VALUES TO CSV FOR PLOTTING IN R
flow_reg_group.to_csv('results/chord/C2F_FG_50_region.csv')


## next do for F2R! 