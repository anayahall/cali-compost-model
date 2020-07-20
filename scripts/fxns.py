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
