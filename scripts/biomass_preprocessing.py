# Script to clean pre-process BIOMASS INVENTORY and make spatial

####################################################################
# First, load packages
import pandas as pd
import os
import geopandas as gpd
from fxns import epsg_meters
import shapely as shp

def MergeInventoryAndCounty(gross_inventory, technical_inventory, county_shapefile, counties_popcen):
    """
        Cleans biomass inventory data and merges with county shapefiles
        gross_inventory      - gross estimate of biomass inventory
        technical_inventory  - technical estimate of biomass inventory
        county_shapefile     - shapefile of county polygons
        counties_popcen      - csv population-weighted county centroids

        Returns: cleaned, spatial biomass data (assigned to pop-weighted 
        county centroids and county polygons)
    """

    ##################################################################
    #read in biomass inventory
    # GROSS inventory
    gbm = pd.read_csv(gross_inventory)

    # TECHNIcounty_shapeL inventory
    tbm = pd.read_csv(technical_inventory)


    gbm.rename(columns={"biomass.feedstock":"feedstock",
                          'biomass.category':'category',
                          'disposal.yields':'disposal_BDT'}, 
                 inplace=True)

    tbm.rename(columns={"biomass.feedstock":"feedstock",
                          'biomass.category':'category',
                          'disposal.yields':'disposal_BDT'}, 
                 inplace=True)    


    # check that all counties in there
    assert len(gbm.COUNTY.unique())==59
    #yup, plus one "other"

    # gbm[gbm['disposal.yields'] == gbm['disposal.yields'].max()]

    # #look at just manure (if feedstock, needs to be capitalized), if category, lower case -- should be equivalent!
    # gbm[(gbm['biomass.feedstock'] == "MANURE") & (gbm['year'] == 2014)].head()

    # #start grouping by: biomass category
    # gbm.groupby(['biomass.category'])['disposal.yields'].sum()
    # gbm[gbm['biomass.category'] == "manure"].groupby(['COUNTY'])['disposal.yields'].sum().head()

    fw_mc = 0.7
    gw_mc = 0.5
    manure_mc = 0.85

    def bdt_to_wettons(df):
        df['wettons'] = 0.0
        n = len(df.index)
        for i in range(n):
            if df.feedstock[i] == "FOOD":
                df.at[i, 'wettons'] = df.disposal_BDT[i] * (1 + fw_mc)
            if df.feedstock[i] == "GREEN":
                df.at[i, 'wettons'] = df.disposal_BDT[i] * (1 + gw_mc)
            if df.feedstock[i] == "MANURE":
                df.at[i, 'wettons'] = df.disposal_BDT[i] * (1 + manure_mc)

    bdt_to_wettons(gbm)
    bdt_to_wettons(tbm)

    # turn from wet tons to wet m3
    gbm['disposal_wm3'] = gbm['wettons'] / (1.30795*(1/2.24))
    tbm['disposal_wm3'] = tbm['wettons'] / (1.30795*(1/2.24))


    # # now load SHAPEFILE for all county_shape COUNTIES to merge this
    # print("p Read in county_shape COUNTIES shapefile and reproject")
    county_shape = gpd.read_file(county_shapefile)
    county_shape.rename(columns = {'NAME': 'COUNTY'}, inplace=True)
    county_shape= county_shape.to_crs(epsg=4326)
    county_shape['county_centroid'] = county_shape['geometry'].centroid 

    # ALSO LOAD IN CSV of population-weighted county centroids - use this not geographic centroid!!
    counties_popcen = pd.read_csv(counties_popcen) # NEW - population weighted means!
    counties_popcen.rename(columns = {'LATITUDE': 'lat', 'LONGITUDE': 'lon', 'COUNAME': 'COUNTY'}, inplace=True)
    counties_popcen['county_centroid'] = [shp.geometry.Point(xy) for xy in 
            zip(counties_popcen.lon, counties_popcen.lat)]

    #COUNTY POLYGONS with BIOMASS DATA(mostly for plotting)
    gbm_shp = pd.merge(county_shape, gbm, on = 'COUNTY')
    # Do same for technical biomass
    tbm_shp = pd.merge(county_shape, tbm, on = 'COUNTY')


    # POPULATION-WEIGHTED COUNTY CENTROIDS with BIOMASS DATA
    gbm_pts = pd.merge(counties_popcen, gbm, on = 'COUNTY')
    tbm_pts = pd.merge(counties_popcen, tbm, on = 'COUNTY')

    print("p BIOMASS PRE_PROCESSING DONE RUNNING")

    return gbm_pts, tbm_pts
