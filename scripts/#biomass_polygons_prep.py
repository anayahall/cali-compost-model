#biomass_polygons_prep.py

gbm_shp, tbm_shp = MergeInventoryAndCounty(
    gross_inventory     = opj(DATA_DIR, "raw/biomass.inventory.csv"),
    technical_inventory = opj(DATA_DIR, "raw/biomass.inventory.technical.csv"),
    county_shapefile    = opj(DATA_DIR, "raw/CA_Counties/CA_Counties_TIGER2016.shp"),
    fips_data           = opj(DATA_DIR, "interim/CA_FIPS.csv")
)



print("exporting shapefile")
out = r"/Users/anayahall/projects/compopt/data/clean/biomass_clean shp.shp"

gbm_shp.to_file(driver='ESRI Shapefile', filename=out)

print("p BIOMASS SHAPEFILE SAVED RUNNING")