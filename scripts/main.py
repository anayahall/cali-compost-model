# THIS IS THE MAIN SCRIPT FROM WHICH THE COMPOSTLP MODEL IS CALLED
# when running locally (primarily for testing) set LOCAL = TRUE
# this will subset the data to run more quickly
# when running on AWS set LOCAL = FALSE
# this will run the full dataset


import cvxpy as cp
import numpy as np
import os
from os.path import join as opj

import pandas as pd
import shapely as shp
import geopandas as gpd
import scipy as sp
import pickle
import time
import yagmail


############################################################
# Change this to activate/decativate print statements throughout
DEBUG = True

LOCAL = False

CROPLANDS = True

############################################################
if LOCAL == True:
    print("RUNNING LOCALLY - no email, just one one")
    # Send email results! 
    SEND_EMAIL = False
    #single run?
    TEST_RUN = True
    # also change ON DATALOAD SCRIPT
else: 
    print("RUNNING ON AWS - yes email, many runs")
    SEND_EMAIL = True
    TEST_RUN = False
############################################################

print(" - main - packages loaded - import compost LP script now") if (DEBUG == True) else ()
from compostLP import Haversine, Distance, Fetch, SolveModel, SaveModelVars

print(" - main - starting solves!!! ") if (DEBUG == True) else ()

############################################################
# Define function to package up and send email for each run! 
def PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost):
    # SAVE RESULTS C2F
    # with open('results/latest_c2f.p', 'wb') as f:
    with open('c2f_'+str(run_name)+'.p', 'wb') as f:
        pickle.dump(c2f_val, f)

    # SAVE RESULTS F2R
    with open('f2r_'+str(run_name)+'.p', 'wb') as f:
        pickle.dump(f2r_val, f)

    # SAVE LAND APPLICATION DICT
    with open('landapp_'+str(run_name)+'.p', 'wb') as f:
        pickle.dump(land_app, f)

    # EMAIL RESULTS
    if SEND_EMAIL == True:
        print("sending email......")
        # Yagmail - send from anayahall@gmail with password generated from gmail
        yag = yagmail.SMTP('anayahall@gmail.com', 'zpnzmthmdosrdcff')
        contents = [
            "**********************************************",
            "SCENARIO: ", str(run_name),
            "**********************************************",
            "**********************************************",        
            "Project Cost (Millions $): ", str(cost_millions),
            "**********************************************",
            "Optimal Object Value (MMT CO2eq):" , str(val),
            # "Optimal Object Value (MMT CO2eq)" , str(val*10**(-9))
            "**********************************************",
            "$/tCO2e MITIGATED: ", str(abatement_cost),
            "**********************************************",
        ]

        file1 = str('c2f_'+str(run_name)+'.p')
        file2 = str('f2r_'+str(run_name)+'.p')
        file3 = str('landapp_'+str(run_name)+'.p')

        yag.send(
            to='anayahall@berkeley.edu',
            subject="compopt run - results with attachment",
            contents=contents, 
            attachments=[file1, file2, file3],
        )
        print(" --- EMAIL SENT --- ")
    else:
        print("(No email sent)")

    print("-------------------------------------") if (DEBUG == True) else ()
    print("-------------------------------------")
    return


from dataload import msw, rangelands, facilities, croplands
############################################################
### RUN SCENARIOS! #########################################
############################################################



if CROPLANDS == False:
    print("RUNNING SCENARIOS FOR RANGELANDS")

    # NAME SCENARIO
    run_name = 'FG_50p'

    print(" ** SCENARIO ** : HALF DISPOSAL aka ", run_name) if (DEBUG == True) else ()

    # RUN THE MODEL!!!
    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green",
        disposal_min = 0.5)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)


    ############################################################

    if (TEST_RUN == True):
        raise Exception("one run!!!") 

    ############################################################

    # NAME SCENARIO
    run_name = "FG_100p"

    print(" ** SCENARIO ** : FG 100 aka ", run_name) if (DEBUG == True) else ()

    # RUN THE MODEL!!!
    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name,
        disposal_min = 1.0, 
        feedstock = 'food_and_green')

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)


    ############################################################

    # NAME SCENARIO
    run_name = "FG_75p"

    print(" ** SCENARIO ** : FG 75 aka ", run_name) if (DEBUG == True) else ()

    # RUN THE MODEL!!!
    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green",
        disposal_min = 0.75)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################


    # NAME SCENARIO
    run_name = "FG_50p_SHigh"

    print(" ** SCENARIO ** : SEQ HIGH aka ", run_name) if (DEBUG == True) else ()

    # RUN THE MODEL!!!
    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green", 
        disposal_min = 0.95, 
        seq_f = -357)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################

    # NAME SCENARIO
    run_name = "FG_100p_2xCapacity"

    print(" ** SCENARIO ** : Double Capacity aka ", run_name) if (DEBUG == True) else ()

    # RUN THE MODEL!!!
    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green", 
        # disposal_min = 0.5, 
        capacity_multiplier = 2)


    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)


    ############################################################

    run_name = "S_high"

    print(" ** SCENARIO ** : FG_S_high aka ", run_name) if (DEBUG == True) else ()

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green", 
        seq_f = -357)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)


    ############################################################

    run_name = "F_25p"

    print(" ** SCENARIO ** : food waste under 25 disposal rate aka ", run_name) if (DEBUG == True) else ()

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food",  
        disposal_min = 0.25)   

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)


    ############################################################

    run_name = "F_50p"
    print(" ** SCENARIO ** : food waste under 50 disposal rate aka ", run_name) if (DEBUG == True) else ()

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name,
        feedstock = "food", 
        disposal_min = 0.5)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)


    ############################################################

    run_name = "F_100p"

    print(" ** SCENARIO ** : food waste under 100 disposal rate aka ", run_name) if (DEBUG == True) else ()

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, feedstock = "food")

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################

    run_name = "F_20r"

    print(" ** SCENARIO ** : food waste under 20 percent recovered aka ", run_name) if (DEBUG == True) else ()

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name,
        feedstock = "food", 
        fw_reduction = 0.2)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################

    run_name = "FG_20r"

    print(" **** SCENARIO: Food and green waste with 20 percent fw reduction aka ", run_name) if (DEBUG == True) else ()

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name,
        feedstock = "food_and_green", 
        fw_reduction = 0.2)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################

    run_name = "F_nocap"

    print("next SCENARIO: food waste ignoring facillity capacity limitations aka ", run_name) if (DEBUG == True) else ()

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food", 
        ignore_capacity = True)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################

    run_name = "FG_nocap"

    print("next SCENARIO: food & green waste ignoring facillity capacity limitations aka ", run_name) if (DEBUG == True) else ()

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green", 
        ignore_capacity = True)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################
    #### SENSITIVITY RUNS ######################################
    ############################################################

    print("STARTING SENSTIVITY SCENARIOS") if (DEBUG == True) else ()

    run_name = "EVs"
    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        kilometres_to_emissions = 0.1)
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################

    # also need to do sensitivitys, for these just need the main results I think, not full df?
    run_name = "landfillEF0"

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green", 
        disposal_min =.5,
        landfill_ef = 0 )
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################
    run_name = "p_EF_high"

    # # print("s ******* scenario: process emis high")
    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green", 
        process_emis = 16)
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################
    run_name = "t_EF_high"

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green", 
        kilometres_to_emissions = 0.69)

    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################
    run_name = "spread_cost_low"

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name,  
        feedstock = "food_and_green", 
        spreader_cost = 3)

    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################
    run_name = "double_cap"

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green", 
        capacity_multiplier = 2)

    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################
    run_name = "collection_cost_high"

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green", 
        c2f_trans_cost = 1.2)


    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

############################################################

###################################################################


elif CROPLANDS == True: 
    print("RUNNING SCENARIOS FOR CROPLANDS")

    ############################################################
    # NAME SCENARIO
    run_name = 'FG_50p_CL'

    print(" ** SCENARIO ** : HALF DISPOSAL aka ", run_name) if (DEBUG == True) else ()

    # RUN THE MODEL!!!
    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green",
        disposal_min = 0.5,
        landuse = croplands)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    if (TEST_RUN == True):
        raise Exception("one run!!!") 

    ############################################################
    # NAME SCENARIO
    run_name = "FG_100p_CL"

    print(" ** SCENARIO ** : FG 100 aka ", run_name) if (DEBUG == True) else ()

    # RUN THE MODEL!!!
    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name,
        disposal_min = 1.0, 
        feedstock = 'food_and_green',
        landuse = croplands)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)


    ############################################################

    # NAME SCENARIO
    run_name = "FG_75p_CL"

    print(" ** SCENARIO ** : FG 75 aka ", run_name) if (DEBUG == True) else ()

    # RUN THE MODEL!!!
    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green",
        disposal_min = 0.75,
        landuse = croplands)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################


    # NAME SCENARIO
    run_name = "FG_50p_SHigh_CL"

    print(" ** SCENARIO ** : SEQ HIGH aka ", run_name) if (DEBUG == True) else ()

    # RUN THE MODEL!!!
    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green", 
        disposal_min = 0.95, 
        seq_f = -357,
        landuse = croplands)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################

    # NAME SCENARIO
    run_name = "FG_100p_2xCapacity_CL"

    print(" ** SCENARIO ** : Double Capacity aka ", run_name) if (DEBUG == True) else ()

    # RUN THE MODEL!!!
    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green", 
        # disposal_min = 0.5, 
        capacity_multiplier = 2,
        landuse = croplands)


    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)


    ############################################################

    run_name = "FG_100p_Shigh_CL"

    print(" ** SCENARIO ** : FG_S_high aka ", run_name) if (DEBUG == True) else ()

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food_and_green", 
        seq_f = -357,
        landuse = croplands)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)


    ############################################################

    run_name = "F_25p_CL"

    print(" ** SCENARIO ** : food waste under 25 disposal rate aka ", run_name) if (DEBUG == True) else ()

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food",  
        disposal_min = 0.25,
        landuse = croplands)   

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)


    ############################################################

    run_name = "F_50p_CL"
    print(" ** SCENARIO ** : food waste under 50 disposal rate aka ", run_name) if (DEBUG == True) else ()

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name,
        feedstock = "food", 
        disposal_min = 0.5,
        landuse = croplands)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)


    ############################################################

    run_name = "F_100p_CL"

    print(" ** SCENARIO ** : food waste under 100 disposal rate aka ", run_name) if (DEBUG == True) else ()

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name, 
        feedstock = "food",
        landuse = croplands)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################

    run_name = "F_20r_CL"

    print(" ** SCENARIO ** : food waste under 20 percent recovered aka ", run_name) if (DEBUG == True) else ()

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name,
        feedstock = "food", 
        fw_reduction = 0.2,
        landuse = croplands)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

    ############################################################

    run_name = "FG_20r_CL"

    print(" **** SCENARIO: Food and green waste with 20 percent fw reduction aka ", run_name) if (DEBUG == True) else ()

    c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost = SolveModel(scenario_name = run_name,
        feedstock = "food_and_green", 
        fw_reduction = 0.2,
        landuse = croplands)

    # Send EMAIL w results
    PackageEmail(c2f_val, f2r_val, land_app, cost_millions, val, abatement_cost)

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################


# okay goal here is to loop through a series of scenarios and save them all

# runs = ()
# for r in runs:
#   for v in vals:



############################################################
############################################################

# # for land app- need to merge with rangeland gdf
######### DO THIS RIGHT BEFORE PLOTTING

# # read in data
# # rangeland polygons
# rangelands = gpd.read_file(opj(DATA_DIR, "raw/CA_FMMP_G/gl_bycounty/grazingland_county.shp"))
# # county polygons

# # first turn into df
# land_df = pd.DataFrame.from_dict(land_app, orient = 'index')
# # need to get OBJECTID as str before merge
# rangelands.OBJECTID = rangelands.OBJECTID.astype(str)
# # merge land_app with rangeland geodataframe 
# rangelands_app = pd.merge(rangelands, land_df, on = 'OBJECTID')


# rangelands_app.to_file("rangelands_app_75.shp")

# raise Exception(" loaded function, ran scenario, prepped for plotting!!!")

############################################################
############################################################





# # Sleep loop
# print("*** Hit CTRL+C to stop ***")
 
# ## Star loop ##
# while True:
#     ### Show today's date and time ##
#     print("Current date & time " + time.strftime("%c"))
 
#     #### Delay for 2 minutes ####
#     time.sleep(120)
