




##### Municipal Solid Waste (points) #####
msw = gpd.read_file(opj(DATA_DIR,
                  msw_shapefile))
# filter to just keep food and green waste (subject of regulations)
msw = msw[(msw['subtype'] == "MSWfd_wet_dryad_wetad") | (msw['subtype'] == "MSWgn_dry_dryad")]

# MSW DATA NOTES: 
# fog = Fats, Oils, Grease; lb = lumber; cd = cardboard; fd = food;
# pp = paper, gn = green; ot = Other ; suffix describes what the 
# waste is deemed suitable for

# rename categories to be more intuitive
msw['subtype'].replace({'MSWfd_wet_dryad_wetad': 'MSW_food', 
                        'MSWgn_dry_dryad': 'MSW_green'}, inplace = True)
# msw_sum =  msw.groupby(['County', 'subtype']).sum() # may combine???
msw_total = msw.groupby(['ID', 'County']).sum()

# ADJUST VALUES 
# I'll be using 100% of the GREEN waste, so leave as is
# for FOOD WASTE, take off 2.5%, 
# then of the remainer divert 62.5% of generation to compost
# (assume that 25% goes straight to compost, 75% goes to AD, which reduces volume of material by half, 
# before being composted)
# equivlant to 0.609375
# create new array of values
new_wt_values = msw[msw['subtype'] == 'MSW_food']['wt']*0.609375
# replace these in place!
msw.loc[msw['subtype'] == 'MSW_food', 'wt'] = new_wt_values
###################################