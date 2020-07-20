import json
from os.path import join as opj

RESULTS_DIR = "/Users/anayahall/projects/compopt/results"



countyareas = {}


foo = {
	'1': {
			'OBJECTID' : '1',
			'COUNTY': 'Sierra',
			'volume': 423
		},
	'2' : {
			'OBJECTID' : '2',
			'COUNTY' : 'Alameda',
			'volume' : 500
			},
	'3' : {
			'OBJECTID' : '3',
			'COUNTY' : 'Sierra',
			'volume' : 350
			}
}

bar = {
	'1': {
			'OBJECTID' : '1',
			'COUNTY': 'Sierra',
			'intake': 143
		},
	'2' : {
			'OBJECTID' : '2',
			'COUNTY' : 'Alameda',
			'intake' : 7500
			},
	'3' : {
			'OBJECTID' : '3',
			'COUNTY' : 'Sierra',
			'intake' : 30
			}
}






# goal
# 'Alameda' : x
county_results = {}
for k,v in foo.items():
	# print('k: ', k)
	# print('v: ', v)
	# print('V.COUNTY: ', v['COUNTY'])
	county = v['COUNTY']
	# print('county', county)
	if county in county_results:
		county_results[v['COUNTY']]['volume'] = county_results[v['COUNTY']]['volume'] + v['volume']
		# print('got in here...')
	else:
		county_results[county] = {}
		county_results[county]['volume'] = v['volume']

for k,v in bar.items():
    # print('k: ', k)
    # print('v: ', v)
    # print('V.COUNTY: ', v['COUNTY'])
    county = v['COUNTY']
    print('county', county)
    if county in county_results:
        if 'county_fac_intake' in county_results[county].keys(): 
            county_results[v['COUNTY']]['county_fac_intake'] = county_results[v['COUNTY']]['county_fac_intake'] + v['intake']
            # print('got in here...')
        else:
            county_results[county]['county_fac_intake'] = v['intake']
    else:
        county_results[county] = {}
        county_results[county]['county_fac_intake'] = v['intake']
    print(county_results)


        # print("RESULTS: ", county)
        # if county_results[county]['county_fac_intake'] in county_results[county]:
        #     print("got here") 
        # else:
        #     print("ehh")


    # # save quantities moved
    # with open(opj(RESULTS_DIR, str(scenario)+'_c2f_values.json'), 'w') as fp:
    #     json.dump(c2f, fp)

    # # save quantities moved
    # with open(opj(RESULTS_DIR, str(scenario)+'_f2r_values.json'), 'w') as fp:
    #     json.dump(f2r, fp)