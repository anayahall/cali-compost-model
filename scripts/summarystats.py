#summarystats.py

import json

with open('c2f.json') as json_file:
    c2f = json.load(json_file)

with open('f2r_DIST.json') as json_file:
    f2r = json.load(json_file)

#109 facilities
#58 counties

avgDict_f2r = {}
for facility in f2r.keys():
    # print(facility)
    temp = 0
    avgDict_f2r[facility] = {}
    avgDict_f2r[facility]['SwisNo'] = facility
    for rangeland in f2r[facility].keys():
        r_string = str(rangeland)
        temp += f2r[facility][r_string]['trans_dist']
        # print(temp) 
    avgDict_f2r[facility]['avg_dist'] = temp*(1/116)


temp = 0
for facility in f2r.keys():
	temp += avgDict_f2r[facility]['avg_dist']
average_dist_facility_rangeland = temp/109


avgDict_c2f = {}
for county in c2f.keys():
    # print(county)
    temp = 0
    avgDict_c2f[county] = {}
    avgDict_c2f[county]['COUNTY'] = county
    for facility in c2f[county].keys():
        temp += c2f[county][facility]['trans_dist']
        # print(temp) 
    avgDict_c2f[county]['avg_dist'] = temp*(1/109)

temp = 0
for county in c2f.keys():
	temp += avgDict_c2f[county]['avg_dist']
average_dist_county_facility = temp/58