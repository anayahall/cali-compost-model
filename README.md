# California Compost Model

A spatial optimization model connecting california county-level MSW to composting
processing facilites and working agricultural lands, leveraging compost soil 
applications as a means of avoiding landfill emissions and sequestering carbon in 
the State's soils. 

*Inputs*: Tabular and geospatial data on MSW (tons), solid waste processing facilities 
by type, location and extent of working land across the state of California.

*Outputs*: Allocation of feedstock (food and green waste) to composters and allocation of finished compost to fields (cubic meters); economic cost associated with production/transportation of feedstocks and composts; abatement potential of 

*Key Parameters*: Diversion rate (50 to 75%), sequestration rate, feedstock ratio, transportation emission factor, collection/hauling cost, available composting capacity

### Status
As of July 2020, this project is near completion. The full repo will be made public 
with a detailed README.md and toy data set prior to publication.

## Data
0. Biomass
1. SWIS
2. Crop Land
3. Grazing land


## Scripts
0. Preprocessing
1. CompostLP (spatial optimization model)
2. Main (Scenario Runs)
3. Analysis and Figures

