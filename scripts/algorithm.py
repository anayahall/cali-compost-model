#the algorithm
import os
os.chdir("/Users/anayahall/projects/grapevine")

import parameters 
import fxns

# # PREPROCESSINGS & Data WRANGLING
# ## compost facilities
# import swis_preprocessing


# ## biomass inventory (both gross & technical)
# import biomass_preprocessing
# ### RESHAPE FROM LONG TO WIDE?

# ## rangelands (for now just rl...)
# import land

# # DISTANCE MATRICIES

# ## calculate cost distance matricies --> formalize into function!
# import distance_mat

# RUN LP ALGORITHM
import LP_test # <- NEED TO MAKE THIS REAL STILL

# VIEW RESULTS???


class Bob:
	def __init__(self, name):
		self.name = name
	def eat(self, food_type):
		self.health+=food_type

bob = Bob("larry")
toe = Bob("larry")
toe.eat(234)



import cvxpy as cp
import numpy as np

#Distances: supply to factory
d_s2f = np.array([
   [0, 1, 2],
   [1, 0, 3],
   [2, 3, 0]
])

#Distances: factory to rangeland
d_f2r = np.array([
   [0, 1, 2, 7],
   [1, 0, 3, 7],
   [2, 3, 0, 7],
   [7, 1, 2, 0]
])

cost_per_mile = 0.3

#Cost matrices
d_s2f = d_s2f*cost_per_mile
d_f2r = d_f2r*cost_per_mile

supply = np.array([
   10, 4, 5
])

fcapacity = np.array([
   3, 15, 4
])

rcapacity = np.array([
   4, 6, 4, 8
])

#Quantity supply to factory
q_s2f = cp.Variable(3)
q_f2r = cp.Variable(4)

#q_s2f = cp.Variable((3,3))
#q_f2r = cp.Variable((4,4))

#Initializing to 0
obj = 0
#cost of this
obj += cp.sum(d_s2f@q_s2f)
#Cost of choosing this factory to fields
obj += cp.sum(d_f2r@q_f2r)

cons = []
cons += [q_s2f<=fcapacity]
cons += [q_f2r<=rcapacity]
cons += [0<=q_s2f]
cons += [0<=q_f2r]
cons += [cp.sum(q_s2f)>=cp.sum(supply)]
cons += [cp.sum(q_f2r)>=cp.sum(supply)]

prob = cp.Problem(cp.Minimize(obj), cons)
val = prob.solve()
print("val",val)
print("q_s2f.value",q_s2f.value)
print("q_f2r.value",q_f2r.value)

