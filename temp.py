import csv
import scipy
from scipy.optimize import *
import numpy as np
import random

#how many standard deviations left of the expected value of returns we want to guarantee
#eg if stds=3, with 95% prob this is a lower bound for returns
Stds=2

#given stds, lower bound for returns
bondlowerbound=0.03

returns=[]
corrmatrix=[]


#Read in Returns Data
with open('returns.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        returns.append(row)


#Read in Correlation Matrix
with open('correlationmatrix.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        corrmatrix.append(row)

import pdb; pdb.set_trace()


#sets bounds for variables, each weight is between 0 and 1
a=(0,1)
bnds=[(a)]*(len(returns)-1)

#these are the constraints, we must have the weights sum to 1
cons=[]
con = {'type': 'eq', 'fun': lambda x: 1-sum(x[i-1] for i in range(1,len(returns)))}
cons = np.append(cons, con)

#make sure our lower bound for return, based on variance, is at least lower bound given above
con = {'type': 'ineq', 'fun': lambda x: ret(x)-Stds*var(x)**0.5-bondlowerbound}
cons = np.append(cons, con)

#get random seed
x0=getxo()

#solve, slsqp is nonlinear optimization with constraints
#may want to change maxiter for large portfolios, may not converge fast so this will give an approximate solution
sol=scipy.optimize.minimize(objective,x0,method='SLSQP',bounds=bnds,constraints=cons,options={'disp': True ,'maxiter':2000000})

#print solution to terminal
#if this takes awhile to solve, the problom either has no solution (infeasible) or too large
sol=sol.x
for i in range(0,len(sol)):
    print(sol[i])
