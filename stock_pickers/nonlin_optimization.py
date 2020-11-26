from google.cloud import bigquery
from datetime import timedelta

import csv
import scipy
from scipy.optimize import *
import numpy as np
import random

#Assumptions
TotalWealth=1000000



#how many standard deviations left of the expected value of returns we want to guarantee
#eg if stds=3, with 95% prob this is a lower bound for returns
Stds=3

#given stds, lower bound for returns
bondlowerbound=0.03

returns=[]
corrmatrix=[]

def getCorrelation(StockA,StockB, corrmatrix):
    vert=-999
    horiz=-999

    for i in range(len(corrmatrix)):
        if corrmatrix[i][0]==StockA:
            vert=i
        if corrmatrix[i][0]==StockB:
            horiz=i
    return corrmatrix[vert][horiz]

#Read in Returns Data
with open('returns.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        returns.append(row)

#Store Column Locations
Stock=0
Ret=1
Std=2
Price=3

#Read in Correlation Matrix
with open('correlationmatrix.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        corrmatrix.append(row)

def objective(x):
    #calculates objective, to minimize returns we multiply by "-1" since scipy only does minimization
    return -1*sum(x[i-1]*float(returns[i][Ret]) for i in range(1,len(returns)))
def var(x):
    #calculates variance based on weights
    return sum(x[i-1]*x[j-1]*float(returns[i][Std])*float(returns[j][Std])*float(getCorrelation(returns[i][Stock],returns[j][Stock], corrmatrix)) for i in range(1,len(returns)) for j in range(1,len(returns)))
def ret(x):
    #calcula
    return -1*objective(x)
def getxo():
    #gets seed for optimization
    x=[]
    for i in range(len(returns)-1):
        x.append(random.uniform(0,1000000))
    x=[item/sum(x) for item in x]
    return x

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





class StockPicker:
  def __init__(self, **kwargs):
    self.predictions = kwargs['predictions']
    self.historical_data = kwargs['historical_data']
    self.correlation_lag = kwargs['correlation_lag']
    self.correlation_matrix = self.__generate_correlation()

  """
    Generates a portfolio based on the predictions.
    This code is adapated from Aaron Kreiner's algorithm and work.
  """
  def generate_portfolio(self):
    #calculates objective, to minimize returns we multiply by "-1" since scipy only does minimization
    def objective(x):
      return -1 * ret(x)

    # Calculates variance based on weights
    def var(x):
      return sum(x[i-1]*x[j-1]*float(returns[i][Std])*float(returns[j][Std])*float(getCorrelation(returns[i][Stock],returns[j][Stock], corrmatrix)) for i in range(1,len(returns)) for j in range(1,len(returns)))

    def ret(x):
      return sum(x[i-1] * float(returns[i][Ret]) for i in range(1,len(returns)))

    def getxo():
      #gets seed for optimization
      x=[]
      for i in range(len(returns)-1):
          x.append(random.uniform(0,1000000))
      x=[item/sum(x) for item in x]
      return x

    def getCorrelation(StockA, StockB, corrmatrix):
        vert=-999
        horiz=-999

        for i in range(len(corrmatrix)):
            if corrmatrix[i][0]==StockA:
                vert=i
            if corrmatrix[i][0]==StockB:
                horiz=i
        return corrmatrix[vert][horiz]

    Stds = 2
    # Set bounds for variables, each weight is between 0 and 1
    a=(0,1)
    bnds=[(a)]*(len(returns)-1)

    # These are the optimization constraints.
    cons= [
      # We must have the weights sum to 1.
      {
        'type': 'eq',
        'fun': lambda x: 1 - sum(x[i] for i in range(0, len(returns) - 1))
      },
      # Make sure our lower bound for return, based on variance, is at least lower bound given below.
      {
        'type': 'ineq',
        'fun': lambda x: ret(x) - Stds*var(x)**0.5 - bondlowerbound
      }
    ]

    #get random seed
    x0=getxo()

    #solve, slsqp is nonlinear optimization with constraints
    #may want to change maxiter for large portfolios, may not converge fast so this will give an approximate solution
    sol=scipy.optimize.minimize(objective,x0,method='SLSQP',bounds=bnds,constraints=cons,options={'disp': True ,'maxiter':2000000})

    #print solution to terminal
    #if this takes awhile to solve, the problom either has no solution (infeasible) or too large
    return sol.x

  """
    Code inspired from: https://www.interviewqs.com/blog/py_stock_correlation.
    Generate correlation matrix for all stocks in consideration.
    This is used to limit risk exposure, as certain stocks are more correlated than others.
  """
  def __generate_correlation(self):
    df = self.historical_data.pivot('date', 'permno', 'adjusted_prc').reset_index()
    min_date = query['date'].max() - timedelta(self.correlation_lag)
    df = df[df['date'] >= min_date]
    # TODO: Find out how this method deals with missing values (NaN). E.g. If stock started 2 months ago, and has a lot NaNs.
    return df.corr(method='pearson')

QUERY = f"""
  SELECT
      date, permno, adjusted_prc
  FROM
      `silicon-badge-274423.features.sp_daily_features`
  WHERE
      date >= '1999-06-01' AND
      date <= '2020-01-01'
"""
client = bigquery.Client(project='silicon-badge-274423')
query = client.query(QUERY).to_dataframe()
kwargs = {
  'predictions': {},
  'historical_data': query,
  'correlation_lag': 180
}
sp = StockPicker(**kwargs)
