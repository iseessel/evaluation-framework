from datetime import timedelta
import pandas as pd
import scipy
from scipy.optimize import *
import numpy as np
import random

class NonLinearOptimization:
  def __init__(self, **kwargs):
    self.predictions = self.__get_predictions(kwargs['predictions'])
    self.client = kwargs['client']
    self.stds = 2
    self.bond_return = 0.03

  def pick(self):
    # NOTE: The ordering of the stocks in these two lists need to correspond.
    corrmatrix = self.__create_correlation_matrix()
    returns = self.__get_predicted_returns()

    weights = self.__optimize(corrmatrix, returns)
    permno_weights = {}
    for i, w in enumerate(weights):
      permno = corrmatrix[0][i + 1]
      permno_weights[permno] = round(w, 6)

    return permno_weights

  def __optimize(self, corrmatrix, returns):
    #Store Column Locations
    Stock, Ret, Std, Price = 0, 1, 2, 3

    # Functions used for scipy optimize
    def getCorrelation(StockA, StockB, corrmatrix):
        vert=-999
        horiz=-999

        for i in range(len(corrmatrix)):
            if corrmatrix[i][0]==StockA:
                vert=i
            if corrmatrix[i][0]==StockB:
                horiz=i

        return corrmatrix[vert][horiz]

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
    con = {'type': 'ineq', 'fun': lambda x: ret(x)-self.stds*var(x)**0.5-self.bond_return}
    cons = np.append(cons, con)

    #get random seed
    x0=getxo()

    #solve, slsqp is nonlinear optimization with constraints
    #may want to change maxiter for large portfolios, may not converge fast so this will give an approximate solution
    sol=scipy.optimize.minimize(objective,x0,method='SLSQP',bounds=bnds,constraints=cons,options={'disp': True ,'maxiter':2000000})

    return sol.x

  def __get_predicted_returns(self):
    df = pd.DataFrame()
    df['Stock'] = self.predictions['permno'].astype(str)
    df['Expected Return'] = (self.predictions['adjusted_prc_pred']/self.predictions['adjusted_prc_train_end']) - 1
    df['Standard Deviation'] = (self.predictions['std_dev_pred']/self.predictions['adjusted_prc_train_end'])
    df['Market Price'] = self.predictions['adjusted_prc_train_end']
    df['Weights'] = 0

    # Ensure correlation matrix and returns matrix are in the same order.
    df = df.sort_values(by='Stock')
    returns_list =  df.values.tolist()
    # Append bond returns last. NOTE: This ordering is coupled with bond correlation below.
    returns_list.append(['bond', 0.03, 0, 1, 0])

    return [df.columns.tolist()]+ returns_list

  def __create_correlation_matrix(self):
    # Get Prices for last 1 year since train_end.
    train_end = self.predictions['train_end'].min()

    QUERY = f"""
      SELECT
          date, permno, adjusted_prc
      FROM
          `silicon-badge-274423.features.sp_daily_features`
      WHERE
          date >= '{ (train_end - timedelta(365)).strftime('%Y-%m-%d') }' AND
          date <= '{train_end.strftime('%Y-%m-%d')}'
    """
    query = self.client.query(QUERY).to_dataframe()
    # Only calculate available permnos.
    available_permnos = self.predictions['permno'].tolist()
    available_permnos = [str(s) for s in available_permnos]
    df = query[query['permno'].isin(available_permnos)].sort_values(by='permno')

    df = df.pivot('date', 'permno', 'adjusted_prc').reset_index()
    min_date = query['date'].max() - timedelta(365)
    df = df[df['date'] >= min_date]

    # Convert prices into returns.
    returns = df.loc[:, df.columns != 'date'].pct_change(1)
    returns['date'] = df['date']

    # Get correlation matrix.
    correlations = returns.corr(method='pearson')

    # Add the correlations of treasury rate.
    correlations['bond'] = 0
    correlations.loc['bond'] = 0
    correlations['bond']['bond'] = 1
    correlations.reset_index(level=0, inplace=True)

    # To list of list with column headers.
    return [correlations.columns.tolist()] + correlations.values.tolist()

  def __get_predictions(self, predictions):
    # TODO: Rank by sharpe ratio and only get those that are positive.
    return predictions[predictions['adjusted_prc_pred'] > predictions['adjusted_prc_train_end']].sort_values(by="permno")
