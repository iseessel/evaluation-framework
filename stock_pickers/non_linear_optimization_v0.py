from datetime import timedelta
import pandas as pd
import scipy
from scipy.optimize import *
import numpy as np
import random


class NonLinearOptimization:
    def __init__(self, **kwargs):
        # NB: The ordering of the correlation matrix and the ordering of the predictions matrix correspond.
        self.predictions = kwargs['predictions']
        self.client = kwargs['client']
        self.correlation_matrix = kwargs['correlation_matrix']
        self.bond_return = kwargs['bond_return']
        self.options = kwargs['options']
        self.stds = 1

    def pick(self):
        # NB: The ordering of the stocks in these two lists need to correspond.
        corrmatrix = self.correlation_matrix
        returns = self.__get_predicted_returns()

        weights = self.__optimize(corrmatrix, returns)
        permno_weights = {}
        for i, w in enumerate(weights):
            permno = corrmatrix[0][i + 1]
            permno_weights[permno] = round(w, 6)

        return permno_weights

    def __optimize(self, corrmatrix, returns):
        # Store Column Locations
        Stock, Ret, Std, Price = 0, 1, 2, 3

        # Functions used for scipy optimize
        def getCorrelation(StockA, StockB, corrmatrix):
            vert = -999
            horiz = -999

            for i in range(len(corrmatrix)):
                if corrmatrix[i][0] == StockA:
                    vert = i
                if corrmatrix[i][0] == StockB:
                    horiz = i

            return corrmatrix[vert][horiz]

        def objective(x):
            # calculates objective, to minimize returns we multiply by "-1" since scipy only does minimization
            return -1 * sum(x[i - 1] * float(returns[i][Ret]) for i in range(1, len(returns)))

        def var(x):
            # calculates variance based on weights
            return sum(x[i - 1] * x[j - 1] * float(returns[i][Std]) * float(returns[j][Std]) * float(getCorrelation(returns[i][Stock], returns[j][Stock], corrmatrix)) for i in range(1, len(returns)) for j in range(1, len(returns)))

        def ret(x):
            return -1 * objective(x)

        def getxo():
            # gets seed for optimization
            x = []
            for i in range(len(returns) - 1):
                x.append(random.uniform(0, 1000000))
            x = [item / sum(x) for item in x]
            return x

        # # sets bounds for variables, each weight is between 0 and 1. .
        # # a = (0, 1)
        # a = (0, 0.1)
        # # a = (0, 0.05)
        # print(f"Diversification: {a}")
        # # Enforce diversification
        # # a = (0, 0.05)
        # bnds = [(a)] * (len(returns) - 1)

        a = self.options['constraint']
        if self.options['constrain_bonds']:
            bnds = [(a)] * (len(returns) - 1)
        else:
            bnds = [(a)] * (len(returns) - 2)
            bnds = bnds + [(0, 1)]

        print(f"Diversification: {a}")
        print(f"Bonds restricted: {bnds[-1]}")

        # these are the constraints, we must have the weights sum to 1
        cons = []
        con = {'type': 'eq', 'fun': lambda x: 1 -
               sum(x[i - 1] for i in range(1, len(returns)))}
        cons = np.append(cons, con)

        # make sure our lower bound for return, based on variance, is at least lower bound given above
        con = {'type': 'ineq', 'fun': lambda x: ret(
            x) - self.stds * var(x)**0.5 - self.bond_return}
        cons = np.append(cons, con)

        # get random seed
        x0 = getxo()

        # solve, slsqp is nonlinear optimization with constraints
        # may want to change maxiter for large portfolios, may not converge fast so this will give an approximate solution
        sol = scipy.optimize.minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons, options={
                                      'disp': True, 'maxiter': 2000000})
        
        ### Add second stage optimization here, did not change anything else
        
        #Store values from previous optimization
        sol=sol.x
        prevreturn=ret(sol)
        prevvar=var(sol)
        x0=sol
        
        # Add Constraints to ensure solution is as good as in first stage, and weight constraints
        cons=[]
        
        con = {'type': 'eq', 'fun': lambda x: 1-sum(x[i-1] for i in range(1,len(returns)))}
        cons = np.append(cons, con)


        con = {'type': 'ineq', 'fun': lambda x: -1*var(x)**0.5+prevvar**0.5}
        cons = np.append(cons, con)

        con = {'type': 'ineq', 'fun': lambda x: ret(x)-prevreturn}
        cons = np.append(cons, con)
        
        #solve, minimize variance
        sol=scipy.optimize.minimize(var,x0,method='SLSQP',bounds=bnds,constraints=cons,options={'disp': True ,'maxiter':200000})
        
        return sol.x

    def __get_predicted_returns(self):
        df = pd.DataFrame()
        df['Stock'] = self.predictions['permno'].astype(str)
        df['Expected Return'] = self.predictions['return_prediction']
        df['Standard Deviation'] = self.predictions['vol_prediction']
        # TODO: This is hacky, but its not used anywhere.
        df['Market Price'] = 1
        df['Weights'] = 0

        # Ensure correlation matrix and returns matrix are in the same order.
        returns_list = df.values.tolist()

        return [df.columns.tolist()] + returns_list
