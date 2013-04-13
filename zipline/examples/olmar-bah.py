# OLMAR + BAH

import numpy as np

import collections
import datetime
import pytz

import pandas
import pandas.io.data

from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_from_yahoo


class OLMARBAH(TradingAlgorithm):
    #globals for get_avg batch transform decorator
    R_P = 1 #refresh period in days
    W_L = [3,4,5] #Vector of different window size
    W_MAX = max(W_L) #Max window size

    def initialize(context):
        #['AMD', 'CERN', 'COST', 'DELL', 'GPS', 'INTC', 'MMM']
        context.stocks = [sid(351), sid(1419), sid(1787), sid(25317), sid(3321), sid(3951), sid(4922)]

        context.m = len(context.stocks) #Number of stocks in portfolio
        context.w = len(W_L) #Number of window sizes

        context.b_t = np.ones(context.m) / context.m #Initial b_t vector
        context.eps = 1.00  #change epsilon here

        context.b_w = np.ones((context.m,context.w)) / context.m #Matrix of different b_t for each window size
        context.s_w = np.ones(context.w) #Vector of cum. returns from each window size

        context.init = False
        context.comPerTrade = 0
        set_slippage(slippage.VolumeShareSlippage(volume_limit=0.25,price_impact=0))
        set_commission(commission.PerShare(cost=0.00))

    def handle_data(context, data):

        # get data
        d = get_data(data,context.stocks)
        if d == None:
           return

        prices = d[0]
        #volumes = d[1]

        if not context.init:
            rebalance_portfolio(context, data, context.b_t)
            context.init = True
            return

        m = context.m #Lenth of b vector
        x_tilde_a = np.zeros(m) #x_tilde vector
        b_optimal = np.zeros(m) #b_vector

        #For each window, we calculate the optimal b_norm
        for k, w in enumerate(W_L):
            #Caluclate predicted x for that window size
            for i, stock in enumerate(context.stocks):
                #Predicted ratio of price change from t to t+1
                x_tilde_a[i] = np.mean(prices[W_MAX-w:,i])/prices[W_MAX-1,i]

            #Update the b_w matrix
            context.b_w[:,k] = find_b_norm(context,x_tilde_a,context.b_w[:,k])

        length_p = len(prices[:,0])    #Length of price vector
        p_t = prices[length_p-1,:]    #Price vector today
        p_y = prices[length_p-2,:]    #Price vector yesterday

        #Ratio of price change (1 x m) vector from t-1 to t
        x_t = np.true_divide(p_t,p_y)

        #w is length of W_L
        #Daily returns (1 x w) vector
        s_d = np.dot(x_t,context.b_w)

        #Cumulative returns (1 x w) vector
        context.s_w = np.multiply(context.s_w,s_d)

        #Calculate return_weights (1 x w) vector
        return_weights = np.true_divide(context.s_w[:],np.sum(context.s_w)) #Weight according to cum. returns

        #Calculate b_{t+1} (n x 1) vector
        b_optimal = np.dot(context.b_w,return_weights) #Calculate the weighted portfolio

        #SJUAR
        rebalance_portfolio(context, data, b_optimal)

        # update portfolio
        context.b_t = b_optimal

    #Calculate b_norm for a give b_t and x_tilde
    def find_b_norm(context,x_tilde,b_t):
        ###########################
        # Inside of OLMAR (algo 2)
        x_bar = x_tilde.mean()

        # Calculate terms for lambda (lam)
        dot_prod = np.dot(b_t, x_tilde)

        num = context.eps - dot_prod
        denom = (np.linalg.norm((x_tilde-x_bar)))**2

        # test for divide-by-zero case
        if denom == 0.0:
            lam = 0 # no portolio update
        else:
            lam = max(0, num/denom)

        b = b_t + lam*(x_tilde-x_bar)

        #Finding min distance b_{t} and b_{t+1}
        b_norm = simplex_projection(b)

        return b_norm

    # set globals R_P & W_L above
    @batch_transform(refresh_period=R_P, window_length=W_MAX)
    def get_data(datapanel,sids):
        p = datapanel['price'].as_matrix(sids)
        v = datapanel['volume'].as_matrix(sids)
        return [p,v]

    def rebalance_portfolio(context, data, desired_port):
        #rebalance portfolio
        current_amount = np.zeros_like(desired_port)
        desired_amount = np.zeros_like(desired_port)

        if not context.init:
            positions_value = context.portfolio.starting_cash
        else:
            positions_value = context.portfolio.positions_value + context.portfolio.cash

        for i, stock in enumerate(context.stocks):
            current_amount[i] = context.portfolio.positions[stock].amount
            desired_amount[i] = desired_port[i]*positions_value/data[stock].price

        diff_amount = desired_amount - current_amount

        for i, stock in enumerate(context.stocks):
            order(stock, diff_amount[i]) #order_stock


def simplex_projection(v, b=1):
    """Projection vectors to the simplex domain

Implemented according to the paper: Efficient projections onto the
l1-ball for learning in high dimensions, John Duchi, et al. ICML 2008.
Implementation Time: 2011 June 17 by Bin@libin AT pmail.ntu.edu.sg
Optimization Problem: min_{w}\| w - v \|_{2}^{2}
s.t. sum_{i=1}^{m}=z, w_{i}\geq 0

Input: A vector v \in R^{m}, and a scalar z > 0 (default=1)
Output: Projection vector w

:Example:
>>> proj = simplex_projection([.4 ,.3, -.4, .5])
>>> print proj
array([ 0.33333333, 0.23333333, 0. , 0.43333333])
>>> print proj.sum()
1.0

Original matlab implementation: John Duchi (jduchi@cs.berkeley.edu)
Python-port: Copyright 2012 by Thomas Wiecki (thomas.wiecki@gmail.com).
"""

    v = np.asarray(v)
    p = len(v)

    # Sort v into u in descending order
    v = (v > 0) * v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)

    rho = np.where(u > (sv - b) / np.arange(1, p+1))[0][-1]
    theta = np.max([0, (sv[rho] - b) / (rho+1)])
    w = (v - theta)
    w[w<0] = 0
    return w




if __name__ == '__main__':

    import os
    headless = "DISPLAY" not in os.environ
    if headless:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algo_name = "olmar-bah"
    throwaway_algo = OLMARBAH().intialize()
    underlyings = throwaway_algo.stocks
    data = load_from_yahoo(stocks=underlyings, indexes={})
    algo = OLMARBAH()
    results = algo.run(data)
    results.portfolio_value.plot()


    if headless:
        plt.show()
    else:
        plt.savefig("%s.png" % algo_name)
