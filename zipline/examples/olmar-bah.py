# OLMAR + BAH

import numpy as np

import collections
import datetime
import functools
import pytz

import pandas
import pandas.io.data

import zipline.algorithm
import zipline.transforms
import zipline.finance.slippage
import zipline.finance.commission
import zipline.utils.factory



INSTRUMENTS = ['AMD', 'CERN', 'COST', 'DELL', 'GPS', 'INTC', 'MMM']


def sid(sid_int):
    return str(sid_int)


def batch_transform(**kwargs_p):
    """wraps batch_transform from zipline.transforms like in
    quantopian
    """
    def batch_transform_zipline(func):
        """Decorator function to use instead of inheriting from BatchTransform.
        For an example on how to use this, see the doc string of BatchTransform.
        """

        @functools.wraps(func)
        def create_window(*args, **kwargs):
            # passes the user defined function to BatchTransform which it
            # will call instead of self.get_value()

            def partial_func(func):
                return func(*args)

            kwargs_w = dict(kwargs_p)
            kwargs_w.update(kwargs)
            print func, len(args), kwargs_w
            return zipline.transforms.BatchTransform(func=partial_func(func),
                                                     **kwargs_w)

        return create_window

    return batch_transform_zipline


class OLMARBAH(zipline.algorithm.TradingAlgorithm):
    refresh_period_days = 1
    window_sizes = (3,4,5)

    def initialize(context):
        context.stocks = [sid(351),
                          sid(1419),
                          sid(1787),
                          sid(25317),
                          sid(3321),
                          sid(3951),
                          sid(4922)]

        context.window_sizes = context.__class__.window_sizes
        context.window_size_max = max(context.window_sizes)

        context.m = len(context.stocks) #Number of stocks in portfolio
        context.w = len(context.window_sizes) #Number of window sizes

        context.b_t = np.ones(context.m) / context.m #Initial b_t vector
        context.eps = 1.00  #change epsilon here

        #Matrix of different b_t for each window size
        context.b_w = np.ones((context.m,context.w)) / context.m
        context.s_w = np.ones(context.w) #Vector of cum. returns from each window size

        context.init = False
        context.comPerTrade = 0

        slippage_model = zipline.finance.slippage.VolumeShareSlippage(
            volume_limit=0.25,
            price_impact=0)
        context.set_slippage(slippage_model)
        context.set_commission(zipline.finance.commission.PerShare(cost=0.00))

    def handle_data(context, data):

        # get data
        d = context.__class__.get_data(data, context.stocks)
        if d is None:
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
        for k, w in enumerate(contex.window_sizes):
            #Caluclate predicted x for that window size
            for i, stock in enumerate(context.stocks):
                #Predicted ratio of price change from t to t+1
                x_tilde_a[i] = (np.mean(prices[context.window_size_max-w:,i])
                                /
                                prices[context.window_size_max-1,i])

            #Update the b_w matrix
            context.b_w[:,k] = find_b_norm(context,x_tilde_a,context.b_w[:,k])

        length_p = len(prices[:,0])    #Length of price vector
        p_t = prices[length_p-1,:]    #Price vector today
        p_y = prices[length_p-2,:]    #Price vector yesterday

        #Ratio of price change (1 x m) vector from t-1 to t
        x_t = np.true_divide(p_t,p_y)

        #w is length of window_sizes
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

    @classmethod
    @batch_transform(refresh_period=refresh_period_days,
                     window_length=max(window_sizes))
    def get_data(cls, datapanel, sids):
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
    data = zipline.utils.factory.load_from_yahoo(stocks=INSTRUMENTS,
                                                 indexes={},
                                                 cache=True)
    algo = OLMARBAH()
    results = algo.run(data)
    results.portfolio_value.plot()


    if headless:
        plt.show()
    else:
        plt.savefig("%s.png" % algo_name)
