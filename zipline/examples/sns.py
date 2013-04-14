#!/usr/bin/python
#
# Copyright 2013 Martin T Dengler
#
# GPL v3+

import collections
import datetime
import pytz

import pandas
import pandas.io.data

from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_from_yahoo

__OCC_SUFFIX_LEN = 15

def option_symbol(root_symbol, expiry, put_or_call, K):
    """OCC 2010 symbology

    Concatenation of:
        Root symbol (varies in width)
        Expiration Year(yy)
        Expiration Month(mm)
        Expiration Day(dd)
        Call/Put Indicator (C or P)
        Strike price unit (5 chars zero-padded)
        Strike price fraction (3 chars zero-padded)

    References:
    http://www.investopedia.com/articles/optioninvestor/10/options-symbol-rules.asp
    http://biz.yahoo.com/opt/symbol.html

    Examples:

>>> sns.option_symbol(*sns.option_attributes(sns.option_symbol("GOOG", pandas.datetime(2010, 9, 17), "C", 530.123456))) == sns.option_symbol("GOOG", pandas.datetime(2010, 9, 17), "C", 530.123456)
True

>>> sns.option_symbol("GOOG", pandas.datetime(2010, 9, 17), "C", 530.123456)                                   'GOOG100917C00530123'

>>> sns.option_symbol(*sns.option_attributes(sns.option_symbol("GOOG", pandas.datetime(2010, 9, 17), "C", 530.123456)))
'GOOG100917C00530123'

    """
    expiry_str = expiry.strftime("%y%m%d")
    K_str = "{0:0>8.0f}".format(int(K * 1000))
    return "".join((root_symbol, expiry_str, put_or_call, K_str))


def option_attributes(option_symbol):
    """returns root_symbol, expiry, put_or_call, K from an OCC 2010
    symbology string"""
    root_symbol = option_symbol[:-__OCC_SUFFIX_LEN]
    rest = option_symbol[-__OCC_SUFFIX_LEN:]

    # remove exchange suffix (FUTURE: handle this better)
    # see http://www.cboe.com/DelayedQuote/QuoteHelp.aspx
    #   No hyphen or letter present = Composite
    #   A = AMEX American Stock Exchange
    #   B = BOX Boston Stock Exchange - Options
    #   E = CBOE Chicago Board Options Exchange
    #   I = BATS
    #   O = NASDAQ
    #   P = NYSE Arca
    #   X = PHLX Philadelphia Stock Exchange
    #   8 = ISE International Securities Exchange
    
    if "-" in rest:
        rest = rest[:rest.index("-") + 1]

    expiry_str = rest[:6]
    put_or_call = rest[6]
    K_str = rest[-8:]

    expiry = pandas.datetime.strptime(expiry_str, "%y%m%d")

    K = int(K_str) / 1000.

    return (root_symbol, expiry, put_or_call, K)


def load_option_prices(option_symbol):
    # see also pandas.io.data.get_options_data get_forwards_data
    data = collections.OrderedDict()
    start = pandas.datetime(2011, 11, 13, 0, 0, 0, 0, pytz.utc)
    end = pandas.datetime.today() - datetime.timedelta(7)
    option_symbol = "GOOG130420C00800000"
    optd = pandas.io.data.DataReader(option_symbol, 'yahoo', start, end).sort_index()
    data[option_symbol] = optd
    panel = pandas.Panel(data)
    panel.minor_axis = ['open', 'high', 'low', 'close', 'volume', 'price']
    panel.major_axis = panel.major_axis.tz_localize(pytz.utc)
    return panel



class SNS(TradingAlgorithm):

    def initialize(self, underlyings=None):
        self.params = dict()
        self.underlyings = underlyings or ["AAPL"]
        for underlying in self.underlyings:
            self.params[underlying] = dict((
                ("invested_call", False),
                ("invested_put", False),
                ))

    def handle_data(self, data):
        for underlying in self.underlyings:
            invested = (self.params[underlying]["invested_call"],
                        self.params[underlying]["invested_put"])
            if not all(invested):
                self.get_involved(underlying, data[underlying].price)
            if any(invested):
                self.get_out_if_needed(underlying, data[underlying].price)

    def get_involved(self, underlying, S_t):
        K_c = S_t

        sigma = 0.2 # TODO: make sensible

        K_u = K_c + (2 * sigma)
        K_d = K_c - (2 * sigma)

        call = option_symbol(underlying, expiry, "C", K_u)
        put = option_symbol(underlying, expiry, "P", K_p)

        params = self.params[underlying]

        params["K_c"] = K_c
        params["K_u"] = K_u
        params["K_p"] = K_p
        params["eps"] = 0.5 * sigma

        params["call"] = call
        params["put"] = put

        if not params["invested_call"]:
            params["invested_call"] = True
            self.order(call, -1)

        if not params["invested_put"]:
            params["invested_put"] = True
            self.order(put, -1)

    def get_out_if_needed(self, underlying, S_t):
        params = self.params[underlying]

        K_u = params["K_u"]
        K_p = params["K_p"]

        eps = params["eps"]

        call = params["call"]
        put = params["put"]

        if S_t > (K_u - eps) and params["invested_call"]:
            self.order(call, 1)
            params["invested_call"] = False

        if S_t < (K_p + eps) and params["invested_put"]:
            self.order(put, 1)
            params["invested_put"] = False



if __name__ == '__main__':

    import os
    headless = "DISPLAY" not in os.environ
    if headless:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    underlyings = ['AAPL']
    data = load_from_yahoo(stocks=underlyings, indexes={})
    sns_algo = SNS(underlyings=underlyings)
    results = sns_algo.run(data)
    results.portfolio_value.plot()


    if headless:
        plt.show()
    else:
        plt.savefig("sns.png")
