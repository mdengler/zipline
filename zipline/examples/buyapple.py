#!/usr/bin/env python
#
# Copyright 2012 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
import pytz

from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_from_yahoo


class BuyApple(TradingAlgorithm):  # inherit from TradingAlgorithm
    """This is the simplest possible algorithm that does nothing but
    buy 1 apple share on each event.
    """
    def initialize(self):
        pass

    def handle_data(self, data):  # overload handle_data() method
        self.order('AAPL', 1)  # order SID (=0) and amount (=1 shares)


if __name__ == '__main__':

    import os
    headless = "DISPLAY" not in os.environ or os.environ["DISPLAY"] == ""
    if headless:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    start = datetime(2008, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2010, 1, 1, 0, 0, 0, 0, pytz.utc)
    data = load_from_yahoo(stocks=['AAPL'], indexes={}, start=start,
                           end=end)
    simple_algo = BuyApple()
    results = simple_algo.run(data)

    ax1 = plt.subplot(211)
    results.portfolio_value.plot(ax=ax1)
    ax2 = plt.subplot(212, sharex=ax1)
    data.AAPL.plot(ax=ax2)
    plt.gcf().set_size_inches(18, 8)

    if headless:
        plt.show()
    else:
        plt.savefig("buyapple.png")
