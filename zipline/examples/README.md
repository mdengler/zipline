Algos for testing out


Trend-following overlay and momentum regime change signal
---------------------------------------------------------

* Calculate the past 12-month momentum for each asset class using monthly excess returns, defined as the excess return above the Treasury Bill.
* If the asset momentum is positive, hold the asset for the next month.  If the momentum is negative, switch into the 90-day U.S. Treasury Bill.

Paper uses the following assets: MSCI US, MSCI EAFE, Barclays Capital Long U.S. Treasury, Intermediate U.S. Treasury, U.S. Credit, U.S. High Yield Corporate, U.S. Government & Credit, and U.S. Aggregate Bond indices, 90-day U.S. Treasury bills, FTSE NAREIT, S&P GSCI, and monthly gold returns
Paper shows that adding this momentum overlay to a 60/40 portfolio adds significant value by reducing drawdowns.
Paper also show this adds value when applied to a 5 asset portfolio (called the parity portfolio in the paper).
https://alpha.turnkeyanalyst.com/ideas/240




Meaningless trend-follower
--------------------------

* Calculate daily MA with different lags and normalize MA by dividing by monthly closing prices.
* Run cross-sectional regressions each month regressing monthly stock returns on the normalized MAs.
* Estimate the expected return for month t using the moving averages of the coefficients in the 12 months prior to month t and the normalized MAs from month t-1.
* Sort expected return into quintile.
* Long top 20% and short 20%, re-balance monthly.

https://alpha.turnkeyanalyst.com/ideas/235





Backwardation spreads
---------------------

* sell contango contracts where forward prices exceed spot prices
* buy  backwardation contracts where forward prices are lower than spot prices

Most of the profit comes from selling the contangoed contracts - what does that imply?

Contango is when the futures price is above the expected future spot price. Because the futures price must converge on the expected future spot price, contango implies that futures prices are falling over time as new information brings them into line with the expected future spot price.

Normal backwardation is when the futures price is below the expected future spot price. This is desirable for speculators who are "net long" in their positions: they want the futures price to increase. So, normal backwardation is when the futures prices are increasing.
