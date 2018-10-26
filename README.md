# High-frequency-Pairs-trading
## Introduction
This is the final project of Statistical Arbitrage course and it aims to apply pairs trading in high frequency data to realize auto-trading.

As we all know, pairs trading takes advantage of market inefficiencies based on a pair of stocks. The perception is to identify two stocks that move together and to take long and short positions simultaneously when they diverge abnormally. Thus, it is expected that the prices of the two stocks will converge to a mean in the future. In this project, we use correlation, cointegration and volitality these 3 criterias to select stock pairs.

## Data
Downloaded the 15-min data from Bloomberg, which are 505 stocks’ close price per 15 min from 4/2/2018 to 10/15/2018. 

Then deleted the stocks which have more than 30% missing data. And for some missing value, substituted it from the last pervious value.
and then calculate the log price for each stock

Downloaded the sector information from website https://www.barchart.com/stocks/sectors/rankings

## Methodology and Simulation
#### Correlation
Combine all the pairs in the same sector, use dataframe.corr() function in Python to calculate the correlation between stocks. 

Then select the pairs whose absolute correlation is greater than 0.90, because I think 0.9 can make sure the pairs have good correlation
#### Volatility
For the remaining stock pairs, calculate the cointegration coefficient and residual by Total Least Square regression. And I did this by the package scipy.odr.odrpack in Python. 

Then, choose the stock pairs with volatility > 0.02 because the mean of all the volatility is 0.023, the median is 0.019, I choose 0.02 as threshold to make sure delete the pairs with very low volatility.
#### Cointegration
After the correlation and Volatility part, for the remaining pairs, I applied the Augmented Dickey–Fuller test to the residuals to test the cointegrated relation between stocks. In python, it is using statsmodels.tsa.stattools package.

Then I sorted the pairs by ADF-test value from smallest to largest and selected the top 10 pairs which have smallest test value
#### Trading rules
I set the initial capital is $10000, and set Open position is in [1, 1.5, 2, 2.5, 3], Close position is in [-0.2, -0.1, 0, 0.1, 0.2] and Stoploss is in [2, 3, 4, 5, 6]. Here the number is the multiple of standard deviation of residual.

Do the iteration in training period to find the optimal 3 parameters which got maximum P&L. Then the parameters are what we used in test period.

#### Out-of-Sample test
I used the optimal parameters we got above to test the performance in test data.
