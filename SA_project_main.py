import pandas_datareader.data as web
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import scipy.odr.odrpack as odrpack
from statsmodels.tsa.stattools import adfuller as adf

def f(B, x):
    return B[0]*x + B[1]

df = pd.read_csv("C:\\Users\\zy108\\Desktop\\Statistical arbitrage\\project\\15_min_clean.csv")
nan_list = []
for i in range(1, len(df.columns)):
    price_list = list(df.iloc[:, i])
    for j in reversed(range(len(price_list))):
        if math.isnan(price_list[j]):
            break
    nan_list.append(j)

threshold = len(df) * 0.3

columns = ['Date']
max_num = -1
for i in range(1, len(df.columns)):
    name = df.columns[i]
    if nan_list[i-1] < threshold:
        if nan_list[i-1] > max_num:
            max_num = nan_list[i-1]
        columns.append(name)

df = df.loc[:, columns]
if max_num > -1:
    df = df.drop(range(max_num + 1))

columns = []
for name in df.columns:
    columns.append(name.split(' ')[0])

df.columns = columns
df.index = range(len(df))

stock_list = df.columns
stock_list = stock_list[1::]
log_stock_list = []

# Calculate log price

for stock in stock_list:
    log_stock_list.append('log_' + stock)
    df['log_' + stock] = df[stock].apply(lambda x: math.log(x))

log_df = df.loc[:, ['Date'] + log_stock_list]
log_df.columns = columns
# split the data into training data and test data
training_df = log_df.loc[range(int(len(log_df) * 1 / 2)), :]
test_df = log_df.loc[range(int(len(log_df) * 1 / 2), len(log_df)), :]



sector_list = ['information-technology', 'consumer-discretionary', 'consumer-staples', 'energies', 'financials',
               'health-care', 'industrials', 'materials', 'real-estate', 'telecom-services', 'utilities']

num = 0
selected_pairs_from_corr = []
pairs_sector_dict = {}

corr_dict = {}

# select pairs in the same sector from correlation criteria
for sector_name in sector_list:
    df1 = pd.read_csv("C:\\Users\\zy108\\Desktop\\Statistical arbitrage\\project\\sector\\{}.csv".format(sector_name))
    df1 = df1.drop([len(df1) - 1])
    equity_list = list(df1['Symbol'])
    num = num + len(equity_list)
    name_list = []
    for name in equity_list:
        if name in columns:
            name_list.append(name)

    sector_df = training_df.loc[:, name_list]

    # correlation criteria, correlation > 0.90
    corr_df = sector_df.corr()

    for i in range(len(name_list)):
        for j in range(i+1, len(name_list)):
            if abs(corr_df.loc[name_list[i], name_list[j]]) > 0.90:
                selected_pairs_from_corr.append(name_list[i] + '/' + name_list[j])
                corr_dict[name_list[i] + '/' + name_list[j]] = corr_df.loc[name_list[i], name_list[j]]
                pairs_sector_dict[name_list[i] + '/' + name_list[j]] = sector_name

# select pairs from cointegration criteria, choose the pairs with the minimum cointegration test value

print(len(selected_pairs_from_corr))

vol_dict = {}
max_vol = 0
vol_list = []
pairs_testvalue_dict = {}
for pairs in selected_pairs_from_corr:
    stock1 = pairs.split('/')[0]
    stock2 = pairs.split('/')[1]
    print(stock1, stock2)
    X = list(training_df.loc[:, stock1])
    Y = list(training_df.loc[:, stock2])

    # Use TLS to determine the hedge ratio
    linear = odrpack.Model(f)
    mydata = odrpack.RealData(X, Y)
    myodr = odrpack.ODR(mydata, linear, beta0=[1., 2.])
    myoutput = myodr.run()

    intercept = myoutput.beta[1]
    slope = myoutput.beta[0]

    residual = []
    for i in range(len(X)):
        residual.append(Y[i] - slope * X[i] - intercept)



    result = adf(residual)
    AIC_test_value = result[0]
    p_value = result[1]

    vol_dict[pairs] = np.std(residual)
    vol_list.append(np.std(residual))

    pairs_testvalue_dict[pairs] = AIC_test_value

sorted_by_value = sorted(pairs_testvalue_dict.items(), key=lambda x: x[1], reverse=False)
sorted_by_vol = sorted(vol_dict.items(), key=lambda x: x[1], reverse=True)

Top_pairs = []
print(max(vol_list))
print(np.mean(vol_list))
print(np.median(vol_list))
print(np.min(vol_list))

# print(sorted_by_value)

# select the pairs by test value
for pairs, vol in sorted_by_vol:
    if vol >= 0.02:
        Top_pairs.append((pairs, vol))
testvalue_dict1 = {}
for pairs, vol in Top_pairs:
    testvalue_dict1[pairs] = pairs_testvalue_dict[pairs]
    # print(pairs, vol_dict[pairs], test_value)

sorted_by_testvalue = sorted(testvalue_dict1.items(), key=lambda x: x[1], reverse=False)
# print(sorted_by_vol)

print(sorted_by_testvalue[:10])
Top_pairs = sorted_by_testvalue[:10]

info_df = pd.DataFrame(np.arange(10 * 5).reshape(10, 5), index= range(10), columns = ['Pairs', 'Correlation', 'Volatility', 'cointegration_test_value', 'Rank'])
num = 0
for pairs, test_value in Top_pairs:
    info_df.loc[num, 'Pairs'] = pairs
    info_df.loc[num, 'Correlation'] = corr_dict[pairs]
    info_df.loc[num, 'Volatility'] = vol_dict[pairs]
    info_df.loc[num, 'cointegration_test_value'] = test_value
    info_df.loc[num, 'Rank'] = num + 1
    num = num + 1

# info_df.to_csv("C:\\Users\\zy108\\Desktop\\Statistical arbitrage\\project\\pairs.csv")
# print(len(Top_pairs))
# print(Top_pairs)

