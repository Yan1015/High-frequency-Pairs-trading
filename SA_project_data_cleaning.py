import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

df = pd.read_excel("C:\\Users\\zy108\\Desktop\\Statistical arbitrage\\project\\Book2_yyy.xlsx")

df = df.drop([0])
df.index = range(len(df))

N = 505

equity_list = []
for i in range(505):
    equity_list.append(df.iloc[0, i * 3])
print(equity_list)
print(len(equity_list))

df = df.drop([0, 1])
df.index = range(len(df))
df = df.drop([len(df) - 1])
print(df)


date_list = list(df.iloc[:, 0])
columns = ['Date'] + equity_list

clean_df = pd.DataFrame(np.arange(len(df) * (N+1)).reshape(len(df), N + 1), index=range(len(df)), columns=columns)
clean_df['Date'] = date_list
clean_df.index = date_list
# clean_df = df.iloc[:, [0, 1]]
# clean_df.columns = ['Date', equity_list[0]]
for i in range(505):
    print(equity_list[i])
    right = df.iloc[:, [i * 3, i * 3 + 1]]
    right.columns = ['Date', equity_list[i]]
    right.index = list(right['Date'])
    value = math.nan
    for date in date_list:
        # print(date)
        if date not in right.index:
            clean_df.loc[date, equity_list[i]] = value
        else:
            value = right.loc[date, equity_list[i]]
            clean_df.loc[date, equity_list[i]] = right.loc[date, equity_list[i]]

print(clean_df)
clean_df.to_csv("C:\\Users\\zy108\\Desktop\\Statistical arbitrage\\project\\15_min_clean.csv", index=False)
