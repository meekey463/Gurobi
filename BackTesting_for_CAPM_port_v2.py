# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 20:42:53 2020

@author: Meekey
"""

import pandas as pd
import numpy as np
from math import sqrt, log, pi
# output_notebook()


import matplotlib
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

# stocks =["^BSESN" , "TCS.NS",  "BAJAJ-AUTO.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "BAJAJFINSV.NS",
#         'INFY.NS', 'VINATIORGA.NS', 'RELIANCE.NS', 'CHOLAFIN.NS', 'BAJFINANCE.NS', 'CAPLIPOINT.NS',
#         'JUBILANT.NS', 'INDUSINDBK.NS'
#         ]

# stocks =["^GSPC" ,"TSLA",  "AAPL", "MSFT", "AMZN", "GOOG",
#         'GE', 'BAC', 'FB', 'WMT', 'TGT']

# stocks =["^BSESN", "TCS.NS", 'INFY.NS', 'EICHERMOT.NS', "NESTLEIND.NS",
#         'VINATIORGA.NS','RELAXO.NS','MUTHOOTFIN.NS',
#         'SONATSOFTW.NS','HDFCBANK.NS','BAJFINANCE.NS'
#         ]


# stocks =["^BSESN" , "TCS.NS",  "PAGEIND.NS", "GILLETTE.NS", "NESTLEIND.NS", "ABBOTINDIA.NS",
#         'INFY.NS'
#         ]

stocks = pd.read_excel("C:\\Users\\Meekey\\Documents\\DeepLearning\\All_Stocks4.xlsx")
stocks = stocks.Symbol.tolist()

start = "2020-04-01"
# end = "2019-01-01"
end = dt.datetime.today()
# end = "2019-12-30"
wts = pd.read_csv("C:\\Users\\Meekey\\Documents\\CAPM\\Spyder Code\\port_wts.csv", index_col=0)

all_wts = pd.read_csv("C:\\Users\\Meekey\\Documents\\CAPM\\Spyder Code\\all_wts.csv", index_col=0)

stk_price = pd.DataFrame()
for ticker in stocks:    
        stk_price[ticker] = yf.download(ticker, start, end, interval = '1mo',
                                        actions = False)["Adj Close"]
stk_price.isna().any()

index_fund = stk_price[stocks[0]].copy()
stk_price = stk_price.drop(stocks[0], axis = 1)

holding_period_return = (stk_price.iloc[-1, :] - stk_price.iloc[1, :])/stk_price.iloc[1, :]
wt_holding_period_return = np.sum(holding_period_return*wts.iloc[:,-1])
wt_holding_period_return   
# stk_price.to_csv("C:\\Users\\Meekey\\Documents\\CAPM\\temp.csv")
index_fund_HPR = (index_fund.iloc[-1] - index_fund.iloc[0])/index_fund.iloc[0]

a, b = all_wts.shape
col_list = all_wts.columns.tolist()
HPR_all_wts = {}
for w in range(0,b-1,5):
    col_name = col_list[w]
    var_wts = all_wts.iloc[:,w].copy()
    var_ret = np.sum(holding_period_return*var_wts)
    HPR_all_wts[col_name] = var_ret
    
df_HPR_all_wts = pd.DataFrame(HPR_all_wts.values(), index=HPR_all_wts.keys()).copy()
df_HPR_all_wts
df_HPR_all_wts.plot(legend=False, 
                    title = "Index Fund HPR:{}\nPortfolio Min Risk HPR:{}".format(round(index_fund_HPR,3), 
                                                                               round(wt_holding_period_return,3)))
# plt.plot(df_HPR_all_wts[0],df_HPR_all_wts[1])

# print("Index Fund Return:{}".format(round(index_fund_HPR,4)))



# #cl_price = pd.read_csv("C:\\Users\\Meekey\\Documents\\CAPM\\temp.csv")
# cl_price = pd.read_csv("C:\\Users\\Meekey\\Documents\\CAPM\\temp.csv")
# cl_price = cl_price.dropna()

# cl_price.Date = pd.to_datetime(cl_price.Date)
# cl_price = cl_price.set_index('Date').resample('M').mean()


# cl_price_ret = cl_price.pct_change()
# cl_price_ret = cl_price_ret.dropna()
# stock_ret_mean = pd.DataFrame(cl_price_ret.mean(), columns=['mean_ret'])


# wt_port_ret = np.sum(wts*stock_ret_mean.iloc[:,-1])
# wt_port_ret*100
port_vs_index = pd.DataFrame()
wts_list = (wts.values.copy()).tolist()
# wts_list = (all_wts.iloc[:,-3].values).tolist()
i=0

for ticker in stk_price.columns:
    port_vs_index[ticker] = stk_price[ticker].pct_change()*wts_list[i]
    i+=1
    
index_fund_ret = index_fund.copy().pct_change()
index_fund_ret_cum = (index_fund_ret + 1).cumprod()


monthly_port_ret = port_vs_index.sum(axis=1)
monthly_port_ret_cum = (monthly_port_ret + 1).cumprod()
monthly_port_ret_cum.plot(label="Port Ret")
index_fund_ret_cum.plot(label="Mkt Ret")
plt.legend()
# stk_price.to_csv("C:\\Users\\Meekey\\Documents\\CAPM\\stk_price.csv")

port_vs_index_2 = pd.DataFrame()

for i in range(0, len(all_wts.columns)-1,1):
    wts_list = (all_wts.iloc[:,i].values.copy()).tolist()
    for ticker in stk_price.columns:
    port_vs_index_2[ticker] = stk_price[ticker].pct_change()*all_wts.iloc[:,i]
    i+=1
monthly_port_ret = port_vs_index.sum(axis=1)
monthly_port_ret_cum = (monthly_port_ret + 1).cumprod()
monthly_port_ret_cum.plot(label="Port Ret")
index_fund_ret_cum.plot(label="Mkt Ret")
plt.legend()
















