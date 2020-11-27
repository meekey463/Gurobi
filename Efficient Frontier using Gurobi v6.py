# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:14:06 2020

@author: Meekey
"""

import quandl
import cvxopt
import pandas as pd
import numpy as np
from bokeh.palettes import Set1
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import Range1d, HoverTool, CrosshairTool, NumeralTickFormatter
from bokeh.layouts import row, gridplot
from bokeh.models.callbacks import CustomJS
from bokeh.transform import dodge
from bokeh.io import push_notebook, output_notebook
from datetime import datetime, timedelta
from math import sqrt, log, pi
output_notebook()


import cvxopt
from gurobipy import *
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.io import output_notebook
from bokeh.models import Range1d, HoverTool, NumeralTickFormatter
output_notebook()
import pandas as pd
import yfinance as yf
import datetime as dt
import bokeh.palettes
from bokeh.io import output_file, show, save

stocks =["^BSESN" , "TCS.NS", 'INFY.NS', 'EICHERMOT.NS', "NESTLEIND.NS",
        'VINATIORGA.NS','RELAXO.NS','MUTHOOTFIN.NS',
        'SONATSOFTW.NS','HDFCBANK.NS','BAJFINANCE.NS'
        ]

# stocks =["^BSESN" , "INFY.BO",   "GILLETTE.NS", "NESTLEIND.NS", "ABBOTINDIA.NS", "LAURUSLABS.NS",
#         'TCS.NS', 'ORIENTBELL.NS', 'TATAPOWER.NS', 'STYLAMIND.BO', 'BAJFINANCE.NS', 
#         'PVR.NS', 'MUTHOOTFIN.NS', 'JUBLFOOD.NS', 'VBL.NS'
#         ]



# stocks =["^BSESN", "TIDEWATER.NS", "SONATSOFTW.NS", "MANORG.BO", "SBIN.NS", "STRTECH.NS",
#         "GRANULES.NS", "M&MFIN.NS", "FSL.NS", "HINDUNILVR.NS", "TATAMOTORS.NS", "ITC.NS", "INDUSINDBK.NS", "LT.NS",
#         "MANAPPURAM.NS", "RELIANCE.NS", "ASIANPAINT.NS", "BAJAJ-AUTO.NS","HCLTECH.NS",
#         "CAPLIPOINT.NS", "PERSISTENT.NS", "KALPATPOWR.NS", "TORNTPOWER.NS", "TATAPOWER.NS", "PVR.NS", "JUBILANT.NS",
#         "JYOTHYLAB.NS", "WELSPUNIND.NS", "VINATIORGA.NS", "BAJFINANCE.NS", "AXISBANK.NS","TITANBIO.BO", 
#         "INFY.BO", "GLENMARK.NS", "BALRAMCHIN.NS", "PIDILITIND.NS", "CHOLAFIN.NS",
#         "KOTAKBANK.NS", "MOTHERSUMI.NS", "POLYPLEX.BO", "BAJAJFINSV.NS", "DABUR.NS", 'TCS.NS'
#         ]

# stocks =["^BSESN" , "TCS.NS",  "BAJAJ-AUTO.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "BAJAJFINSV.NS",
#         'INFY.NS', 'VINATIORGA.NS', 'RELIANCE.NS', 'CHOLAFIN.NS', 'BAJFINANCE.NS', 'CAPLIPOINT.NS',
#         'JUBILANT.NS', 'INDUSINDBK.NS'
#         ]


# stocks =["^BSESN" , "TCS.NS",  "PAGEIND.NS", "GILLETTE.NS", "NESTLEIND.NS", "ABBOTINDIA.NS",
#         'INFY.NS'
#         ]

#stocks = ['^BSESN', 'MUTHOOTFIN.NS', 'PVR.NS', 'JUBLFOOD.NS', 'HCLTECH.NS',
#           'HDFCBANK.NS', 'CAPLIPOINT.NS', 'SANOFI.NS', 'IPCALAB.NS', 'AARTIDRUGS.NS',
#           'GRANULES.NS'
#          ]

#Rakesh Jhunjhunwala
#'ORIENTCEM-EQ.NS', , 'PPL.NS' #no data available
# stocks = ['^BSESN', 'APTECHT.NS', 'TMRVL.NS', 'NCC.NS', 'RALLIS.NS', 'ATFL.BO',
#           'GEOJITFSL.NS', 'DELTACORP.NS', 'TITAN.NS', 'CRISIL.NS', 'VIPIND.NS',
#           'IONEXCHANG.BO', 'MCX.NS', 'ANANTRAJ.NS', 'FSL.NS', 'FORTIS.NS', 'LUPIN.NS',
#           'TATAMOTORS.NS',  'MANINFRA.NS', 'INDHOTEL.NS', 'AUTOIND.NS',
#           'DBREALTY.NS', 'EDELWEISS.NS', 'ESCORTS.NS', 'FEDERALBNK.NS', 'GMRINFRA.NS',
#           'JUBILANT.NS', 'KARURVYSYA.NS', 'PRAKASH.NS', 'PROZONINTU.NS', 'TV18BRDCST.NS',
#           'BI.BO', 'DCAL.NS'
#           ]

# stocks =["^BSESN" , "INDUSINDBK.NS",  "SBIN.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "AXISBANK.NS"
#         ]

# stocks =["MATRIMONY.NS", "AMARAJABAT.NS", "RAMCOCEM.NS",
#          "NTPC.NS","TATASTEEL.NS", "MINDTREE.NS"
#         ]

# stocks =["^GSPC" , "TSLA",  "AAPL", "MSFT", "AMZN", "GOOG",
#         'GE', 'BAC', 'FB', 'WMT', 'TGT']


start = "2010-01-01"
# end = dt.datetime.today()
end = "2015-12-30"
title = "Using quarterly stock price"
industry = "Small Cap"

stk_price = pd.DataFrame()
for ticker in stocks:    
        stk_price[ticker] = yf.download(ticker, start, end)["Adj Close"]
        
stk_price.isna().any()
addn_risk = 0

            
stk_price.to_csv("C:\\Users\\Meekey\\Documents\\CAPM\\temp.csv")

#cl_price = pd.read_csv("C:\\Users\\Meekey\\Documents\\CAPM\\temp.csv")
cl_price = pd.read_csv("C:\\Users\\Meekey\\Documents\\CAPM\\temp.csv")
cl_price = cl_price.dropna()

cl_price.Date = pd.to_datetime(cl_price.Date)
cl_price = cl_price.set_index('Date').resample('M').mean()

market_price = cl_price[stocks[0]]
cl_price = cl_price.drop(stocks[0], axis = 1)

market_ret = market_price.pct_change().dropna()
cum_market_ret = (market_ret +1).cumprod()

cl_price_ret = cl_price.pct_change()
cl_price_ret = cl_price_ret.dropna()
cum_stock_ret = (cl_price_ret + 1).cumprod()

#function to compare returns visually
# ret_com_plt = pd.DataFrame() 
# ret_com_plt = cl_price_ret[['NCC.NS','RALLIS.NS', 'MCX.NS']].copy()
# ret_com_plt['market'] = market_ret
# ret_com_plt.plot(figsize = (10,5))
# ret_com_plt.corr()


cl_price_ret.hist(bins = 20, figsize = (20, 15))


#cum_stock_ret.plot()
#convert cov matrix to yearly by multiplying by 12 months in a year
matrix_cov_stock_ret = (cl_price_ret.cov()) * 12
#matrix_cov_stock_ret

from bokeh.palettes import all_palettes
from bokeh.palettes import Category20c, Viridis
#CAGR
t = len(cum_stock_ret)/12
stk_annual_growth = (cum_stock_ret.tail(1))**(1/t) - 1
mkt_annual_growth = (cum_market_ret[-1])**(1/t) - 1
stk_annual_growth['market'] = mkt_annual_growth

#bar_colors = list(Category20c[len(stk_annual_growth.columns)+1])
bar_colors = list(Viridis[256][len(stk_annual_growth.columns)+1])

# stk_annual_growth.T.plot.bar(title = 'CAGR', color = bar_colors.pop())
stk_annual_growth.T.plot.bar(title = 'CAGR')
#avg_market_ret = market_ret.iloc[-4:].mean()
avg_market_ret = mkt_annual_growth


import seaborn as sns
from scipy import stats

#Calculate Stock Beta
stk_beta ={}
stk_ex_ret ={}
#rf = 0.065/12
def calc_stock_beta(mkt_ret, stock_ret):
#    mr = mkt_ret.copy()
#    sr = stock_ret.copy()
    for ticker in stock_ret:
        stock_beta, stk_alpha = stats.linregress(mkt_ret, 
                               stock_ret[ticker])[0:2]
        stk_beta[ticker] = stock_beta
        #stk_ex_ret[ticker] = rf + stock_beta*(mkt_ret[-1] - rf)
        #print(mkt_ret[-1])
        #stk_ex_ret[ticker] = rf + stock_beta*(mkt_ret.iloc[-3:].mean() - rf)
        #plt.barh(ticker, round(stock_beta,2))

calc_stock_beta(market_ret, cl_price_ret)
#market_ret[-1]

#cl_price = pd.read_csv("C:\\Users\\Meekey\\Documents\\CAPM\\temp.csv")
#cl_price.Date = pd.to_datetime(cl_price.Date)
#cl_price = cl_price.set_index('Date').resample('3m').mean()
#
#market_price = cl_price['^BSESN']
#cl_price = cl_price.drop(['^BSESN'], axis = 1)
#market_ret = market_price.pct_change().dropna()
#
#
#
#avg_market_ret = market_ret.iloc[-3:].mean()


#calculate yearly stk exp return
# mkt_annual_growth = 0.95
rf = 0.0325
annual_expected_stock_ret = {}
for ticker in stk_beta:
    annual_expected_stock_ret[ticker] = rf + stk_beta[ticker] * (mkt_annual_growth - rf)

annual_expected_stock_ret = pd.core.series.Series(annual_expected_stock_ret)

#stats = pd.DataFrame()
#def performance_statistics(data):
##    change = (data /  data.shift()- 1).iloc[1:]
##    stats = pd.concat((change.mean(), change.std(), data.iloc[-1] / data.iloc[0] - 1.0),axis=1)
#    #print(data.mean(), data.std(), (data + 1).prod()-1)
#    stats = pd.concat((data.mean(), data.std(), (data + 1).prod()-1),axis=1)
#    stats.columns = ['Mean return', 'Standard deviation', 'Total return']
#    return stats
#stats = performance_statistics(cl_price_ret)
#stats

#weights = np.random.random_sample((len(stocks)-1))
#weights = weights/np.sum(weights)
#weights = [0.1, 0.1, 0.15, 0.35, 0.1, 0.1, 0.1]
#weights = np.array([0.2,0.1,0.1,0.1,0.25,0.25])
#weights

from gurobipy import *



#ret = [0.01,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,0.021,
#       0.022,0.023,0.024,0.025,0.026,0.027,0.028,0.029,0.03,0.031,0.032,0.033,
#       0.034,0.035,0.036,0.037,0.038,0.039,0.04
#       ]
syms = matrix_cov_stock_ret.columns
pd.set_option('display.max_rows', None)
ret_st =  10
ret_end = 105
step = 5
risk = []
df=pd.DataFrame()
for mean_ret in range(ret_st,ret_end,step):
    # mean_ret = 120
    mean_ret  = mean_ret/1000
    m = Model()
    #create one variable for each stock
    portvars = [m.addVar(name = symb, lb = 0.0) for symb in syms]
    portvars = pd.Series(portvars, index = syms)
    portfolio = pd.DataFrame({'Variable' : portvars})
    m.update()
    p_total = portvars.sum()
    #p_return = stats['Mean return'].dot(portvars)
    p_return = annual_expected_stock_ret.dot(portvars)
    p_risk = matrix_cov_stock_ret.dot(portvars).dot(portvars)
    #p_risk = np.dot(weights.T,np.dot(matrix_cov_stock_ret, weights))
    m.setObjective(p_risk, GRB.MINIMIZE)
    #summ of weights
    m.addConstr(p_total==1)
    fixreturn = m.addConstr(p_return==mean_ret)
    m.update()
    #select simplex algorithm
    m.setParam('Method', 1)
    m.optimize()
    #portfolio['2.5% return'] = portvars.apply(lambda x:x.getAttr('x'))
    wts = portvars.apply(lambda x:x.getAttr('x'))
    m.remove(fixreturn)
    #portfolio_var = np.dot(wts,np.dot(matrix_cov_stock_ret, wts.T))
    portfolio_var = np.matmul(wts.T,np.matmul(matrix_cov_stock_ret, wts))
    portfolio_std = np.sqrt(portfolio_var)
    #portfolio_std
    risk.append(portfolio_std)
    df[mean_ret] = wts

#saving weights to backtest for different risk class
df.to_csv("C:\\Users\\Meekey\\Documents\\CAPM\\Spyder Code\\all_wts.csv")

 
x = np.array(risk)
#y = np.array(ret)
y = np.array(range(ret_st,ret_end,step))/1000
efficient_frontier_table = pd.DataFrame()
efficient_frontier_table['risk'] = x
efficient_frontier_table['ret'] = y

import matplotlib.pyplot as plt
plt.figure(figsize=(16, 5))
plt.scatter(x, y, c = (y-rf)/x)
plt.xlabel('risk')
plt.ylabel('return')
plt.colorbar(label = 'Risk-Return Ratio')


#flowers['petal_length'][flowers['species']=='setosa']
min_risk = {}
min_risk['min_portfolio_risk'] = min(risk)
pd.DataFrame(min_risk.items()).to_csv("C:\\Users\\Meekey\\Documents\\CAPM\\Spyder Code\\min_risk.csv")
#med_risk = np.float64(risk[29:30])
#cal_wts4_exp_ret = risk[int((len(risk))/12)]
ret4_min_risk = efficient_frontier_table['ret'][efficient_frontier_table['risk']==min(risk)]
#ret4_min_risk = efficient_frontier_table['ret'][efficient_frontier_table['risk']>cal_wts4_exp_ret]
wts4_min_risk = df[ret4_min_risk + addn_risk]
wts4_min_risk.columns = ['wts']
#annual growth


#saving weights to backtest
wts4_min_risk.to_csv("C:\\Users\\Meekey\\Documents\\CAPM\\Spyder Code\\port_wts.csv")







bar_colors = list(Viridis[256][len(wts4_min_risk.T.columns)+1])
# wts4_min_risk.plot.bar(title = 'ret:{},risk:{}'.format(ret4_min_risk,round(min(risk),2)), color = bar_colors.pop())
# wts4_min_risk.plot.bar(title = 'ret:{},risk:{}'.format(ret4_min_risk,round(min(risk),2)))

wts4_min_risk.plot.bar(title = 'Wts-Min_Risk ret:{},risk:{}'.format(ret4_min_risk.values,round(min(risk),2)), color = 'green')


#weighted_expected_stock_return =pd.DataFrame()
weighted_expected_stock_return= wts4_min_risk.copy()
weighted_expected_stock_return['annual ret']=annual_expected_stock_ret
# wt_exp_port_monthly_ret = np.sum(weighted_expected_stock_return['wts']*weighted_expected_stock_return['annual ret'])
wt_exp_port_monthly_ret = np.sum(weighted_expected_stock_return.iloc[:,0]*weighted_expected_stock_return.iloc[:,-1])
wt_exp_port_monthly_ret

wt_exp_port_std = np.sum(weighted_expected_stock_return.iloc[:,0]*cl_price_ret.std())
wt_exp_port_std

sharpe_ratio = (wt_exp_port_monthly_ret - rf)/wt_exp_port_std
sharpe_ratio

import matplotlib.pyplot as plt
plt.figure(figsize=(16, 5))
plt.scatter(x, y, c = (y-rf)/wt_exp_port_std)
plt.xlabel('risk')
plt.ylabel('return')
plt.title("Portfolio Sharpe Ratio : {}".format(round(sharpe_ratio,2)))
plt.colorbar(label = 'Sharpe-Ratio')



risk[int(len(risk)/2)]

#portfolio_var = np.dot(weights.T,np.dot(matrix_cov_stock_ret, weights))
#portfolio_std = np.sqrt(portfolio_var)
#portfolio_std

#sharpe_ratio = (wt_exp_port_monthly_ret - rf)/min(risk)
# sharpe_ratio = (wt_exp_port_monthly_ret - rf)/min(risk)



stats = pd.DataFrame()
def performance_statistics(data):
#    change = (data /  data.shift()- 1).iloc[1:]
#    stats = pd.concat((change.mean(), change.std(), data.iloc[-1] / data.iloc[0] - 1.0),axis=1)
    #print(data.mean(), data.std(), (data + 1).prod()-1)
    stats = pd.concat((data.mean(), data.std()*np.sqrt(12), (data + 1).prod()-1),axis=1)
   #stats = pd.concat((stock_rolling_mean, stock_rolling_var, (data + 1).prod()-1),axis=1)
    stats.columns = ['Mean return', 'Standard deviation', 'Total return']
    return stats
#calculate stats on last 12 months return
stats = performance_statistics(cl_price_ret[-12:-1])
stats

cl_price_ret[-12:-1].std() * np.sqrt(12)

t = len(cum_stock_ret)/12
stk_annual_growth = (cum_stock_ret.tail(1))**(1/t) - 1
mkt_annual_growth = (cum_market_ret[-1])**(1/t) - 1
stk_annual_growth['market'] = mkt_annual_growth

