# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 00:46:55 2020

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


stocks = ['^BSESN', 'TITANBIO.BO', 'TORNTPOWER.NS', 'POLYPLEX.NS', 'TATAPOWER.NS', 'PVR.NS', 'EICHERMOT.NS',
         'JUBILANT.NS', 'MUTHOOTFIN.NS', 'GTPL.BO', 'AARTIDRUGS.NS', 'TIDEWATER.NS', 'TATAELXSI.NS', 'JUBLFOOD.NS',
         'SEQUENT.NS', 'JYOTHYLAB.NS', 'LAURUSLABS.NS', 'VBL.NS', 'PONNIERODE.NS', 'BBTC.NS', 'ALLCARGO.NS',
         'WELSPUNIND.NS', 'SONATSOFTW.NS', 'MANORG.BO', 'GTPL.NS', 'AUROPHARMA.NS', 'MATRIMONY.BO', 'HDFCBANK.NS',
         'SBIN.NS', 'STRTECH.NS', 'RALLIS.NS', 'GRANULES.NS', 'M&MFIN.NS', 'VINATIORGA.NS',
         'DMART.NS', 'FSL.NS', 'RELAXO.NS', 'TATASTEEL.NS', 'HINDUNILVR.NS', 'CROMPTON.NS',
         'CONCOR.NS', 'MOTHERSUMI.NS', 'BAJFINANCE.NS', 'BATAINDIA.NS', 'TATAMOTORS.NS', 'IPCALAB.NS',
         'CADILAHC.NS', 'VOLTAS.NS', 'ITC.NS', 'INDUSINDBK.NS', 'RAMCOCEM.NS', 'ICICIBANK.NS', 'AXISBANK.NS',
         'LT.NS', 'TCS.BO', 'MANAPPURAM.NS', 'RELIANCE.NS', 'ASIANPAINT.NS',
         'BAJAJ-AUTO.NS', 'IOLCP.BO', 'WHIRLPOOL.NS', 'HCLTECH.NS', 'HEG.NS', 'CAPLIPOINT.NS',
              #from icTracker
              'CROMPTON.NS',  'SRF.NS', 'SANOFI.NS', 'SBILIFE.NS', 'SOLARINDS.NS',
               'PIIND.NS', 'MPHASIS.NS', 'MINDTREE.NS', 'NAUKRI.NS', 'ICICIGI.NS', 'IGL.NS',
               'HEROMOTOCO.NS','ESCORTS.NS'
    
              ]


stocks =["^BSESN" , "TCS.NS",  "PAGEIND.NS", "GILLETTE.NS", "NESTLEIND.NS", "ABBOTINDIA.NS",
        'INFY.NS'
        ]




start = "2015-01-01"
end = "2019-12-30"
title = "Using quarterly stock price"
industry = "Small Cap"

# data_exp = 100
cl_price = pd.DataFrame()
for ticker in stocks:    
        cl_price[ticker] = yf.download(ticker, start, end)["Adj Close"]
cl_price.isna().any()
# org_data = cl_price.copy()
            
cl_price = cl_price[cl_price.notna()]

cl_price.isna().any()


cl_price.to_csv("C:\\Users\\Meekey\\Documents\\CAPM\\Master_Data_top6Stocks.csv")
cl_price = pd.read_csv("C:\\Users\\Meekey\\Documents\\CAPM\\Master_Data_top6Stocks.csv")
cl_price.Date = pd.to_datetime(cl_price.Date)
cl_price = cl_price.set_index('Date').resample('M').mean()

#save BSE in market price and rest in cl_price
market_price = cl_price[stocks[0]]
cl_price = cl_price.drop([stocks[0]], axis = 1)

market_ret = market_price.pct_change().dropna()
cum_market_ret = (1 + market_ret).cumprod()

cl_price_ret = cl_price.pct_change()
cl_price_ret = cl_price_ret.dropna()
cum_cl_price_ret = (cl_price_ret + 1).cumprod()

import seaborn as sns
from scipy import stats

#Calculate Stock Beta
stk_beta ={}
stk_ex_ret ={}
rf = 0.05
def calc_stock_beta(mkt_ret, stock_ret, rf):
#    mr = mkt_ret.copy()
#    sr = stock_ret.copy()
    for ticker in stock_ret:
        stock_beta, stk_alpha = stats.linregress(mkt_ret, 
                               stock_ret[ticker])[0:2]
        stk_beta[ticker] = stock_beta
        #stk_ex_ret[ticker] = rf + stock_beta*(mkt_ret[-1] - rf)
        #print(mkt_ret[-1])
        #stk_ex_ret[ticker] = rf + stock_beta*(mkt_ret[-1] - rf)
        #plt.barh(ticker, round(stock_beta,2))

calc_stock_beta(market_ret, cl_price_ret, rf)


t = len(cum_cl_price_ret)/12
stk_annual_growth = (cum_cl_price_ret.tail(1))**(1/t) - 1
mkt_annual_growth = (cum_market_ret[-1])**(1/t) - 1
#stk_annual_growth['market'] = mkt_annual_growth
stk_annual_growth


rf = 0.0325
expected_stock_ret = {}
for ticker in stk_beta:
    expected_stock_ret[ticker] = rf + stk_beta[ticker] * (mkt_annual_growth- rf)

expected_stock_ret

stats = pd.DataFrame()
def performance_statistics(data):
#    change = (data /  data.shift()- 1).iloc[1:]
#    stats = pd.concat((change.mean(), change.std(), data.iloc[-1] / data.iloc[0] - 1.0),axis=1)
    #print(data.mean(), data.std(), (data + 1).prod()-1)
    stats = pd.concat((data.mean(), data.std(), (data + 1).prod()-1),axis=1)
   #stats = pd.concat((stock_rolling_mean, stock_rolling_var, (data + 1).prod()-1),axis=1)
    stats.columns = ['Mean return', 'Standard deviation', 'Total return']
    return stats
stats = performance_statistics(cl_price_ret)
stats


# weights = np.random.random_sample((len(cl_price_ret.columns)))
# weights = np.random.random_sample(len(cl_price_ret.columns))
# weights = weights/np.sum(weights)
weights = 1
weights = np.repeat(a= weights/len(cl_price_ret.columns) , repeats = len(cl_price_ret.columns))

annual_expected_stock_ret = {}
for ticker in stk_beta:
    annual_expected_stock_ret[ticker] = rf + stk_beta[ticker] * (mkt_annual_growth - rf)

annual_expected_stock_ret = pd.core.series.Series(annual_expected_stock_ret)
annual_expected_stock_ret

#multiply by 22 to get monthly var
matrix_cov_stock_ret = (cl_price_ret.cov()) * 12
matrix_cov_stock_ret

portfolio_var = np.dot(weights,np.dot(matrix_cov_stock_ret, weights))
portfolio_std = np.sqrt(portfolio_var)
portfolio_std

def performance_extremes(stats):
    extremes = pd.concat((stats.idxmin(),stats.min(),stats.idxmax(),stats.max()),axis=1)
    extremes.columns = ['Minimizer','Minimum','Maximizer','Maximum']
    return extremes
extremes = performance_extremes(stats)
extremes

def performance_scatter(stats, **kwargs):
    fig = figure(**kwargs)
    fig.plot_width = 900
    fig.plot_height = 400
    source = ColumnDataSource(stats)
    hover = HoverTool(tooltips=[('Symbol','@index'),
                                ('Standard deviation','@{Standard deviation}'),
                                ('Mean return','@{Mean return}')])
    fig.add_tools(hover)
    fig.circle('Standard deviation', 'Mean return', size=8, color='maroon', source=source, hover_fill_color='grey')
    fig.text('Standard deviation', 'Mean return', 'index', text_font_size='10px', x_offset=4, y_offset=-2, source=source)
    fig.xaxis.axis_label='Volatility (standard deviation)'
    fig.yaxis.axis_label='Mean return'
    fig.xaxis[0].formatter = NumeralTickFormatter(format="0.0%")
    fig.yaxis[0].formatter = NumeralTickFormatter(format="0.0%")
    return fig
show(performance_scatter(stats))

















































