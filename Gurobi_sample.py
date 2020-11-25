# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:58:58 2020

@author: Meekey
"""
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



%matplotlib inline

#stocks =["^BSESN" , "MARICO.NS", "QUESS.NS", "CAPLIPOINT.NS",
#         "HAVELLS.NS", "MANAPPURAM.NS", "MUTHOOTFIN.NS", "GRANULES.NS",
#         "IGL.NS", "CHOLAFIN.NS"
#        ]

stocks =["RALLIS.NS", "SONATSOFTW.NS", "AUROPHARMA.NS",
         "VINATIORGA.NS","RELAXO.NS"
        ]


#cl_price = pd.read_csv("C:\\Users\\Meekey\\Documents\\CAPM\\low_perf.csv")
#cl_price = cl_price.dropna()

cl_price = pd.DataFrame()
for ticker in stocks:
    cl_price[ticker] = yf.download(ticker, "2019-08-30", "2020-08-30",
            interval = '1mo')["Adj Close"]

#cl_price.to_csv("C:\\Users\\Meekey\\Documents\\CAPM\\temp.csv")
#
#cl_price = pd.read_csv("C:\\Users\\Meekey\\Documents\\CAPM\\temp.csv")
#cl_price = cl_price.dropna()
#
#
#cl_price.Date = pd.to_datetime(cl_price.Date)
#cl_price = cl_price.set_index('Date').resample('M').mean()



cl_price_ret = cl_price.pct_change()
cl_price_ret = cl_price_ret.dropna()
syms = cl_price_ret.columns

stats = pd.concat((cl_price_ret.mean(), cl_price_ret.std(), (cl_price_ret+1).mean()-1), axis = 1)
stats.columns = ["Mean_Ret", "Volatility", "Total_Ret"]
stats

extremes = pd.concat((stats.idxmin(), stats.min(), stats.idxmax(), stats.max()), axis = 1)
extremes.columns = ["Minimizer", "Minimum", "Maximizer", "Maximum"]
extremes

syms = stocks
m = Model("portfolio")
portvars = [m.addVar(name = symb, lb = 0.0) for symb in syms]
portvars = pd.Series(portvars, index = syms)
portfolio = pd.DataFrame({'Variable' : portvars})
m.update()
portfolio

#total budget
p_total = portvars.sum()

#portfolio mean return
p_return = stats['Mean_Ret'].dot(portvars)

#squqred volatility of portfolio
Sigma = (cl_price_ret/cl_price_ret.shift()).cov()
p_risk = Sigma.dot(portvars).dot(portvars)

#Set objective function : minimize risk
m.setObjective(p_risk, GRB.MINIMIZE)
#fix budget
m.addConstr(p_total, GRB.EQUAL, 1)
#m.addConstr(p_total == 1)
#select simplex algorithm
m.setParam('Method', 1)

m.optimize()
portfolio['Minimum Risk'] = portvars.apply(lambda x:x.getAttr('x'))

#Add return target
ret50 = 0.5 * extremes.loc['Mean_Ret','Maximum']
fixreturn = m.addConstr(p_return == ret50)
m.optimize()

portfolio['50% Max Risk'] = portvars.apply(lambda x:x.getAttr('x'))


portfolio['Maximum Risk'] = [1.0 if x == 'SONATSOFTW.NS' else 0.0 for x in syms]
portfolio['Maximum Risk']


fig = figure(x_axis_type = 'datetime', y_axis_type = 'log', tools = "pan, box_zoom, reset")
pgrowth = cl_price_ret.dot(portfolio.loc[:,'Minimum Risk':'Maximum Risk'])
pgrowth = (pgrowth + 1).cumprod()

for (t, c) in zip(['Minimum Risk', '50% Max Risk', 'Maximum Risk'],['red', 'green', 'blue']):
    fig.line(pgrowth.index, pgrowth[t], legend = t, color = c)

fig.y_range = Range1d(1.0, 1.1 * pgrowth.max().max())
show(fig)
#fig.legend.orientation = "top_left"


#squared volatility for portfolio
#p_risk = Sigma.dot(portvars).dot(portvars)

#corr = (cl_price_ret/cl_price_ret.shift()).corr()

#Sigma = (cl_price_ret/cl_price_ret.shift()).cov()


cvxopt.solvers.options['abstol'] = cvxopt.solvers.options['reltol'] = cvxopt.solvers.options['feastol'] = 1e-8
n = cl_price_ret.shape[1]
P = 2 * cvxopt.matrix(Sigma.values)
q = cvxopt.matrix(np.zeros(n))
G = cvxopt.matrix(-np.eye(n,n))
h = cvxopt.matrix(np.zeros(n))
A = cvxopt.matrix(np.ones((1,n)))
b = cvxopt.matrix(np.ones(1))
solution = cvxopt.solvers.qp(P, q, G, h, A, b)
sol_x = np.array(solution['x'])[:,0]
sol_x *= (sol_x > 1e-4)
min_risk = sol_x / sol_x.sum()












fig = figure(tools = "pan, box_zoom, reset")
source = ColumnDataSource(stats)
hover = HoverTool(tooltips = [('Symbol', '@index'), ('Volatility', '@Volatility'), ('Mean_Ret', '@Mean_Ret)])
fig.add_tools(hover)

fig.circle('Volatility', 'Mean_Ret', size = 5, color = 'maroon', source = source)
fig.text('Volatility', 'Mean_Ret',  text_font_size = '10px',
         x_offset = 3, y_offset = -2, source = source)

show(fig)

Sigma = cl_price_ret.cov()
Sigma.loc[['MARICO.NS', '^BSESN', 'CHOLAFIN.NS'], ['MARICO.NS', '^BSESN', 'CHOLAFIN.NS']]


m = Model("portfolio")

portvars = [m.addVar(name = symb, lb = 0.0) from symb in stats]
portvars = pd.Series(portvars, index = syms)
portfolio.pd.DataFrame({'Variables' : portvars})






