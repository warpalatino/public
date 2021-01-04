import numpy as np 
import pandas as pd 

# for plotting data
# import matplotlib.pyplot as plt
from matplotlib import pylab as plt
from pylab import rcParams
rcParams['figure.figsize'] = 12,5

#imports from statsmodel
#Import Hodrick-Prescott filter
from statsmodels.tsa.filters.hp_filter import hpfilter
#Import ETS decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
#Import Holt-Winters methods
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# --- HP filter
#import given data
# ----
df = pd.read_csv('data/macrodata.csv', index_col=0, parse_dates=True)
# print(df.head())

#plot data
# ----
# gdp_plot = df['realgdp'].plot(figsize=(12,5))
# plt.show()

gdp_cycle, gdp_trend = hpfilter(df['realgdp'], lamb=1600)
df['gdp_trend'] = gdp_trend
df['gdp_cycle'] = gdp_cycle

# df[['gdp_trend', 'gdp_cycle']].plot()
# df['gdp_trend'].plot()
# df['gdp_cycle'].plot()
# plt.show()


# --- ETS decomposition
#import given data
airline = pd.read_csv('data/airline_passengers.csv', index_col='Month', parse_dates=True)
airline.dropna()
# print(airline)
# airline.plot()
# plt.show()
ets_model = seasonal_decompose(airline['Thousands of Passengers'], model='multiplicative')
#read the different components
ets_model.trend
ets_model.seasonal
ets_model .resid
#plot the components
# ets_model.plot()
# plt.show()


# --- EWMA model
#create a simple MA
airline['6mts-simple-MA'] = airline['Thousands of Passengers'].rolling(window=6).mean()
airline['12mts-simple-MA'] = airline['Thousands of Passengers'].rolling(window=12).mean()
# airline.plot()
# plt.show()

#create a EWMA (more recent points have more weight than MA)
airline['EWMA-12'] = airline['Thousands of Passengers'].ewm(span=12).mean()
airline[['Thousands of Passengers', 'EWMA-12']].plot()
plt.show()


# --- Holt winters
#adding frequency to our airline data - month start
airline.index.freq= 'MS'
#formulas related to the stats span and alpha from EWMA
span = 12
alpha = 2 / (span+1)

#1) simple exponential smoothing
holt_winters = SimpleExpSmoothing(airline['Thousands of Passengers']).fit(smoothing_level=alpha, optimized=False)
df['simple_smoothing'] = holt_winters.fittedvalues

#2) double exponential smoothing
holt_winters2 = ExponentialSmoothing(airline['Thousands of Passengers'], trend='add').fit()
df['double_exp_smoothing'] = holt_winters2.fittedvalues

#3) triple exponential smoothing
holt_winters3 = ExponentialSmoothing(airline['Thousands of Passengers'], trend='mul', seasonal='mul', seasonal_periods=12).fit()
df['triple_exp_smoothing'] = holt_winters3.fittedvalues