import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.distributions import chi2
from math import sqrt

import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA

from arch import arch_model


# ------------------------

# load data 
# ----------
raw_csv_data = pd.read_csv("../data/Index2018.csv") 
df_comp=raw_csv_data.copy()
# -- make the index a datetime object
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index("date", inplace=True)
df_comp=df_comp.asfreq('b')
# -- fill na values
df_comp=df_comp.fillna(method='ffill')
# -- redefine column names and add a new column on returns - we will be working on returns
df_comp['market_value']=df_comp.ftse
# -- let's delete redundant data
del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']



# split dataset (on straight data = prices)
# ----------
size = int(len(df_comp) * 0.8)
df = df_comp.iloc[:size]
df_test = df_comp.iloc[size:]



# manipulate train dataset for ARCH
# ----------
# -- create returns 
df['returns'] = df.market_value.pct_change(1)*100
# -- create squared returns to measure vol
df['sq_returns'] = df.returns.mul(df.returns)




# observe data for both returns and vol of returns
# ----------
df.returns.plot(figsize=(20,5))
plt.title("Returns", size = 24)
df.sq_returns.plot(figsize=(20,5))
plt.title("Volatility", size = 24)
plt.show()



# run GARCH(1,1) model 
# ----------
model_garch_11 = arch_model(df.returns[1:], mean = "Constant", vol = "GARCH", p = 1, q = 1).fit(update_freq = 5)
print(model_garch_11.summary())




