import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARMA
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts 

from math import sqrt

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
# -- redefine column names and add a new column on returns
df_comp['market_value']=df_comp.ftse
df['returns'] = df.market_value.pct_change(1)*100
df = df.iloc[1:]
# -- delete redundant data
del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']



# split dataset (on straight data = prices)
# ----------
size = int(len(df_comp) * 0.8)
df = df_comp.iloc[:size]
df_test = df_comp.iloc[size:]



# review ACF (in reality is more functional to run auto_arima vs checking ACF/PACF manually, but this is for sake of example)
# ----------
sgt.plot_acf(df.returns[1:], zero = False, lags = 40)
plt.title("ACF for Returns", size=24)



# select MA model (by looking to PACF here) and iterating through more models...until LLR will stop going down
# ----------
model_ret_ma_8 = ARMA(df.returns[1:], order=[0,8]).fit()
print(model_ret_ma_8.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ret_ma_7, model_ret_ma_8)))
# => by comparing the LLR stat and AIC/BIC from models' summary we can see what is the best order ... (we would find out MA(0,8))
# => remember that auto_arima is much easier...



# compare LLR results across models
# ----------
def LLR_test(mod_1, mod_2, DF=1):
    L1 = mod_1.fit().llf
    L2 = mod_2.fit().llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR, DF).round(3)
    return p

# -- comparing what looked like the best models (through iterations done only in online class)
# -- We found a simpler MA(6) model and a more complex MA(8) model both performing well, where MA(8) was better. 
print('LLR test: ',LLR_test(model_ret_ma_6, model_ret_ma_8, DF = 2))




# analyzing residuals
# ----------
df['res_ret_ma_8'] = model_ret_ma_8.resid[1:]
print("The mean of the residuals is " + str(round(df.res_ret_ma_8.mean(),3)) + "\nThe variance of the residuals is " + str(round(df.res_ret_ma_8.var(),3)))
# -- let's see if there is any significant error that the model has missed (via ACF or PACF)
sgt.plot_acf(df.res_ret_ma_8[2:], zero = False, lags = 40)
plt.title("ACF Of Residuals for Returns",size=24)
# -- plotting all residuals
df.res_ret_ma_8[1:].plot(figsize = (20,5))
plt.title("Residuals of Returns", size = 24)
# plt.show()

