import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARMA
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts 

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
# -- redefine column names
df_comp['market_value']=df_comp.ftse
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



# review ACF and PACF (in reality is more functional to run auto_arima vs checking ACF/PACF manually, but this is for sake of example)
# ----------
sgt.plot_acf(df.market_value, zero = False, lags = 40)
plt.title("ACF for Prices", size = 20)

sgt.plot_pacf(df.market_value, lags = 40, alpha = 0.05, zero = False, method = ('ols'))
plt.title("PACF for Prices", size = 20)
plt.show()
# => we know data is non-stationary from a previous exercise (so we should not use an AR model...auto_arima and full ARIMA would help here)


# select AR model (by looking to PACF here) and iterating through more models...until LLR will stop going down
# ----------
model_ar = ARMA(df.market_value, order=(1,0)).fit()
print(model_ar.summary())
print('----------')
model_ar_4 = ARMA(df.market_value, order=(4,0)).fit()
print(model_ar_4.summary())
print('----------')
model_ar_7 = ARMA(df.market_value, order=(7,0)).fit()
print(model_ar_7.summary())
print('----------')
# => by comparing the LLR stat and AIC/BIC from models' summary we can see what is the best order ... (we would find out AR(7,0))
# => remember that auto_arima is much easier...


# compare LLR results across models
# ----------
def LLR_test(mod_1, mod_2, DF=1):
    L1 = mod_1.fit().llf
    L2 = mod_2.fit().llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR, DF).round(3)
    return p

print('LLR test 1: ', LLR_test(model_ar_1, model_ar_4, DF=3))
print('LLR test 2: ', LLR_test(model_ar_4, model_ar_7, DF=3))


# analyzing residuals
# ----------
df['res_price'] = model_ar_7.resid
df.res_price.mean()
df.res_price.var()
# -- let's see if there is any significant error that the model has missed (via ACF or PACF)
sgt.plot_acf(df.res_price, zero = False, lags = 40)
plt.title("ACF Of Residuals for Prices",size=24)
# plt.show()
# -- plotting all residuals
df.res_price[1:].plot(figsize=(20,5))
plt.title("Residuals of Prices",size=24)
# plt.show()