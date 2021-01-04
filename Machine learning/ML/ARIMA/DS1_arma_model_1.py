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
# -- redefine column names and add a new column on returns - we will be working on returns
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



# review ACF and PACF (in reality is more functional to run auto_arima vs checking ACF/PACF manually, but this is for sake of example)
# ----------
# not done here


# select ARMA model (by looking to PACF here) and iterating through more models...until LLR will stop going down
# ----------
model_ret_ar_1_ma_1 = ARMA(df.returns[1:], order=(1,1)).fit()
print(model_ret_ar_1_ma_1.summary())
print('----------')
# -- going manually through multiple iterations (not reported here) would lead to this higher order model
model_ret_ar_1_ma_3 = ARMA(df.returns[1:], order=(1,3))
print(model_ret_ar_1_ma_3.summary())
print('----------')
model_ret_ar_3_ma_2 = ARMA(df.returns[1:], order=(3,2))
print(model_ret_ar_3_ma_2.summary())
print('----------')
# => by comparing the LLR stat and AIC/BIC from models' summary we can see what is the best order ... (we would find out ARMA(3,2))
# => remember that auto_arima is much easier...



# compare LLR results across models
# ----------
def LLR_test(mod_1, mod_2, DF=1):
    L1 = mod_1.fit().llf
    L2 = mod_2.fit().llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR, DF).round(3)
    return p

print("\n ARMA(3,2): \tLL = ", model_ret_ar_3_ma_2.llf, "\tAIC = ", model_ret_ar_3_ma_2.aic)
print("\n ARMA(1,3): \tLL = ", model_ret_ar_1_ma_3.llf, "\tAIC = ", model_ret_ar_1_ma_3.aic)




# analyzing residuals
# ----------
df['res_ret_ar_3_ma_2'] = results_ret_ar_3_ma_2.resid[1:]
# -- let's see if there is any significant error that the model has missed (via ACF or PACF)
df.res_ret_ar_3_ma_2.plot(figsize = (20,5))
plt.title("Residuals of Returns", size=24)
# -- plotting all residuals
sgt.plot_acf(df.res_ret_ar_3_ma_2[2:], zero = False, lags = 40)
plt.title("ACF Of Residuals for Returns",size=24)
# plt.show()

