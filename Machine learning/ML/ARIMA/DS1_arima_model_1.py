import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats.distributions import chi2
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
# -- redefine column names and add a new column on returns - we will be working on returns
df_comp['market_value']=df_comp.ftse




# split dataset (on straight data = prices)
# ----------
size = int(len(df_comp) * 0.8)
df = df_comp.iloc[:size]
df_test = df_comp.iloc[size:]
# -- creating returns column from train dataset
df['returns'] = df.market_value.pct_change(1)*100



# review ACF and PACF (in reality is more functional to run auto_arima vs checking ACF/PACF manually, but this is for sake of example)
# ----------
# not done here



# select ARMA model (by looking to PACF here) and iterating through more models
# ----------
model_arima_111 = ARIMA(df.market_value, order=(1,1,1)).fit()
print(model_arima_111.summary())
print('----------')
model_arima_511 = ARIMA(df.market_value, order=(5,1,1)).fit()
print(model_arima_511.summary())
print('----------')




# compare LLR results across models to see which model is best
# ----------
def LLR_test(mod_1, mod_2, DF=1):
    L1 = mod_1.fit().llf
    L2 = mod_2.fit().llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR, DF).round(3)
    return p

print("\nLLR test p-value = " + str(LLR_test(model_arima_111, model_arima_511, DF = 4)))




# analyzing residuals
# ----------
df['residuals_model_arima_111'] = model_arima_111.resid.iloc[:]
sgt.plot_acf(residuals_model_arima_111[1:], zero = False, lags = 40)
plt.title("ACF Of Residuals for ARIMA(1,1,1)",size=20)
df['residuals_model_arima_511'] = model_arima_511.resid.iloc[:]
sgt.plot_acf(residuals_model_arima_511[1:], zero = False, lags = 40)
plt.title("ACF Of Residuals for ARIMA(5,1,1)",size=20)
plt.show()

