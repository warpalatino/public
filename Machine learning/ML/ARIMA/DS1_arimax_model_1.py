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




# review ACF and PACF (in reality is more functional to run auto_arima vs checking ACF/PACF manually, but this is for sake of example)
# ----------
# not done here



# run ARIMAX model using S&P500 vakues as exogenous factor to explain FTSE values
# ----------
model_arimax_111 = ARIMA(df.market_value, order=(1,1,1), exog=df.spx).fit()
print(model_arimax_111.summary())
print('----------')





# analyzing residuals
# ----------
df['residuals_model_arimax_111'] = model_arimax_111.resid.iloc[:]
sgt.plot_acf(residuals_model_arimax_111[1:], zero = False, lags = 40)
plt.title("ACF Of Residuals for ARIMAX(1,1,1)",size=20)
plt.show()

