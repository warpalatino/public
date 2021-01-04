import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import statsmodels.graphics.tsaplots as sgt 
import statsmodels.tsa.stattools as sts 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


# -------------------


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
df_comp['market_value']=df_comp.spx
# -- delete redundant data
del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']


# split dataset
# ----------
size = int(len(df_comp) * 0.8)
df = df_comp.iloc[:size]
df_test = df_comp.iloc[size:]



# test for stationarity
# ----------

def adf_test(series,title=''):
    print('*******************************')
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis - reject")
        print("Data is stationary")
    else:
        print("Weak evidence against the null hypothesis - do not reject")
        print("Data is NON-stationary")
    
    print('*******************************')
# --- |||

# now run the test on the specific series 
adf_run = adf_test(df['market_value'])
print(adf_run)



# test for seasonality
# ----------
# --- ETS decomposition
ets_model = seasonal_decompose(df['market_value'], model='multiplicative')
# -- plot the components
ets_model.plot()
# plt.show()



# show ACF and PACF charts
# ----------
acf = sgt.plot_acf(df.market_value, lags = 40, zero = False)
pacf = sgt.plot_pacf(df.market_value, lags = 40, zero = False, method = ('ols'))
plt.title("ACF S&P", size = 24)
plt.title("PACF S&P", size = 24)
plt.show()