import numpy as np 
import pandas as pd 

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#for plotting data
import matplotlib.pyplot as plt
from matplotlib import pylab as plt
from pylab import rcParams
rcParams['figure.figsize'] = 12,5

#ignore statsmodels warnings
import warnings
warnings.filterwarnings('ignore')

#import stuff for ARIMA models
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA, ARIMA, ARMAResults, ARIMAResults
#import testing tools
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

#import pmdarima
from pmdarima import auto_arima


#------------------------------------------


#load datasets (from FRED)
# ----------
df2 = pd.read_csv('data/TradeInventories.csv', index_col='Date', parse_dates=True)
df2.index.freq= 'MS'
df2.index = pd.to_datetime(df2.index, format="%Y-%m-%d")
# print(df2.index)



#first, show the data
# ----------
df2.plot()
# plt.show()


#second, run tests
# ----------
# adfuller
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

adf_run = adf_test(df2['Inventories'])
# print(adf_run)
# => high p-value, data is NON-stationary ... we need ARIMA!

# then, run ETS decomposition test on seasonality
ets_test = seasonal_decompose(df2['Inventories'], model='add')
ets_test.plot()
# plt.show()
#seasonal model needed, it would be SARIMA (but we ignore it for this exercise)


# ** bonus_step, check ACF and PACF out of curiosity to see what they say
# ----------
plot_acf(df2['Inventories'], lags=40)
plot_pacf(df2['Inventories'], lags=40)
# plt.show()


#third, run auto_arima to find out order
# ----------
summary = auto_arima(df2['Inventories'], seasonal=False, error_action="ignore").summary()
# print(summary)
#the online class finds ARIMA(1,1,1)


#fourth, split data set
# ----------
train = df2.iloc[:252]
test = df2.iloc[252:]


#fifth, run ARIMA model with auto_arima (or modified) order
# ----------
train_model = ARIMA(train['Inventories'], order=(1,1,1)).fit()
# print(train_model.summary())



#sixth, run predictions on full set into the future
# ----------
start = len(train)
end = len(train) + len(test) - 1

model = ARIMA(df2['Inventories'], order=(1,1,1)).fit()
predictions = model.predict(start=len(df2),end=len(df2)+12, typ='levels').rename('ARIMA(1,1,1) full predictions')  #type allows to choose difference in variables = linear, or = levels for the original series
# print(predictions.index)


#seventh, show prediction accuracy
# ----------
df2['Inventories'].plot(legend=True)
predictions.plot(legend=True)
plt.show()


#eight, extract data from model
# ----------
# model_mean = predictions.mean()