import numpy as np 
import pandas as pd 

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

#import pmdarima
from pmdarima import auto_arima


#------------------------------------------


#load datasets
# ----------
df1 = pd.read_csv('data/DailyTotalFemaleBirths.csv', index_col='Date', parse_dates=True)
df1.index.freq='D'
df1 = df1[:120] #let's grab only the first 120 days of data

#first, show the data
# ----------
df1['Births'].plot()
# plt.show()


#second, run tests on stationarity
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

# adf_run1 = adf_test(df1['Births'])
# print(adf_run1)
# => data is stationary

#third, run auto_arima to find out order
summary1 = auto_arima(df1['Births'], seasonal=False).summary()
# print(summary1)
#it finds out that ARMA (0,0) is the best model - only constant?
#the online class finds ARMA(2,2)


#fourth, split data set
# ----------
train = df1.iloc[:90]
test = df1.iloc[90:]


#fifth, run ARMA model with auto_arima (or modified) order
# ----------
model = ARMA(train['Births'], order=(2,2)).fit()
print(model.summary())



#sixth, run predictions
# ----------
start = len(train)
end = len(train) + len(test) - 1

predictions = model.predict(start,end).rename('ARMA(2,2) predictions')


#seventh, show prediction accuracy
# ----------
test['Births'].plot(legend=True)
predictions.plot(legend=True)
plt.show()


#eight, extract data from model
# ----------
model_mean = predictions.mean()