import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

#ignore statsmodels warnings
import warnings
warnings.filterwarnings('ignore')

#import stuff for ARIMA models
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
#import testing tools
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
#evaluation tools
from statsmodels.tools.eval_measures import rmse

#import pmdarima
from pmdarima import auto_arima


#------------------------------------------


#load datasets (from FRED)
# ----------
df = pd.read_csv('data/co2_mm_mlo.csv')
#create datetime index from existing info, which is independent columns with year and month
df['date'] = pd.to_datetime({'year':df['year'], 'month':df['month'], 'day':1})   
df.set_index('date', inplace=True)
#remove columns which are not needed
df.drop('year', axis=1, inplace=True) 
df.drop('month', axis=1, inplace=True) 
df.drop('decimal_date', axis=1, inplace=True) 
df.drop('average', axis=1, inplace=True) 
# print(df.info)


#first, show the data
# ----------
# df['interpolated'].plot()
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

# adf_run = adf_test(df['interpolated'])
# => data is NON-stationary ... we need ARIMA!

# then, run ETS decomposition test on seasonality
ets_test = seasonal_decompose(df['interpolated'], model='add')
ets_test.plot()
plt.show()
#seasonal model needed, it would be SARIMA (but we ignore it for this exercise)



#third, run auto_arima to find out order
# ----------
# summary = auto_arima(df['interpolated'], seasonal=True, m=12, error_action="ignore").summary()
# print(summary)
#=> we find that the best model is SARIMAX(0,1,3) x (1,0,1,12)


# #fourth, split data set (total lines 729)
# # ----------
train = df.iloc[:717]
test = df.iloc[717:]


#fifth, run SARIMA train_model with the order determined by auto_arima 
# ----------
train_model = SARIMAX(train['interpolated'], order=(0,1,3), seasonal_order=(1,0,1,12)).fit()
# print(train_model.summary())



#sixth, test predictions vs test set
# ----------
start = len(train)
end = len(train) + len(test) - 1

predictions = train_model.predict(start,end, typ='levels').rename('SARIMA predictions vs test')  

test['interpolated'].plot(legend=True)
predictions.plot(legend=True)
# plt.show()


#seventh, evaluate the model on rmse error
# ----------
error = rmse(test['interpolated'], predictions)
std = test['interpolated'].std()
error_result = error/std * 100
# print('rmse error is the following percentage out of standard dev: ')
# print(error_result)


# eight, run forecast into the future with full dataset
# # ----------
model = SARIMAX(df['interpolated'], order=(0,1,3), seasonal_order=(1,0,1,12)).fit()
forecast = model.predict(len(df),len(df)+12, typ='levels').rename('SARIMA forecast')  

df['interpolated'].plot(legend=True)
forecast.plot(legend=True)
plt.show()


