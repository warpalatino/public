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
df = pd.read_csv('data/RestaurantVisitors.csv', index_col='date', parse_dates=True)
df.index.freq='D'
df1 = df.dropna()
#change floating data into integers
cols = ['rest1', 'rest2', 'rest3', 'rest4', 'total']
for x in cols:
    df1[x] = df1[x].astype(int)
# print(df1.info)


#first, show the data
# ----------
rest_traffic = df1['total'].plot()
#let's visualize holidays on the same plot: first, identify the holiday dates, then plot the points with vertical line in the right dates
hols = df1[df1['holiday']==1].index
for day in hols:
    rest_traffic.axvline(x=day, color='orange', alpha=0.7)
# plt.show()


#second, run tests
# ----------
# adf1uller
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
        print("Strong evidence against the null hypothesis - reject null")
        print("Data is stationary")
    else:
        print("Weak evidence against the null hypothesis - accept null")
        print("Data is NON-stationary")
    
    print('*******************************')

# adf = adf_test(df1['total'])
# => data is stationary 

# then, run ETS decomposition test on seasonality
ets_test = seasonal_decompose(df1['total'], model='add')
# ets_test.plot()
# plt.show()
# => data is seasonal (weely), let's use a SARIMA



# third, run auto_arima to find out order
# ----------
# be mindful to add "exogenous" for SARIMAX, which needs to read a dataframe, not a series --> add double brackets
# summary = auto_arima(df1['total'], exogenous=df1[['holiday']], seasonal=True, m=7, error_action="ignore").summary()
# print(summary)
# => we find that the best model is SARIMAX(0,0,1) x (2,0,0,7)


# fourth, split data set (total lines 478)
# ----------
train = df1.iloc[:436]
test = df1.iloc[436:]


#fifth, run SARIMA train_model with the order determined by auto_arima 
# ----------
train_model = SARIMAX(train['total'], order=(0,0,1), seasonal_order=(2,0,0,7), enforce_invertibility=False).fit()
print(train_model.summary())
# enforce invertibility allows to keep coefficients below 1, just to avoid ValueError



#sixth, test predictions vs test set
# ----------
start = len(train)
end = len(train) + len(test) - 1

predictions = train_model.predict(start,end, exog=test[['holiday']]).rename('SARIMAX predictions vs test')  

test['total'].plot(legend=True)
predictions.plot(legend=True)
plt.show()


#seventh, evaluate the model on rmse error
# ----------
error = rmse(test['total'], predictions)
std = test['total'].std()
error_result = error/std * 100
print('rmse error is the following percentage out of standard dev: ')
print(error_result)


# eight, run forecast into the future with full dataset
# ----------
#to predict with exog, it is key to know/assume exog in the future (here we have future hols in dataset)
exog_in_future = df[478:][['holiday']] 
model = SARIMAX(df1['total'], exog=test[['holiday']], order=(0,0,1), seasonal_order=(2,0,0,7)).fit()
forecast = model.predict(len(df1),len(df1)+38, exog=exog_in_future).rename('SARIMAX forecast')  


df1['total'].plot(legend=True)
forecast.plot(legend=True)
plt.show()


