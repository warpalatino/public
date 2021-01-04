import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse


#------------------------------------------


#load datasets (from FRED)
# ----------
df = pd.read_csv('data/M2SLMoneyStock.csv', index_col=0, parse_dates=True)
df.index.freq='MS'

sp = pd.read_csv('data/PCEPersonalSpending.csv', index_col=0, parse_dates=True)
df.index.freq='MS'

#combine them into one dataframe
df = df.join(sp)
df = df.dropna()



#first, show the data
# ----------
df.plot()
plt.show()



#second, run tests
# ----------
#adf1uller
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

adf1 = adf_test(df['Money'])
# => data is NON-stationary 
adf2 = adf_test(df['Spending'])
# => data is NON-stationary 




# second-bis, run differencing if data NON - stationary
# ----------
# pandas .diff method will run differencing in the entire dataframe
df_transformed = df.diff()
# repeat the adf tests to check if one differencing ha solved it 
adf3 = adf_test(df_transformed['Money'])
adf4 = adf_test(df_transformed['Spending'])
# as one of the two series is not yet stationary, we run another differencing round on both
df_transformed = df_transformed.diff().dropna()
adf5 = adf_test(df_transformed['Money'])
adf6 = adf_test(df_transformed['Spending'])




# third, split data set 
# ----------
nobs = 12   #number of observations for the test set
train = df_transformed.iloc[:-nobs] #start = beginning of dataframe, or 0, to -12 from the end
test = df_transformed.iloc[-nobs:] #start at -12 from the end to go to the end


#fourth, run the grid search manually via a loop
# ----------
model = VAR(train)
lags = [1,2,3,4,5,6,7]

for p in lags:
    results = model.fit(p)
    print(f'order {p}')
    print(f'AIC {results.aic}')
    print('--------')

#best order is p=5



#fifth, run the model with the chosen order p
# ----------
p = 5
results = model.fit(p)
# print(results.summary())



#sixth, run the temporary forecast vs test set
# ----------
# forecast via .forecast: 
# (1) y = nparray p * k, where k is the series
# (2) the next steps to predict in the future (= to train)
lagged_values = train.values[-p:]    #this returns the dataframe as nparray for the last p lines in p * k shape
z = results.forecast(y=lagged_values, steps=nobs)
#transform z back into a dataframe
test_start = '2015-01-01'
index = pd.date_range(test_start, periods=nobs, freq='MS')
df_forecast = pd.DataFrame(data=z, index=index, columns=['Money_diff2x', 'Spending_diff2x'])


# sixth-bis, reverse the differencing to make values turn normal, if needed
# ----------
# we transform the differences by taking the cumulative sum of a column and adding it to a one line difference to revert it 
df_forecast['Money_diff1x'] = (df['Money'].iloc[-nobs-1]-df['Money'].iloc[-nobs-2]) + df_forecast['Money_diff2x'].cumsum()
df_forecast['Money_Fcst'] = df['Money'].iloc[-nobs-1] + df_forecast['Money_diff1x'].cumsum()

df_forecast['Spending_diff1x'] = (df['Spending'].iloc[-nobs-1]-df['Spending'].iloc[-nobs-2]) + df_forecast['Spending_diff2x'].cumsum()
df_forecast['Spending_Fcst'] = df['Spending'].iloc[-nobs-1] + df_forecast['Spending_diff1x'].cumsum()



#seventh, plot temp forecast vs test
# ----------
#define the original test range, then plot the forecast
test_range = df[-nobs:]
test_range[['Money', 'Spending']].plot()
df_forecast[['Money_Fcst', 'Spending_Fcst']].plot()
# test_range[['Money']].plot(legend=True)
# df_forecast[['Money_Fcst']].plot(legend=True)
# test_range[['Spending']].plot(legend=True)
# df_forecast[['Spending_Fcst']].plot(legend=True)
plt.show()



#eight, measure the error via rmse
# ----------
error1 = rmse(test_range['Money'], df_forecast['Money_Fcst'])
error2 = rmse(test_range['Spending'], df_forecast['Spending_Fcst'])
std1 = test_range['Money'].std()
std2= test_range['Spending'].std()
error_result1 = error1/std1 * 100
error_result2 = error2/std2 * 100
print('rmse error is the following percentage out of standard dev: ')
print(error_result1)
print(error_result2)


# ninth, run full forecast
# ----------


