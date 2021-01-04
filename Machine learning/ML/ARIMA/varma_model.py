import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse

#import pmdarima
from pmdarima import auto_arima


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



# third, run auto_arima to find out order
# ----------
# we expand max iterations from 50 (standard) to a higher limit to avoid errors
summary1 = auto_arima(df['Money'], maxiter=500, error_action="ignore").summary()
summary2 = auto_arima(df['Spending'], maxiter=500, error_action="ignore").summary()
print(summary1)
print(summary2)
# => we find that the best model is:
# => VARMA(1,2,2) for Money
# => VARMA(1,1,2) for Spending



# third-bis, run differencing as data is NON - stationary (see that we have an integration of 1 or 2 needed from auto_arima suggestion)
# ----------
# pandas .diff method will run differencing in the entire dataframe, twice in one line
df_transformed = df.diff().diff()
df_transformed = df_transformed.dropna()




# fourth, split data set 
# ----------
nobs = 12   #number of observations for the test set
train = df_transformed.iloc[:-nobs] #start = beginning of dataframe, or 0, to -12 from the end
test = df_transformed.iloc[-nobs:] #start at -12 from the end to go to the end


#fifth, fit the VARMA model
# ----------
# VARMAX needs as input the suggested p and q (no d) and a trend c which is the standard = linear trend
# fit requires both maxiter and disp, where disp is display=False => display less information
model = VARMAX(train, order=(1,2), trend='c').fit(maxiter=500, disp=False)
print(model.summary())



#sixth, run a prediction vs test set
# ----------
df_forecast = model.forecast(nobs)



# sixth-bis, reverse the differencing to make values turn normal and understandable
# ----------
# we transform the differences by taking the cumulative sum of a column and adding it to a one line difference to revert it 
df_forecast['Money_diff1x'] = (df['Money'].iloc[-nobs-1]-df['Money'].iloc[-nobs-2]) + df_forecast['Money_diff2x'].cumsum()
df_forecast['Money_Fcst'] = df['Money'].iloc[-nobs-1] + df_forecast['Money_diff1x'].cumsum()

df_forecast['Spending_diff1x'] = (df['Spending'].iloc[-nobs-1]-df['Spending'].iloc[-nobs-2]) + df_forecast['Spending_diff2x'].cumsum()
df_forecast['Spending_Fcst'] = df['Spending'].iloc[-nobs-1] + df_forecast['Spending_diff1x'].cumsum()



#seventh, plot temp forecast vs test
# ----------
#as test is related to differences values, we want to show the original dataframe for the test portion, vs the prediction
df['Money'][-nobs:].plot(legend=True)
df_forecast['Money_Fcst'].plot(legend=True)
plt.show()

df['Spending'][-nobs:].plot(legend=True)
df_forecast['Spending_Fcst'].plot()
plt.show()



#eight, measure the error via rmse
# ----------
error1 = rmse(df['Money'][-nobs:], df_forecast['Money_Fcst'])
error2 = rmse(df['Spending'][-nobs:], df_forecast['Spending_Fcst'])
std1 = df['Money'][-nobs:].std()
std2= df['Spending'][-nobs:].std()
error_result1 = error1/std1 * 100
error_result2 = error2/std2 * 100
print('rmse error is the following percentage out of standard dev: ')
print(error_result1)
print(error_result2)


# ninth, run full forecast
# ----------


