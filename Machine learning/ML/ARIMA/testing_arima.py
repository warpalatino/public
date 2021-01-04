import numpy as np 
import pandas as pd 

# for plotting data
# import matplotlib.pyplot as plt
from matplotlib import pylab as plt
from pylab import rcParams
rcParams['figure.figsize'] = 12,5

#ignore statsmodels warnings
import warnings
warnings.filterwarnings('ignore')

#import tests from statsmodel
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.graphics.tsaplots import month_plot, quarter_plot

# ---------------------

#load seasonal dataset (=> trend, means non - stationary)
df1 = pd.read_csv('data/airline_passengers.csv', index_col='Month', parse_dates=True)
df1.index.freq= 'MS'

#load non-seasonal dataset (=> no trend, means stationary)
df2 = pd.read_csv('data/DailyTotalFemaleBirths.csv', index_col='Date', parse_dates=True)
df2.index.freq='D'


#testing

#augmented Dickey-Fuller test

# ||| ---> made function to properly read an ADF report <--- |||
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
adf_run1 = adf_test(df1['Thousands of Passengers'])
print(adf_run1)

adf_run2 = adf_test(df2['Births'])
print(adf_run2)


#Granger causality test

#load a new dataset
df3 = pd.read_csv('data/samples.csv', index_col=0, parse_dates=True)
df3.index.freq='MS'

# run the test after inputing a guessed maxlag
granger1 = grangercausalitytests(df3[['a', 'd']], maxlag=2)


# seasonality check
month_plot(df1['Thousands of Passengers'])
#if we have quarterly data, we can resample the data
quarterly_data = df1['Thousands of Passengers'].resample(rule='Q').mean()
quarter_plot(quarterly_data)
# plt.show()
