import numpy as np 
import pandas as pd 

# for plotting data
# import matplotlib.pyplot as plt
from matplotlib import pylab as plt
from pylab import rcParams
rcParams['figure.figsize'] = 12,5
#import lagplot from pandas for ACF and PACF plots
from pandas.plotting import lag_plot

#ignore statsmodels warnings
import warnings
warnings.filterwarnings('ignore')

#Import Holt-Winters methods
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#Import differencing to change from non stationary to stationary
from statsmodels.tsa.statespace.tools import diff
#import stuff for ACF and PACF
import statsmodels.api as sm
from statsmodels.tsa.stattools import acovf, acf, pacf, pacf_yw, pacf_ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


#imports from sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error


# -------------
# SIMPLE FORECASTING
#import data - data until 1960, so 1961 will be the future
df = pd.read_csv('data/airline_passengers.csv', index_col='Month', parse_dates=True)
df.index.freq= 'MS'

#train and test dataset (out of 144 entries, via)
train_data = df.iloc[:109]
test_data = df.iloc[108:]

#fit the model
model = ExponentialSmoothing(train_data['Thousands of Passengers'], trend='mul', seasonal='mul', seasonal_periods=12).fit()

#true data split into the two sets
train_data['Thousands of Passengers'].plot(legend=True, label='Training')
test_data['Thousands of Passengers'].plot(legend=True, label='Test')

#forecast into the future (36 models)
test_predictions = model.forecast(36)

#compare prediction vs testing dataset to see how good it is
test_predictions.plot(legend=True, label='FCST')
# plt.show()

#evaluate errors
#MEA
test_description = test_data.describe()
# print(test_description)
std_dev = test_data['Thousands of Passengers'].std()
print(std_dev)
mea = mean_absolute_error(test_predictions, test_data)
print(mea)
print("MEA is the following percentage of one standard dev: ")
print((mea/std_dev)*100)

print("----")
#MSE and RMSE
mse = mean_squared_error(test_predictions, test_data)
rmse = np.sqrt(mse)
print(rmse)
print("RMSE is the following percentage of one standard dev: ")
print((rmse/std_dev)*100)


#now run final forecasting (if testing above deemed acceptable)
final_model = ExponentialSmoothing(df['Thousands of Passengers'], trend='mul', seasonal='mul', seasonal_periods=12).fit()
final_predictions = final_model.forecast(36)

#compare set with predictions
df['Thousands of Passengers'].plot(legend=True, label='data')
final_predictions.plot(legend=True, label='final prediction')
# plt.show()


# ------- stationarity
#transform non stationary into stationary
df2 = pd.read_csv('data/samples.csv', index_col=0, parse_dates=True)
#subtract the time series to itself, shifted by one day
df2['b'] - df2['b'].shift(1)
#or via statsmodel...
diff(df2['b'], k_diff=1)


# ------- ACF and PACF
#non stationary => df
plot_acf(df, lags=40)
# plt.show()

# stationary
df3 = pd.read_csv('data/DailyTotalFemaleBirths.csv', index_col='Date', parse_dates=True)
df3.index.freq='D'

plot_acf(df3, lags=40)
plot_pacf(df3, lags=40)
plt.show()