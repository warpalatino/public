import pandas as pd 
import matplotlib.pyplot as plt

#import prophet and prophet's diagnostics
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric

from statsmodels.tools.eval_measures import rmse

# ---------------------------


# load and inspect data, which needs to be in a very specific format with columns 'ds' and 'y'
# ------
df = pd.read_csv('data/Miles_traveled.csv')
# df.rename(columns={'date':'ds', 'beer':'y'}, inplace=True)
df.columns = ['ds','y']
df['ds'] = pd.to_datetime(df['ds'])
# df.index.freq='MS'
# print(df.head())



# split dataset 
# ------
print('Length of dataseries is: ')
print(len(df))
# -- monthly data, 588 datapoints; keep last year, or broadly 2% for testing (i.e. data from last year, 12 values)
split_point = 0.98
limit = round(len(df) * split_point)
test_range = round(len(df) * (1 - split_point))
# print(limit)
# print(test_range)
train = df.iloc[:limit]
test = df.iloc[limit:]
# print(train)
# print(test)



# create and fit the model 
# ------
# Prophet has many possible options to go inside the parenthesis, check docs
m = Prophet()
# first example ever here, so we will not split the databse in two parts (yet)
m.fit(df)



# predict the future
# ------
future = m.make_future_dataframe(periods=12, freq = 'MS')     # future is the new dataframe including the forecasted future dates
# -- now predict the values and add them to the future dataframe
forecast = m.predict(future)
# print(forecast.head())
# -- grab specific model columns to review them
# print(forecast.colums)
# forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']].tail()


# plot the forecast via prophet plotting
# ------
m.plot(forecast)
# -- we can plot components directly
m.plot_components(forecast)
# plt.show()


# option to customize plot to directly compare to test dataset
# ------
custom_plot = forecast.plot(x='ds', y='yhat', label='Predictions', legend=True)
test.plot(x='ds', y='y', label='Test dataset', legend=True, ax=custom_plot )
plt.show()



# evaluate prophet's model precision, method #1 (manual/standard)
# ------
predictions = forecast.iloc[-test_range:]['yhat']
# print(predictions)
# print(test['y'])
error = rmse(test['y'], predictions)
std = test['y'].std()
error_result = error/std * 100
print('rmse error is the following percentage out of standard dev: ')
print(error_result)


# evaluate prophet's model precision, method #2 (prophet's own diagnostics)
# ------
# -- define initial training period (typically 3x the horizon)
initial = 5 * 365 
# for this example will have to be 5 years in days, as needed by prophet, and exactly in the format below
initial = str(initial) + ' days'

# -- define spacing between cutoff dates
period = 5 * 365 
# for this example will have to be 5 years in days, as needed by prophet, and exactly in the format below
period = str(period) + ' days'

# -- define forecast horizon to cross-check
horizon = 365
horizon = str(horizon) + ' days'

# -- with the above data set in the right format, now we re ready for the cross validation procedure
# -- the output will be a dataframe with forecasted values vs true test values 
# -- a forecast is made for every observed point between cutoff and cutoff + horizon
df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon)
# verify key stats for the cross validation dataframe
print(performance_metrics(df_cv))
plot_cross_validation_metric(df_cv, metric='rmse')
plt.show()