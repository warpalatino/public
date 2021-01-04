import pandas as pd 
from fbprophet import Prophet
import matplotlib.pyplot as plt


# ---------------------------


# load and inspect data, which needs to be in a very specific format
# ------
df = pd.read_csv('data/BeerWineLiquor.csv')
# df.rename(columns={'date':'ds', 'beer':'y'}, inplace=True)
df.columns = ['ds','y']
df['ds'] = pd.to_datetime(df['ds'])
# df.index.freq='MS'
# print(df.head())



# create and fit the model 
# ------
# Prophet has many possible options to go inside the parenthesis, check docs
m = Prophet()
# first example ever here, so we will not split the databse in two parts (yet)
m.fit(df)


# predict the future
# ------
future = m.make_future_dataframe(periods=24, freq = 'MS')     # future is the new dataframe including the forecasted future dates
# print('--------_> df tail')
# print(df.tail())
# print('--------_> future tail')
# print(future.tail())
# -- now predict the values and add them to the future dataframe
forecast = m.predict(future)
# print(forecast.head())
# -- grab specific model columns
# print(forecast.colums)
forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']].tail()


# plot the forecast via prophet plotting
# ------
m.plot(forecast)
plt.show()
# -- we can plot components directly
m.plot_components(forecast)


# evaluate prophet precision
# ------
# (see file 2)