import pandas as pd 
import matplotlib.pyplot as plt

#import prophet and trend tools
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

from statsmodels.tools.eval_measures import rmse

# ---------------------------


# load and inspect data, with specific trend changes
# ------
df = pd.read_csv('data/HospitalityEmployees.csv')
df.columns = ['ds','y']
df['ds'] = pd.to_datetime(df['ds'])
# print(df.head())




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


# plot the forecast via prophet plotting and review changes to trend
# ------
chart = m.plot(forecast)
# -- we can plot components directly
m.plot_components(forecast)
# plt.show()

# -- add trend changepoints 
change_points = add_changepoints_to_plot(chart.gca(), m, forecast)
plt.show()


