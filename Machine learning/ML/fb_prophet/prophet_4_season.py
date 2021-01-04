import pandas as pd 
import matplotlib.pyplot as plt

#import prophet and trend tools
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

from statsmodels.tools.eval_measures import rmse

# ---------------------------


# [1] load and inspect data, with specific trend changes
# ------
df = pd.read_csv('data/airline_passengers.csv')
df.columns = ['ds','y']
df['ds'] = pd.to_datetime(df['ds'])
# print(df.head())




# [2] create and fit the model 
# [5] refitting the model after point 4 below, I add seasonality to Prophet and re-run all steps
# ------
# Prophet has many possible options to go inside the parenthesis, check docs
m = Prophet(seasonality_mode='multiplicative')
# first example ever here, so we will not split the databse in two parts (yet)
m.fit(df)



# [3] predict the future
# ------
future = m.make_future_dataframe(periods=50, freq = 'MS')     # future is the new dataframe including the forecasted future dates
# -- now predict the values and add them to the future dataframe
forecast = m.predict(future)
# print(forecast.head())



# [4] plot the forecast via prophet plotting and review changes to trend
# ------
chart = m.plot(forecast)
# -- we can plot components directly
m.plot_components(forecast)
# -- add trend changepoints 
change_points = add_changepoints_to_plot(chart.gca(), m, forecast)
plt.show()
# => we identify visually some seasonality that increases over time (=multiplicative instead of additive)

