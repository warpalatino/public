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

#import stuff for ACF and PACF
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#import stuff for ARIMA models
from statsmodels.tsa.ar_model import AR, ARResults

#imports from sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error

#------------------------------------------

#simple AR model
#--> import and show data
df = pd.read_csv('data/uspopulation.csv', index_col='DATE', parse_dates=True)
df.index.freq = 'MS'
# df.plot()
# plt.show()

#--> split dataset 90/10
train_limit = round(len(df)*0.9)
test_limit = round(len(df)*0.1)
# print(train_limit)
# print(test_limit)

train = df.iloc[:train_limit]
test = df.iloc[train_limit:]

start = len(train)
end = len(train) + len(test) - 1
# print(start)
# print(end)

#++++++++++++++
#fit AR(1)
model1 = AR(train['PopEst'])
AR1fit = model1.fit(maxlag=1)    #use 1 maxlag = lag coefficient behind, main parameter, and many more methods available
AR1fit.params   #reporting constand and coeff

#predict
prediction1 = AR1fit.predict(start=start, end=end)    #
prediction1 = prediction1.rename('AR(1) prediction')
# print(prediction1)

#compare prediction1 vs test set
# test.plot(legend=True)
# prediction1.plot(legend=True)
# plt.show()

#++++++++++++++
#fit AR(2), using more history for better predictions
model2 = AR(train['PopEst'])
AR2fit = model2.fit(maxlag=2)
AR2fit.params
prediction2 = AR2fit.predict(start=start, end=end) 
prediction2 = prediction2.rename('AR(2) prediction')

#compare prediction2 vs test set
# test.plot(legend=True)
# prediction1.plot(legend=True)
# prediction2.plot(legend=True)
# plt.show()


#++++++++++++++
#how to choose the best order p for the AR model then?
#let's ask statsmodel
#fit AR(p)
model = AR(train['PopEst'])
ARfit = model.fit(ic='t-stat')  #ic is for information criterion
best_fit = ARfit.params #this will find how many coefficients in the model, and thus model order
# print(best_fit)

#run the prediction to verify how it looks
best_prediction = ARfit.predict(start=start, end=end) 
best_prediction = best_prediction.rename('AR(13) best prediction')

#evaluate model
labels = ['AR1', 'AR2', 'AR13']
predictions = [prediction1, prediction2, best_prediction]
for i in range(len(predictions)):
    error = mean_squared_error(test['PopEst'], predictions[i])
    # print('MSE was: ', error)

#plot all predictions
# test.plot(legend=True)
# prediction1.plot(legend=True)
# prediction2.plot(legend=True)
# best_prediction.plot(legend=True)
# plt.show()

#++++++++++++++
#forecast the future after rifitting
model_fcst = AR(df['PopEst'])
ARfit_fcst = model_fcst.fit()
print(ARfit_fcst.params)
forecast = ARfit_fcst.predict(start=len(df), end=len(df)+12)    #let's predict 12 months into the future
forecast = forecast.rename('final forecast')

#plotting the old set and the forecasted new one
df['PopEst'].plot(legend=True)
forecast.plot(legend=True)
plt.show()