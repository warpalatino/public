import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator #data preprocessing
from tensorflow.keras.callbacks import EarlyStopping

from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# ---------------------------


# [1] load and inspect data - it is about a weather wireless sensor recording new data every 10 mins for 4.5 months
# the sensors register a lot of variables including temp, wind, humidity etc so we have many variables to play with
# ------
df = pd.read_csv('data/energydata_complete.csv', index_col='date', infer_datetime_format=True)  #infer_datetime_format makes pandas understand that the timestamp has not only a date but also time
df.plot()
# plt.show()



# [2] run tests 
# ------
# not running any test on the many variables available



# [3] split dataset 
# ------
# -- we will predict only 24 hours into the future
# print('Length of dataseries is: ')
# print(len(df))  
# let's grab the month of May 2016 to use as reference
df1 = df.loc['2016-05-01':].round(2)    #removing precision to existing data (=noise to the network)
# len(df) = 3853
# 10 mins observations will be 60/10 = 6 observations per hour, which every day will mean 24*6 = 144 observations
# we keep two days of data for test
test_days = 2
daily_observations = (60/10) * 24
test_observations = int(test_days * daily_observations)  #this will return 288 observations as cut-off between train and test dataset
train = df.iloc[:test_observations]
test = df.iloc[test_observations:]
# print(train)
# print(test)



# [4] scale/normalize data for the RNN (only on train data here)
# ------
# -- normalize as % of max value (via sklearn)
scaler = MinMaxScaler() # call the function/formula
scaler.fit(train)   # fit model to train dataset, so that a max value can be found for scaling
train_normalised = scaler.transform(train)  # fitting all dataset in a range between 0 and 1 => normalized!  (= basically we take the max and we calculate each value as % of the max, more or less...)
test_normalised = scaler.transform(test)



# [5] pre_process data into batches in format understood by RNN
# ------
# in general put below in n_input how many values the model look at to predict the next one; seasonal dataset may require the entire seasonal range for good predictions
n_input = 144     #let's forecast one day ahead = 144 observations
batch_size = 1  # batch to use at each round, smaller is likely to be better with timeseries
# -- all of the above means that keras generator takes two sequential datapoints and outputs the third datapoint that follows
# -- RNN generator formula below requires (a) source data as input, (2) final data to predict as output, (3) source n values to use for preiction, (4) batch to predict at each network round
generator = TimeseriesGenerator(train_normalised, train_normalised, length=n_input, batch_size=batch_size)
# -- what does the first batch look like in terms of dimensions?
x,y = generator[0]
print('given the array x (which has one value per each variable in the set): ')
print(x.flatten())
print('predict this y: ')
print(y)
# -- verify the shape of the dataset
train_shape = train_normalised.shape
# print(train_shape)   #it will return 3565 rows by 28 features/variables (or 28 columns in dataframe)
n_features = train_shape[1]  # in multi-variate RNNs this is the number of variables/features



# define the model, according to new keras instructions 
# https://www.tensorflow.org/guide/keras/rnn
# ------
model = keras.Sequential()
model.add(layers.LSTM(25))     
model.add(layers.Dense(train_shape[1], activation='relu', input_shape=(n_input, n_features))) #one neuron per feature means 28 neurons in this example
model.compile(optimizer='adam', loss='mse')




# fit (and run) the model 
# ------
# -- we can add early stopping to stop the model once it stops improving
early_stop = EarlyStopping(monitor='val_loss', patience=1)
test_generator = TimeseriesGenerator(test_normalised, test_normalised, length=n_input, batch_size=batch_size)
# -- now we can go ahead with fitting the model, incuding the early_stop feature
model_run = model.fit_generator(generator, epochs=2, validation_data=test_generator, callbacks=[early_stop])
print(model.summary())


# plot of the loss function to see loss convergence
# ------
# -- to see the progress of the loss function
loss = model.history.history['loss']
# print(loss)
loss_range = range(len(loss))
plt.plot(loss_range, loss)
# plt.show()



# run the model as experiment to predict first value into the future
# ------
# -- we take the last annual data to predict against the train set (=annual data, 12 observations)
first_batch = train_normalised[-n_input:]    #last 12 points of training set
# print(first_batch)
# -- let's make sure that the data has the shape desired by ther RNN, or reshape
first_batch = first_batch.reshape((batch_size, n_input, n_features))    # we will end up with shape 1 * 144 * 28
# -- now we can run the prediction; in this example, we take the previous 12 datapoints to predict the first point in the future, so putput is going to be 1 array/value
first_prediction = model.predict(first_batch)



# we are ready to run rolling predictions against the test dataset and into the future
# ------
# -- create space for holding predictions
test_predictions = []   
# -- generalise the batches to go through the test with right format, this here is a starting point
current_batch = first_batch #remember that the shape of this is (1,12,1), which we can obtain via current_batch.shape
# -- how long the training set? well, run predictions against that length (I can change the value below)
values_to_predict = len(test)
for i in range(values_to_predict):
    current_prediction =  model.predict(current_batch)[0]   #[0] here is just to format the array in a shape that can later be plotted, it removes one squared bracket
    test_predictions.append(current_prediction)
    # -- updating current batch to continue through rolling predictions. How to do it? Instructions on code below: 
    # -- Given shape of vector current_batch (1,12,1)... 
    # -- first, let's drop first value at each new period out of the 12 needed for next prediction:
    # -- the code [:, 1:, :] means "take all values (1) from 1st vector dimension, drop first value out of 12 values to roll to next period via '1:' meaning from second value to last, then take all values (1) in third dimension
    # -- second, add the new predicted value at each new period (keeping the right vector shape) via [[current_prediction]] appended to np.array
    # -- finally, make sure the current_prediction is added to the right dimension of 12 values, which is axis=1
    current_batch = np.append(current_batch[:, 1:, :], [[current_prediction]], axis=1)


# invert scaling to understand values in the original form
# ------
real_predictions = scaler.inverse_transform(test_predictions)
print(real_predictions)



# plot the findings
# ------
# -- conveniently create a new dataframe (with test dataset already) to add RNN predictions (to compare to test dataset) as new column
real_predictions = pd.DataFrame(data=real_predictions, columns=test.columns)
test.plot()     #test will have 288 rows and 28 columns
plt.show()



# save model (to avoid long re-training next time)
# ------
model.save('multiv_RNN.h5')
# to re-load, copy/paste and then run code below:
# from tensorflow.keras.models import load_model
# new model = load_model('first_RNN.h5')
# -- we can also load anything from a different path by adding the abolute path into parenthesis


# evaluate model 
# ------
# we can evaluate against MSE or other errors, but we will need to be careful to do it on a variable by variable basis
