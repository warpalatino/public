import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator #data preprocessing

# enconding
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# ---------------------------
# deep learning - ANN for classification

# ***
# we have data showing which clients with which characteristics remained customers of a bank or not 
# we train the model on such data and try to see if we can learn and how can we predict
# ***


# [1] load and pre-process data
# ------
dataset = pd.read_csv('../data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values    #this includes all rows, but we do not need the first three columns (raw number, customer id, surname) as they will not impact the classification decision; we also remove the last column, which we report as y
y = dataset.iloc[:, -1].values  # this is the dependent variable => customer with certain features stays or leaves
# print(dataset.head())
# print(X)
# print(y)



# [2] encode categorical data (sex via label enconding, then Geography column via hot encoding)
# ------
# - first, label enconding (0/1) for female/male column
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2]) # X[:, 2] this indicates that we take all row of second column, the one indicating sex, for normal 0/1 label encoding
# print(X)
# - second, hot econding for geography column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')   # the code [1] will indicate that we have to transform the second column of the updated dataframe
X = np.array(ct.fit_transform(X))   # transform all data into an np array with different information
# print(X)




# [3] split dataset 
# ------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# [4] scale/normalize data for the ANN (only on training data here)
# ------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




# [5] define the model, according to new keras instructions 
# https://www.tensorflow.org/guide/keras/rnn
# ------
model = keras.Sequential()
# - add network layers
# how many neurons (i.e. hyperparameters in general)? experiment
model.add(layers.Dense(units=6, activation='relu'))     # input layer
model.add(layers.Dense(units=6, activation='relu'))     # hidden layer
model.add(layers.Dense(units=1, activation='sigmoid'))     # output layer (only 1 neuron, as we need a 0/1 answer)




# [6] fit (and run) the model 
# ------
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy']) # adam is a way to perform the stochastic gradient descent
model.fit(X_train, y_train, batch_size = 50, epochs = 50)
print(model.summary())




# [7] run the model as experiment to predict one value 
# ------
# -- now we can run the prediction; in this example, we take the previous 12 datapoints to predict the first point in the future, so putput is going to be 1 array/value

# ***
# example - Predict if a customer (with certain characteristics) will leave the bank: YES/NO
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $ 60000
# Number of Products: 2
# Credit card? Yes
# Active Member: Yes
# Estimated Salary: $ 50000
# ***

# the client sample variable below is an array into an array, bringing it to a 2D array as required by the predict method
client_sample = [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]] # 1,0,0 is France via hot encoding and all the orher variables describe the model client
normalised_client_sample = sc.transform(client_sample)    
first_prediction = model.predict(normalised_client_sample)  # this prediction returns a probability (of the client leaving) between 0 and 1
print('first prediction: ')
print('False = client stays; True = client leaves')
print(first_prediction > 0.5)   # we print and filter here if the predicted probability of a client leaving is above 50%: if True, client is predicted to leave 
print('----------------')




# [8] we are ready to run rolling predictions against the test dataset
# ------
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5) # again, here we interpret the predicted probability as above (line 115)
# -- show here the prediction against the real test result to find out accuracy
predictions = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
print(predictions)




# [9] measure model accuracy
# ------
cm = confusion_matrix(y_test, y_pred)
# print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)



# # invert scaling to understand values in the original form
# # ------
# real_predictions = scaler.inverse_transform(test_predictions)
# print(real_predictions)



# # plot the findings
# # ------
# # -- first, conveniently amend the test dataframe (with test dataset already) to add RNN predictions (to compare to test dataset) as new column
# expanded_test_set = test.copy()
# expanded_test_set['Predictions'] = real_predictions
# expanded_test_set.plot()
# plt.show()



# # save model (to avoid long re-training next time)
# # ------
# model.save('second_RNN.h5')
# # to re-load, copy/paste and then run code below:
# # from tensorflow.keras.models import load_model
# # new model = load_model('first_RNN.h5')
# # -- we can also load anything from a different path by adding the abolute path into parenthesis



# # full forecasting into the future
# # ------
# # -- same process as 'rolling predictions against the test dataset'
# # -- just make sure to change the variable 'values_to_predict'
