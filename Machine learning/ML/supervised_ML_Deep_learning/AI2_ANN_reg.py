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
# deep learning - ANN for regression

# ***
# dataset - we have data on temperature, pressure, humidity and vacuum to predict an energy output
# ***


# [1] load and pre-process data
# ------
dataset = pd.read_excel('../data/Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# print(dataset.head())
# print(X)
# print(y)



# [2] split dataset 
# ------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# [3] scale/normalize data for the ANN (only on training data here)
# ------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




# [4] define the model, according to new keras instructions 
# https://www.tensorflow.org/guide/keras/rnn
# ------
model = keras.Sequential()
# - add network layers
# how many neurons (i.e. hyperparameters in general)? experiment
model.add(layers.Dense(units=6, activation='relu'))     # input layer
model.add(layers.Dense(units=6, activation='relu'))     # hidden layer
model.add(layers.Dense(units=1))     # output layer (only 1 neuron, as we need one energy output; activation: while sigmoid is useful to return 0/1 and softmax useful to return multiple classifications, with regressions we leave activation empty )




# [5] fit (and run/train) the model 
# ------
model.compile(optimizer='adam', loss='mean_squared_error') # adam is a way to perform the stochastic gradient descent
model.fit(X_train, y_train, batch_size = 35, epochs = 50)
# print(model.summary())





# [6] we are ready to run rolling predictions against the test dataset
# ------
# -- here we predict the energy output from a given set of data => energy output is between 400 and 500
y_pred = model.predict(X_test)
# -- set the format desired to display later on Terminal
np.set_printoptions(precision=2)
# -- compare predicted value vs real/test value (that is known) - below we reshape the variables/vectors (y_pred first, y_test later) to display them vertically
predictions = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)  # the final value 1 is to display vertically, 0 would be horizontally
print(predictions)




