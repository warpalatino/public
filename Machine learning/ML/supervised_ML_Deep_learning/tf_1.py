import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import mean_squared_error


# ---------------------------


# let's create data with a broadly linear shape (where linear is delivered by a linear regression y = mx + b + noise)
# ------
m =2
b =3
x = np.linspace(0,50,100)

np.random.seed(101)
noise = np.random.normal(loc=0, scale=4, size=len(x))   #loc is mean, scale is std, size is how many values, which is 100 values from x

#predict y
y = (2*x) + b + noise
# plt.plot(x, y, '*') # where * is to have x and y crossing
# plt.show()


# let's create a keras model with dense layers
# ------
# chose model and add layers => input, hidden and output
# number is the total neurons to deploy, activation is activation function, inpout dimension is x only one variable



model = keras.Sequential(
    [
        layers.Dense(4, input_shape=(1,), activation="relu", name="layer1"),
        layers.Dense(4, activation="relu", name="layer2"),
        layers.Dense(1, name="layer3"),
    ]
)

#create weights into the model => build it => return a tensor
a = tf.ones((1, 1))
b = model(a)
print("----------") 
print("Number of weights after calling the model:", len(model.weights)) 

# compile the model and see the summary details
model.compile(loss='mse', optimizer='adam')
model.summary()


# # fit the model
# # ------
# # epochs will depend on how big the dataset, etc.
model.fit(x, y, epochs=250)



# see model results
# ------
# to see the progress of the loss function
loss = model.history.history['loss']
# print(loss)


# #to plot the evolution of the model vs loss function
epochs = range(len(loss))
# plt.plot(epochs, loss)
# plt.show()


# now run the regression against the data (=full forecast)
# ------
predictions = model.predict(x)
plt.plot(x,y, '*')
plt.plot(x, predictions, 'r')
plt.show()


# verify errors
# ------
mse = mean_squared_error(y, predictions)
print(mse)