import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# to encode dummies
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# -------------------------


# load data
# ------------
dataset = pd.read_csv('../data/50_Startups.csv')
# https://stackoverflow.com/questions/509211/understanding-slice-notation/509295#509295
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# organize categorical data
# ------------
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# split dataset
# ------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# run the linear regression model
# ------------
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# run a future prediction vs test set
# ------------
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


