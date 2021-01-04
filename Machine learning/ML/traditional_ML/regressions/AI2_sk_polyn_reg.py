import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# to encode dummies
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# -------------------------


# load data
# ------------
dataset = pd.read_csv('../data/Position_Salaries.csv')
# https://stackoverflow.com/questions/509211/understanding-slice-notation/509295#509295
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# => no split here, we will use entire dataset

# run the linear regression model
# ------------
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# run the polynomial regression model
# ------------
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# visualize regression results
# ------------
# -- plot linear reg
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# -- plot polyn reg
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# run a future prediction vs test set
# ------------
# y_pred = regressor.predict(X_test)
# np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


