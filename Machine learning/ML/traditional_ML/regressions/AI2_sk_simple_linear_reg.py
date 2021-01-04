import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -------------------------


# load data
# ------------
dataset = pd.read_csv('../data/Salary_Data.csv')
# https://stackoverflow.com/questions/509211/understanding-slice-notation/509295#509295
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# split dataset
# ------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# run the linear regression model
# ------------
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# run a future prediction vs test set
# ------------
y_pred = regressor.predict(X_test)


# visualize results vs test
# ------------
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()