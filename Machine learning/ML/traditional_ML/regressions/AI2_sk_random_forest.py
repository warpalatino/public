import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# to run Random forest
from sklearn.ensemble import RandomForestRegressor


# -------------------------
# Random forest


# load data
# ------------
dataset = pd.read_csv('../data/Position_Salaries.csv')
# https://stackoverflow.com/questions/509211/understanding-slice-notation/509295#509295
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values



# run the Random forest model
# ------------
# -- more info on kernel functions here https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
# rbf = radial basis function, standard choice for SVR
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)


# make a prediction
# ------------
prediction = regressor.predict([[6.5]])



# visualize regression results
# ------------
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()





