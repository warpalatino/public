import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# to run SVR
from sklearn.svm import SVR

# to execute feature scaling
from sklearn.preprocessing import StandardScaler

# -------------------------
# support vector regression


# load data
# ------------
dataset = pd.read_csv('../data/Position_Salaries.csv')
# https://stackoverflow.com/questions/509211/understanding-slice-notation/509295#509295
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)
y = y.reshape(len(y),1)
# print(y)


# normalize data
# ------------
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
# print(X)
# print(y)



# run the SVR model
# ------------
# -- more info on kernel functions here https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
# rbf = radial basis function, standard choice for SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)


# make a prediction
# ------------
prediction = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))



# visualize regression results
# ------------
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()





