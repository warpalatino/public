import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# -------------------------


# load data
# ------------
data = pd.read_csv('../data/1.01. Simple linear regression.csv')
# print(data.head())
# print(data.describe())



# define regression variables
# ------------
# -- we want to know if SAT is a good predictor for later GPA
# use as independent variable (x) the SAT score
x1 = data ['SAT']
# use as dependent variable (y) the GPA
y = data ['GPA']



# explore data before the regression
# ------------
# Plot a scatter plot (first we put the horizontal axis, then the vertical axis)
plt.scatter(x1,y)
# Name the axes
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
# Show the plot
# plt.show()



# check the data shape when dealing with ML libraries
# ------------
print(x1.shape)
print(y.shape)
# -- sklearn works with data in matrix form, let's reshape current data
# we ask numpy to figure out the dimension by using -1 with a single feature
x_matrix = x1.values.reshape(-1,1)
print(x_matrix.shape)



# fit the regression model with sklearn
# ------------
reg = LinearRegression()
reg.fit(x_matrix,y)     #variables must be in this order, i.e. independent first
# -- extract results
print(reg.score(x_matrix,y))   #for R-squared
print(reg.coef_)    #for coefficients (in an array)
print(reg.intercept_)   #for constant



# make predictions
# ------------
# -- let's add random input data (SAT) to get possible GPAs
new_data = pd.DataFrame(data=[1740,1760],columns=['SAT'])
# -- make a prediction
new_data['Predicted_GPA'] = reg.predict(new_data)
print(new_data)


# show results
# ------------
plt.scatter(x1,y)
yhat = (reg.coef_ * x_matrix) + reg.intercept_
chart = plt.plot(x1,yhat, lw=4, c='orange', label ='regression line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()