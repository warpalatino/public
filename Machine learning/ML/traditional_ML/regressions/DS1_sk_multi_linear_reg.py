import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# -------------------------


# load data
# ------------
data = pd.read_csv('../data/1.02. Multiple linear regression.csv')
# print(data.head())
# print(data.describe())



# define regression variables
# ------------
# -- we want to know if SAT is a good predictor for later GPA
# use as independent variable (x) the SAT score
x = data [['SAT', 'Rand 1,2,3']]
# use as dependent variable (y) the GPA
y = data ['GPA']



# explore data before the regression
# ------------
# -- not exploring in multivariate regression



# check the data shape when dealing with ML libraries
# ------------
# -- sklearn works with data in matrix form, x is already a matrix => nothing to do
print(x.shape)
print(y.shape)



# fit the regression model with sklearn
# ------------
reg = LinearRegression()
reg.fit(x,y)     #variables must be in this order, i.e. independent first
# -- extract results
print('R2: ', reg.score(x,y))   #for R-squared
print('coefficients: ', reg.coef_)    #for coefficients (in an array)
print('constant: ', reg.intercept_)   #for constant
# -- calculating the adjusted R-squared
r2 = reg.score(x,y)
n = x.shape[0]
p = x.shape[1]
adjusted_r2 = 1- (1-r2) * (n-1) / (n-p-1)
print('R2: ', adjusted_r2)


# # make predictions
# # ------------
# # -- let's add random input data (SAT) to get possible GPAs
# new_data = pd.DataFrame(data=[1740,1760],columns=['SAT'])
# # -- make a prediction
# new_data['Predicted_GPA'] = reg.predict(new_data)
# print(new_data)


# # show results
# # ------------
# plt.scatter(x1,y)
# yhat = (reg.coef_ * x_matrix) + reg.intercept_
# chart = plt.plot(x1,yhat, lw=4, c='orange', label ='regression line')
# plt.xlabel('SAT', fontsize = 20)
# plt.ylabel('GPA', fontsize = 20)
# plt.show()