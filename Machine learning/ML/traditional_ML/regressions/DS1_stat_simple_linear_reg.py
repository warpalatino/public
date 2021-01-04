import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


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



# fit the regression model
# ------------
# add a constant
x = sm.add_constant(x1)
# Fit the model, according to the OLS (ordinary least squares) method 
results = sm.OLS(y,x).fit()
# print a summary of the regression
print(results.summary())
print('Parameters: ', results.params)
print('R2: ', results.rsquared)
# -- for a full list of options to extract parameters from the summary
# print(dir(results))



# see the results, given the resulting regression from above
# https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html
# ------------
# re-create a scatter plot with underlying data
plt.scatter(x1,y)
# Define the regression equation, so we can plot it later
yhat = (results.params[1] * x1) + results.params[0]
# plot the regression line (with some chart attributes)
chart = plt.plot(x1, yhat, lw=4, c='orange', label ='regression line')
# Label the axes
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()