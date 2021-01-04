import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# -------------------------


# load data, it contains a random categorical variable on top of SAT scores and GPAs
# ------------
data = pd.read_csv('../data/1.02. Multiple linear regression.csv')
# print(data.head())
# print(data.describe())


# define regression variables
# ------------
# -- we want to know if SAT is a good predictor for later GPA
# use as independent variables both the SAT score and the random categorization
x1 = data [['SAT','Rand 1,2,3']]
# use as dependent variable (y) the GPA
y = data ['GPA']



# explore data before the regression
# ------------
# -- with multivariate regressions it is hard to explore data on two dimensions only



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
# ------------
# also very difficult to show here
