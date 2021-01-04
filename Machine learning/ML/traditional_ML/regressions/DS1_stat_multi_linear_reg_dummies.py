import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# -------------------------


# load data, it contains SAT, GPAs but also attendance as dummy variable
# ------------
raw_data = pd.read_csv('../data/1.03. Dummies.csv')
# print(raw_data.head())
# print(raw_data.describe())



# categorical variable (attendance = yes/no) needs mapping (attendance = 1/0)
# ------------
data = raw_data.copy()
data['Attendance'] = data['Attendance'].map({'Yes': 1, 'No': 0})
# print(data)
# print(data.describe())



# define regression variables
# ------------
# -- we want to know if SAT is a good predictor for later GPA
# use as independent variables both the SAT score and the categorical variable
x1 = data [['SAT','Attendance']]
# use as dependent variable (y) the GPA
y = data ['GPA']




# fit the regression model
# ------------
# -- add a constant
x = sm.add_constant(x1)
# -- Fit the model, according to the OLS (ordinary least squares) method 
results = sm.OLS(y,x).fit()
# print a summary of the regression
print(results.summary())
print('Parameters: ', results.params)
print('R2: ', results.rsquared)
# -- for a full list of options to extract parameters from the summary
# print(dir(results))



# see the results, given the resulting regression from above (colouring the datapoints according to categorical variable)
# ------------
# -- we can represent the multivariate regression even with multivariables, if categorical
# -- create a scatter plot of SAT and GPA, using the categorical series 'Attendance' as color
plt.scatter(data['SAT'],data['GPA'], c=data['Attendance'], cmap='RdYlGn_r')
# -- define the two regression equations, depending on whether they attended (yes), or didn't (no)
# -- we just have to change the constant, adding from the summary either 0 or the coefficient for attendance (if 1 = attending)
yhat_no = (results.params[0] + 0) + (results.params[1] * data['SAT'])   # not attending
yhat_yes = (results.params[0] + results.params[2]) + (results.params[1] * data['SAT'])  # attending
# -- plot the two regression lines
chart1 = plt.plot(data['SAT'],yhat_no, lw=2, c='#006837')
chart2 = plt.plot(data['SAT'],yhat_yes, lw=2, c='#a50026')
# Name your axes :)
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()
