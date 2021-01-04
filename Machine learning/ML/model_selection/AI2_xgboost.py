import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score


# -------------------------
# XGBoost

# dataset has patients (rows) and features to predict if we can predict if a tumor is benign or not


# load data
# ------------
dataset = pd.read_csv('../data/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



# split dataset
# ------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# run XGBoost
# ------------
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# make a prediction
# ------------
y_pred = classifier.predict(X_test)


# create confusion matrix
# ------------
cm = confusion_matrix(y_test, y_pred)
# print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# applying k-Fold Cross Validation: we obtain the average accuracy across our model on different test sets
# ------------
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10) # cv indicates the splits that we want to perform on the train set
print("Accuracy after k-fold validation: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

