import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


# -------------------------
# Kernel - SVM


# load data
# ------------
dataset = pd.read_csv('../../data/Social_Network_Ads.csv')
# https://stackoverflow.com/questions/509211/understanding-slice-notation/509295#509295
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# => x are two set of values on a different scale, while y is a zero to 1 categorization


# split dataset
# ------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)


# normalize train and test sets via feature scaling
# ------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(X_train)
# print(X_test)


# run the Kernel-SVM model
# ------------
# -- the simple code different vs standard SVM is that we apply the rbf kernel below, useful when data-points are not linearly separable
# for details on the classifier, see here https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


# make a prediction (vs test and on new values)
# ------------
# -- first, predict vs train set
y_pred = classifier.predict(X_test)
show_pred = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
# print(show_pred)
# => this would lead to a set of categorization values, either zero or one, to be compared to test set for accuracy
# -- second, predict vs new value
values = [[30,87000]]
normalized_values = sc.transform(values)
prediction = classifier.predict(normalized_values)
# print(prediction)
# => this would lead to a categorization value, either zero or one


# create confusion matrix
# ------------
cm = confusion_matrix(y_test, y_pred)
print(cm)
# => it would return [[65  3][ 8 24]]
score = accuracy_score(y_test, y_pred)
print(score)
# => this can help reading the accuracy vs matrix results = 0.89


# visualize regression results on training set (ignore warnings)
# ------------
# -- first, retransform values from normalized to standard
X_set, y_set = sc.inverse_transform(X_train), y_train
# -- second, shape the plot
# => we have to create a grid with .meshgrid (https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy)
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# => use enumerate to automatically get a counter in a loop and change the variable accordingly
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show() 

# visualize regression results on test set (ignore warnings)
# ------------
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
# plt.show()



