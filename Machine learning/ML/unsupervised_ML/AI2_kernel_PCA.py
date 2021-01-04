import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# -------------------------
# UNSUPERVISED ML
# kernel PCA - principal component analysis


# ***
# dataset is about wines (in each row) and their characteristics
# decision to predict to which customer segment does this wine belong?
# can we reduce the amount of wine features to track so to recognise which customer segment will buy each new wine in the shop?
# *** 


# load data
# ------------
dataset = pd.read_csv('../data/Wine.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# split dataset
# ------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# normalize train and test sets via feature scaling
# ------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# apply the PCA technique to the model before training
# ------------
# for details on the model: https://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html 
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)


# run/train the Logistic model (but any other classification model would work) on the new principal components from dataset
# ------------
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# predict
# -------------
y_pred = classifier.predict(X_test)


# create confusion matrix
# ------------

cm = confusion_matrix(y_test, y_pred)
# print(cm)
score = accuracy_score(y_test, y_pred)
print(score)


# # visualize regression results on training set (ignore warnings)
# # ------------
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
# plt.title('Logistic Regression (Training set)')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend()
# plt.show()

# # visualize regression results on test set (ignore warnings)
# # ------------
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
# plt.title('Logistic Regression (Test set)')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend()
# plt.show()



