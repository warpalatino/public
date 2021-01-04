import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import scipy.cluster.hierarchy as sch   #for dendogram
from sklearn.cluster import AgglomerativeClustering #for model


# -------------------------
# UNSUPERVISED LEARNING
# k-means clustering


# load data
# ------------
dataset = pd.read_csv('../data/Mall_Customers.csv')
# https://stackoverflow.com/questions/509211/understanding-slice-notation/509295#509295
X = dataset.iloc[:, [3, 4]].values


# how many custers? Find optimal method (via dendogram)
# ------------
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
# => now explore the chart "manually" and identify how many possible clusters, then provide the number as input below



# run the model
# ------------
# for details on the model, see here https://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)



# visualizing the clusters
# ------------
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()



