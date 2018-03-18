# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values

#plotting dendogram
import scipy.cluster.hierarchy as sch
sch.dendrogram(sch.linkage(X,method='ward'))
plt.title("Dendogram")
plt.xlabel("People")
plt.ylabel("Euclidena Distance")
plt.show()

#fitting to dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage='ward')
y_pred=hc.fit_predict(X)


colors=["red","blue","green","cyan","magenta"]
cluster_name=["cluster 1","cluster 2","cluster 3","cluster 4","cluster 5"]
for i in range(0,5):
    plt.scatter(X[y_pred==i,0],X[y_pred==i,1],s=100,c=colors[i],label=cluster_name[i])

#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("CLusters")
plt.show()