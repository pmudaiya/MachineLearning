#Importing Libararies
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values # if we do not use values we will get TypeError: unhashable type: 'numpy.ndarray'

##using elbow method to find optimal clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",n_init=10,max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("clusters")
plt.ylabel("wcss")
plt.title("Elbow curve to find optimum clusters")
plt.show()

kmeans=KMeans(n_clusters=5,init="k-means++",n_init=10,max_iter=300,random_state=0)
y_clisters=kmeans.fit_predict(X)

colors=["red","blue","green","cyan","magenta"]
cluster_name=["cluster 1","cluster 2","cluster 3","cluster 4","cluster 5"]
for i in range(0,5):
    plt.scatter(X[y_clisters==i,0],X[y_clisters==i,1],s=100,c=colors[i],label=cluster_name[i])

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("CLusters")
plt.show()