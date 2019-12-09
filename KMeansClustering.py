# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:36:54 2018

@author: BharadwN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Now we will be importing the dataset
dataset= pd.read_csv("C:\\Users\\bharadwn\\Desktop\\data science\\Clustering\\K-Means Clustering\\Mall_Customers.csv")

#Now we will split the dataset into x which is our independent vars and y which is our dependent vars
x = dataset.iloc[:, [3,4] ].values

#using elbow method to find the optimum number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter= 300, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
#plt.label("The Elbow Method")
plt.plot(range(1,11), wcss)
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

#Applying K-Means to the Mall dataset.
kmeans = KMeans(n_clusters=5, init = "k-means++", random_state=0)
y_kmeans = kmeans.fit_predict(x)
#By this point our model has clustered the individual points into their respective clusters by assigning the cluster number

#Now we will visualize the points within the clusters
plt.scatter(x[y_kmeans==0,0], x[y_kmeans==0,1], s=100, color="red", label="Careful")
plt.scatter(x[y_kmeans==1,0], x[y_kmeans==1,1], s=100, color="blue", label="average")
plt.scatter(x[y_kmeans==2,0], x[y_kmeans==2,1], s=100, color="green", label="Target")
plt.scatter(x[y_kmeans==3,0], x[y_kmeans==3,1], s=100, color="cyan", label="Careless")
plt.scatter(x[y_kmeans==4,0], x[y_kmeans==4,1], s=100, color="magenta", label="sensible")
#By this point we have plotted all the points in their respective cluster colors

#Now we will draw a boundary surrounding it
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c="Yellow", label="Centroid")
plt.title("Clusters of clients")
plt.xlabel("Annual income of clients($)")
plt.ylabel("Spending score(1-100)")
plt.legend()
plt.show()

