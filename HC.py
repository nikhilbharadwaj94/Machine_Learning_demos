# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:53:55 2018

@author: BharadwN
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Now we will be importing the dataset
dataset= pd.read_csv("C:\\Users\\bharadwn\\Desktop\\data science\\Clustering\\K-Means Clustering\\Mall_Customers.csv")

#Now we will split the dataset into x which is our independent vars and y which is our dependent vars
x = dataset.iloc[:, [3,4] ].values

#Now we will use dendogram to decide the number of clusters to be made.
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method="ward"))
#By this point our dendrogram is built

plt.title("Dendrogram")
plt.xlabel("customers")
plt.ylabel("Euclidian distances")
plt.show()

#Now we will create HC and fit into the data
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
#Linkage = ward minimises the variance between the data points it seems.
y_hc= hc.fit_predict(x)

#Visualizing the clustering results
plt.scatter(x[y_hc==0,0], x[y_hc==0,1], s=100, color="red", label="Careful")
plt.scatter(x[y_hc==1,0], x[y_hc==1,1], s=100, color="blue", label="average")
plt.scatter(x[y_hc==2,0], x[y_hc==2,1], s=100, color="green", label="Target")
plt.scatter(x[y_hc==3,0], x[y_hc==3,1], s=100, color="cyan", label="Careless")
plt.scatter(x[y_hc==4,0], x[y_hc==4,1], s=100, color="magenta", label="sensible")
#By this point we have plotted all the points in their respective cluster colors

#Now we will draw a boundary surrounding it
plt.title("Clusters of clients")
plt.xlabel("Annual income of clients($)")
plt.ylabel("Spending score(1-100)")
plt.legend()
plt.show()

