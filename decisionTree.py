# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:23:13 2018

@author: BharadwN
"""
#I just copy pasted the template code as the only thing that will be different is the code for the model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:\\Users\\bharadwn\\Desktop\\data science\\classification\\Logistic_Regression\\Social_Network_Ads.csv")
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

#Now we will split the dataset into trining sets
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

#Feature scaling is not required here since the algorithm dosent depend on the Euclidian distance.
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Insert Decision Tree model below
from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(criterion="entropy", random_state=0)   #There are many types of kernels. check in ctrl+i and try all of them when you have a moment to spare.
#By just doing this our classifier is ready


#Now we will fit the classifier into the training set i.e. train the model
classifier.fit(x_train, y_train)

#Now we will predict the results
y_pred= classifier.predict(x_test)

#Now we will use something called cnfusion matrix which will basically give us the number of correct and incorrect predictions
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) #Basically we are comparing the results here.


#Now we will be visualizing the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
#Now we will be visualizing the test set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classifier (test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()