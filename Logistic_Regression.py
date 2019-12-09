# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:11:51 2018

@author: BharadwN
"""

#----------------Logistic Regression-------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:\\Users\\bharadwn\\Desktop\\data science\\classification\\Logistic_Regression\\Social_Network_Ads.csv")
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
# In x i am ignoring the employee ID's as a data scientist i dont think it will matter
#I will now be encoding the gender data and split it
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_gender = LabelEncoder()
#x[:, 0] = labelencoder_gender.fit_transform(x[:, 0])
#by this point we will get a table where the gender is converted to either 0/1 value
#but our machine learning algorithm might think that we are rating the gender so we will be splitting the categorical column
#into 2 separate column where there will be 1 if country is present else 0
#this we will be doing using OneHotEncoder
#onehotencoder = OneHotEncoder(categorical_features = [0])
#x = onehotencoder.fit_transform(x).toarray()
#by this point we have the gender values in 2 different columns

#We will now split the datasets into test set and trining set.
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

#Now we will perform Feature Scaling to convert the huge numbers to small values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Now i will fit the logistic regression into the training set to create the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)
#By now the classifier learns the co-relation between x_train and y_train and it will generate a classification model

#Now we will predict the test set results
y_pred = classifier.predict(x_test)

#creating the confusion matrix to evaluate the performance of the classifier
#We will compare the actual y results i.e(y_test) with the predicted results(y_pred)
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

#Now we will be visualizing the test set results
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
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
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
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()