# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:50:09 2018

@author: BharadwN
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing datasets
dataset = pd.read_csv("C:\\Users\\bharadwn\\Desktop\\data science\\regression\\polynomial\\Polynomial_Regression\\position_Salaries.csv")

x = dataset.iloc[: , 1:2].values
y = dataset.iloc[: , 2].values

#Now we need to split the datasets into test set and training set using
#the sklearn.preprocessing train_test_split method
#since we have only 10 records, kiril told it dosent make sense to split the
#data into trining and test set so we wont be doing that

#Now we will create a simple linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#Now we will create a polynomial linear regression so that we can compare the 
#two models
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(x_poly, y)

#Now we will use pyplot to visualize the results of the simple linear regression models we have created.
plt.scatter(x, y , color = "red")
plt.plot(x, lin_reg.predict(x), color="blue")
plt.title("Truth or Bluff(Simple Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
#Now we will use pyplot to visualize the results of thepolynomial regression models we have created.
#to get a more continuous plot we have to divide X into numbers with smaller increments
#We will call this as X grid and substitute it in our visualization graphs only
x_grid = np.arange(min(x), max(x), 0.1) #This will give us a vector
x_grid = x_grid.reshape(len(x_grid), 1) #This will give us an array
plt.scatter(x, y , color = "red")
plt.plot(x_grid, lin_reg_poly.predict(poly_reg.fit_transform(x_grid)), color="blue")
plt.title("Truth or Bluff(Polynomial Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Predict the salary at level 6.5 using linear regression
lin_reg.predict(6.5)

#Predict the salary at level 6.5 using linear regression
lin_reg_poly.predict(poly_reg.fit_transform(6.5))
