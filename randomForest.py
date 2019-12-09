# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:54:33 2018

@author: BharadwN
"""

#----------------Random Forest-------------------
import pandas as pd
import numpy as np
import matplotlib. pyplot as plt

dataset = pd.read_csv("C:\\Users\\bharadwn\\Desktop\\data science\\regression\\polynomial\\Polynomial_Regression\\position_Salaries.csv")

#Now i will be splitting the values into x and y terms
x = dataset.iloc[: , 1:2].values
y = dataset.iloc[: , 2].values

#For some reason we did not do feature scaling here. I wonder Why!!
#Now we will fit the Random Forest regressor to the dataset.
from sklearn.ensemble import RandomForestRegressor

rfregressor = RandomForestRegressor()
rfregressor.fit(x, y)

#Visualizing the results with higher resolution
x_grid = np.arange(min(x), max(x), 0.01) # 0.01 represents the increment size
# the above statement will give us a vector. We will now convert it into an array

x_grid = x_grid.reshape((len(x_grid)), 1)
plt.scatter(x, y , color="red")
plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.xlabel("Position Level")
plt.ylabel("salary")
plt.title("Salary estimation using Random Forest Regression")
plt.show()


