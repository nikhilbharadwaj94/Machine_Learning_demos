# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 13:12:31 2018

@author: BharadwN
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing datasets
dataset = pd.read_csv("C:\\Users\\bharadwn\\Desktop\\data science\\regression\\polynomial\\Polynomial_Regression\\position_Salaries.csv")

x = dataset.iloc[: , 1:2].values
y = dataset.iloc[: , 2].values

#For some reason we did not do feature scaling here. I wonder Why!!
#Now we will fit the decision tree regressor to the dataset.
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

#Visualizing the results with higher resolution
x_grid = np.arange(min(x), max(x), 0.01) # 0.01 represents the increment size
# the above statement will give us a vector. We will now convert it into an array

x_grid = x_grid.reshape((len(x_grid)), 1)
plt.scatter(x, y , color="red")
plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.xlabel("Position Level")
plt.ylabel("salary")
plt.title("Salary estimation using Decision Tree Regression")
plt.show()


