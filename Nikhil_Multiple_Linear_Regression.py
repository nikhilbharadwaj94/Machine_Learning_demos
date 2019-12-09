# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 17:48:02 2018

@author: BharadwN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#This is for importing the dataset
dataset = pd.read_csv("50_Startups.csv")
x= dataset.iloc[: , : -1].values
y= dataset.iloc[:, -1].values
#-------------------------------------------------------
#Now we have to split the data into various sets using the sklearn package
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_country = LabelEncoder()
x[: , -1] = labelencoder_country.fit_transform(x[:, -1])
#By this point we have converted names into numbers like 1,2,3 etc
onehotencoder= OneHotEncoder(categorical_features =[-1])
x = onehotencoder.fit_transform(x).toarray()
#By this point 1,2,3 etc is replaced by 1 where country name is present
#0 otherwise. Also the country column is split into various columns containing
#only 1's and 0's
#--------------------------------------------------
#to avoid dummy variable trap we will remove one column from the existing
#set of columns. For this we will omit the first column i.e. 0
x = x[:, 1:]
#basically we are selecting all rows and omitting only the first column
#Now we will split the data set into training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#test_size determines % of data we want to keep in test set. The remining
#will be transferred to train set.

#IMPORTANT- apparently we dont need to perform feature scaling for multiple linear regression
#because the package we choose will automatically take care of it for us

#Now we will create a model out of the multiple variables available.

from sklearn.linear_model import LinearRegression
#we are importing linear regression because we are still creating a linear
#model but with multiple variables

regressor = LinearRegression()
regressor.fit(x_train, y_train)
#At this point the model is trained from the training set
#--------------------------------------------------------
#Now we will be predicting the output based on the model

y_pred= regressor.predict(x_test)

#--------------------------------------------------------------
#Now he is teaching backwards modelling or something like that :P
#For this we are using the statsmodel package.
import statsmodels.formula.api as sm

#Now we will be appending the dataset with a column of 1's as it is a necessary step to
#get the correct model i.e. we are inserting X0 = 1 in the formula
x = np.append(arr = np.ones((50,1)).astype(int), values =x, axis =1)
#ITERATION 1------------------------------------------------------------------
x_opt = x[: , [0,1,2,3,4,5]] #basically we are selecting all the variables in the first step
#Now we will be creating a regressor using ordinary least square method
regressor_OLS= sm.OLS(endog = y, exog = x_opt).fit()
#Let us assume that the initial selection level SL= 5% ie 0.05
#in backward elimination method we will keep eliminating the variables with the highest p
#values in regressor_OLS.summary() from x_opt
regressor_OLS.summary()
#ITERATION 2------------------------------------------------------------------------------------
#from the summary output i found x2 containing highest p value so i will be removing it from
#x_opt in this iteration
x_opt = x[:, [0,1,3,4,5]]
regressor_OLS= sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
#ITERATION 3-----------------------------------------------------------------
x_opt = x[:, [0,3,4,5]]
regressor_OLS= sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
#ITERATION 4--------------------------------------------------------------
x_opt = x[:, [0,3,5]]
regressor_OLS= sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
#ITERATION 5--------------------------------------------------------------
x_opt = x[:, [0,3]]
regressor_OLS= sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
#because P>|t| column values were all lesser than 0.05, we have now finished creating our model
#and it contains 0,3