# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 11:59:23 2018

@author: BharadwN
"""

#This can be started only after installing Keras
#Data Preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#getting the dataset
dataset = pd.read_csv("C:\\Users\\bharadwn\\Desktop\\data science\\ANN\\Churn_Modelling.csv")
x = dataset.iloc[:, 3:13].values #Independent variables
y = dataset.iloc[:, 13].values#dependent variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encoding country
labelencoder_country = LabelEncoder()
x[:, 1] = labelencoder_country.fit_transform(x[:, 1])
#Encoding gender
labelencoder_gender = LabelEncoder()
x[:, 2] = labelencoder_gender.fit_transform(x[:, 2])
#by this point we will get a table where the country names are converted to either 0,1 or 2 value
#but our machine learning algorithm might think that we are rating the countries so we will be splitting the categorical column
#into 3 separate column where there will be 1 if country is present else 0
#this we will be doing using OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:] #We are taking all the columns except the first one to avoid dummy variable trap
#by this point we have the country values in 3 different columns
#-------------------------------------------------------------------------------
#Now we will split the dataset into trining sets
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)

#Now we will perform Feature Scaling to convert the huge numbers to small values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Insert ANN model HERE!!
#import keras This is required i guess but it isnt working for me for some reason
from keras.models import Sequential #this class is to initiate the NN
from keras.layers import Dense #This class is to create the hidden layers
#Initializing the ANN model
classifier = Sequential()

#Adding the input and first hidden layer
classifier.add(Dense(output_dim= 6, init= "uniform", activation = "relu", input_dim=11 ))

#Adding 2nd hidden layer here
classifier.add(Dense(output_dim= 6, init= "uniform", activation = "relu"))

#Adding the output layer
classifier.add(Dense(output_dim= 1, init= "uniform", activation = "sigmoid"))

#Compiling the ANN.
classifier.compile(optimizer= "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
#Now we will fit the classifier into the training set i.e. train the model
classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)

#Now we will predict the results
y_pred= classifier.predict(x_test)

#Now we will use something called cnfusion matrix which will basically give us the number of correct and incorrect predictions
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

