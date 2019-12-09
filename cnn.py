# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:06:03 2018

@author: BharadwN
"""

#Before we begin we have to make sure that Keras is installed on the system.
#For CNN's since we have the dataset split in folders we dont need to write code for 
#Pre-processing it and all.

#Importing the required packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#We will directly start building the model now
#Initializing the CNN
classifier = Sequential()

#Adding the convolution layer- Step 1
#keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation=None, weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='default', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
#32, 3, 3 says we will have 32 feature maps of 3X3 grid in the convolution layer.
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation="relu"))

#Performing Max Pooling - Step 2
classifier.add(MaxPooling2D(pool_size= (2,2)))
#The above one line will take care of the pooling functionality

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full connection for the hidden CNN layer
classifier.add(Dense(output_dim = 128, activation = "relu"))

#Now we will create the output layer with full connection. It has only 2 possible outputs i.e. CAT or DOG
classifier.add(Dense(output_dim = 2, activation = "sigmoid"))
#By this point our model is completely built

#Now we need to compile the model i.e. the classifier
classifier.compile(optimizer = "adam", loss="binary_crossentropy", metrics =["accuracy"])

#Now we will fit the CNN into our images
#First we need to apply some transformations to the training set

from keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator(
        rescale=1./255, #the values here represent how much the images will get 
        #transformed. We will keep the default values however we can change them 
        #if we want.
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                 'C:\\Users\\bharadwn\\Desktop\\data science\\CNN\dataset\\training_set',
                                                 target_size=(64, 64),#it was 150, 150 by default
                                                    batch_size=32, #Check document for info
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:\\Users\\bharadwn\\Desktop\\data science\\CNN\dataset\\test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000, #Changed from 2000 to 8000 as we have 8000 training images
        epochs=25, #Originally it was 50 but 50 will take too much time
        validation_data=test_set,
        validation_steps=2000)
