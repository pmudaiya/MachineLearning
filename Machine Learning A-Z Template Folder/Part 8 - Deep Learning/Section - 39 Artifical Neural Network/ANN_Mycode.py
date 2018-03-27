# Artificial Neural Network
"""
Best link to follow to install all 3
https://www.udemy.com/machinelearning/learn/v4/questions/2320940
"""
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

#important Points keep CUDA version 9.0 because 9.1 is not supported
# cudnn should be 7.0 7.1 is not supported

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMporting Data
dataset=pd.read_csv("Churn_Modelling.csv")
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#Categorical Data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#we do not need apply One hotencoder in Sex because it has only 2 values 0 or 1, 
#If more than values are there that we neeed to apply
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#Removing FIrst column to avoid Dummy Variable Trap
X = X[:, 1:]

#Splitting into Train and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling is compulsory or very important in Deep Learning to avoid large calculations
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Creating ANN
#importing libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
#Intialization ANN Layer By Layer
classifier=Sequential()

#Adding input layer and hidde layer
classifier.add(Dense(6,activation='relu',kernel_initializer='glorot_uniform',input_shape=(11,)))

#Adding second Hidden layer
classifier.add(Dense(6,activation='relu',kernel_initializer='glorot_uniform'))

#Adding Output Layer
classifier.add(Dense(1,activation='sigmoid',kernel_initializer='glorot_uniform'))

#Compiling Ann and applying STochastic Gradient
#Loss function="Logarithmic" Adam is advanced stochastic Gradient
classifier.compile(optimizer="adam" , loss="binary_crossentropy",metrics=['accuracy'])

#Fitting to dataset
classifier.fit(X_train,y_train,epochs=100,batch_size=10)

#Predicting Test Set Values
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


