import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=np.nan)

dataset = pd.read_csv('Data.csv')
X= dataset.iloc[: , :-1].values
y=dataset.iloc[:, 3].values


#missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy ='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3]= imputer.transform(X[:, 1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0]);
onehotencoder=OneHotEncoder(categorical_features= [0])
X= onehotencoder.fit_transform(X).toarray()
labelencoder_Y=LabelEncoder()
y=labelencoder_X.fit_transform(y);

#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train= sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)