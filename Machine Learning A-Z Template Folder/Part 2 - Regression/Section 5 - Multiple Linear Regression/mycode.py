# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3]);
onehotencoder=OneHotEncoder(categorical_features= [3])
X= onehotencoder.fit_transform(X).toarray()

#remove ummy variable trap
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Creating Linear Regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting value
y_pred=regressor.predict(X_test)

#Backward Elimination
import statsmodels.formula.api as sm
X=np.append(arr= np.ones((50,1)).astype(int),values=X, axis=1)

# automatic  backward elimination
def backwardelimination(x,sl):
    num=len(x[0])
    print(num)
    for i in range(0,num-1):
        regressor_OLS=sm.OLS(endog=y, exog=x).fit()
        ma=max(regressor_OLS.pvalues).astype(float)
        print(ma)
        if ma>sl:
            for j in range(0,num-i):
                if (regressor_OLS.pvalues[j]==ma):
                    x=np.delete(x,j,1)
    regressor_OLS.summary()
    return x


X_opt=X[:,[0,1,2,3,4,5]]
SL=0.05
num=6;     
X_modeled=backwardelimination(X_opt,SL)
            
#left backard eelimination with adjusted R-squared