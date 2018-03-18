# simple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#for linear regression we do need scaling library will do it for us

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicitng test set values
y_pred=regressor.predict(X_test)

#viisualising the training set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Experience vs salary(training data)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test,y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Experience vs salary(test data)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()