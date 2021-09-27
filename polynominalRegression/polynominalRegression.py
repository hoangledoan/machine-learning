from matplotlib import colors
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#training polynominal linear regression
from sklearn.preprocessing import PolynomialFeatures
polly_reg = PolynomialFeatures(degree=4)
X_polly = polly_reg.fit_transform(X)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg = lin_reg.fit(X_polly, y)

#plotting the polynominal linear regression model
plt.scatter(X, y, color ='red')
plt.plot(X, lin_reg.predict(X_polly), color = 'blue')
plt.show()

#predicting the salary
print (lin_reg.predict(polly_reg.fit_transform([[6.5]])))