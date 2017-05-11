from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pp
import numpy as np

data = pd.read_csv('Auto.csv', usecols=range(0,8), parse_dates=True)
df = pd.DataFrame({'mpg': data['mpg'],'horsepower': data['horsepower'].apply(pd.to_numeric, errors='coerce')})
train, test = train_test_split(df, test_size = 0.5)

X_train = train[['horsepower']]
Y_train = train['mpg']

X_test = test[['horsepower']]
Y_test = test['mpg']

# Create linear regression object
lm = linear_model.LinearRegression()
lm.fit(X_train, Y_train)

print("Mean squared error with out Poly: %.2f"
      % np.mean((Y_test -lm.predict(X_test)) ** 2))

# Polyomial 2-degree
poly = pp.PolynomialFeatures(2)
x = poly.fit_transform(X_train)
x_t = poly.fit_transform(X_test)
lm_poly = linear_model.LinearRegression()
lm_poly.fit(x, Y_train)

print("Mean squared error with 2-degree Poly: %.2f"
      % np.mean((Y_test - lm_poly.predict(x_t)) ** 2))
      
# Polyomial 3-degree
poly = pp.PolynomialFeatures(3)
x = poly.fit_transform(X_train)
x_t = poly.fit_transform(X_test)
lm_poly = linear_model.LinearRegression()
lm_poly.fit(x, Y_train)

print("Mean squared error with 3-degree Poly: %.2f"
      % np.mean((Y_test - lm_poly.predict(x_t)) ** 2))

