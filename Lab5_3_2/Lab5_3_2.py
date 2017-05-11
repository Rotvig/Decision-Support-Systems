import numpy as np
import pandas as pd
from sklearn import metrics, linear_model
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('Auto.csv', usecols=range(0,8), parse_dates=True)
df = pd.DataFrame({'mpg':data['mpg'],'horsepower':  pd.to_numeric(data['horsepower'])})
X_train = df[['horsepower']]
Y_train = df['mpg']

# Create linear regression object
lm = linear_model.LinearRegression()
lm.fit(X_train, Y_train)

print("Coeffecients: %.2f" % lm.coef_)

#Running cross_val_score for polynomial with different degrees
for i in range(1, 6):
    poly = PolynomialFeatures(degree=i)
    X_train_poly = poly.fit_transform(X_train)
    lm = linear_model.LinearRegression()
    #Setting splits to amount of data points, leave-one-out cross-validation
    k_fold = KFold(n_splits=X_train.shape[0]) 
    test = cross_val_score(lm, X_train_poly, Y_train, cv=k_fold, scoring = 'neg_mean_squared_error')
    print np.mean(-test)