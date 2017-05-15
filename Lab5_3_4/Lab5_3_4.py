import numpy as np
import pandas as pd
from random import randint
from sklearn import linear_model
from scipy import stats

def alpha(data,index):
    X = data['X'][index]
    Y = data['Y'][index]
    return ((np.var(Y) - np.cov(X,Y)) / (np.var(X) + np.var(Y) - 2*np.cov(X,Y)))[0,1]

def boot(data, index):
    X = data['horsepower'][index]
    Y = data['mpg'][index]
    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
    return [intercept, slope]


port = pd.read_csv('Portfolio.csv', usecols=range(1,3), parse_dates=True)
print alpha(port, range(0,100))
print alpha(port, np.random.choice(range(0, 100), size=100, replace=True))


auto = pd.read_csv('Auto.csv', usecols=range(0,8), parse_dates=True)
auto_converted = pd.DataFrame({'mpg':auto['mpg'],'horsepower':  pd.to_numeric(auto['horsepower'])})
print boot(auto_converted, range(1,392))
print boot(auto_converted, np.random.choice(range(0, 392), size=392, replace=True))




