import numpy as np
import pandas as pd
from random import randint
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def boot_python(data, function, num_of_iteration):
    n = data.shape[0]
    idx = np.random.randint(0, n, (num_of_iteration, n))
    stat = np.zeros(num_of_iteration)
    for i in xrange(len(idx)):
        stat[i] = function(data, idx[i])
    return {'Mean': np.mean(stat), 'std. error': np.std(stat)}

def boot_python2(data, function, num_of_iteration):
    n = data.shape[0]
    idx = np.random.randint(0, n, (num_of_iteration, n))
    stat = np.zeros((num_of_iteration, 2))
    for i in xrange(len(idx)):
        stat[i] = function(data, idx[i])
    return {'Mean intercept': np.mean(stat[:,1]), 
            'std. error intercept': np.std(stat[:,1]), 
            'Mean slope': np.mean(stat[:,0]), 
            'std. error slope': np.std(stat[:,0])}


def boot_python3(data, function, num_of_iteration):
    n = data.shape[0]
    idx = np.random.randint(0, n, (num_of_iteration, n))
    stat = np.zeros((num_of_iteration, 3))
    for i in xrange(len(idx)):
        stat[i] = function(data, idx[i])
    return {'Mean intercept': np.mean(stat[:,0]), 
            'Mean horsepower': np.mean(stat[:,1]), 
            'Mean I(horsepower^2)': np.mean(stat[:,2]),
            'std. error intercept':  np.std(stat[:,0]),
            'std. error horsepower':  np.std(stat[:,1]),
            'std. error I(horsepower^2)':  np.std(stat[:,2])}

def alpha_fn(data, index):
    X = data['X'][index]
    Y = data['Y'][index]
    return ((np.var(Y) - np.cov(X,Y)) / (np.var(X)+np.var(Y) - 2*np.cov(X,Y)))[0,1]

def boot_fn(data, index):
    X = data['horsepower'][index]
    Y = data['mpg'][index]
    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
    return [slope,  intercept]

def boot_fn2(data, index):
    formula = 'mpg ~ horsepower+I(horsepower**2)'
    model = smf.glm(formula=formula, data=data, subset=index)
    result = model.fit()
    return result.params

portData = pd.read_csv('Portfolio.csv', usecols=range(1,3), parse_dates=True)

# print alpha_fn(portData, range(0,100))
# print alpha_fn(portData, np.random.choice(range(0, 100), size=100, replace=True))
# print boot_python(portData, alpha_fn, 1000)

autoData = pd.read_csv('Auto.csv', usecols=range(0,8), parse_dates=True)
#print boot_fn(autoData, range(0,392))
#print boot_fn(autoData, np.random.choice(range(0, 392), size=392, replace=True))
#print boot_fn(autoData, np.random.choice(range(0, 392), size=392, replace=True))

# print boot_python2(autoData, boot_fn, 1000)

# formula = 'mpg ~ horsepower'
# model = smf.glm(formula=formula, data=autoData)
# result = model.fit()
# print(result.summary())
# print "Incercept t-value"
# print (result.params["Intercept"] - result.params["horsepower"])/result.bse["Intercept"]
# print "Horsepower t-value"
# print (result.params["horsepower"] - 0)/result.bse["horsepower"]

print boot_python3(autoData, boot_fn2, 1000)

formula = 'mpg ~ horsepower+I(horsepower**2)'
model = smf.glm(formula=formula, data=autoData)
result = model.fit()
print(result.summary())
print "Incercept t-value"
print (result.params["Intercept"] - result.params["horsepower"])/result.bse["Intercept"]
print "Horsepower t-value"
print (result.params["horsepower"] - result.params["I(horsepower ** 2)"])/result.bse["horsepower"]
print "I(Horsepower^2) t-value"
print (result.params["I(horsepower ** 2)"] - 0)/result.bse["I(horsepower ** 2)"]