import numpy as np
import pandas as pd
from random import randint
from sklearn import linear_model
from scipy import stats
import statsmodels.formula.api as smf

def boot_python(data, function, num_of_iteration):
    n = data.shape[0]
    idx = np.random.randint(0, n, (num_of_iteration, n))
    stat = np.zeros((num_of_iteration, 2))
    for i in xrange(len(idx)):
        stat[i] = function(data, idx[i])
    return {'Mean intercept': np.mean(stat[:,1]), 
            'std. error intercept': np.std(stat[:,1]), 
            'Mean slope': np.mean(stat[:,0]), 
            'std. error slope': np.std(stat[:,0])}

def alpha(data,index):
    X = data['X'][index]
    Y = data['Y'][index]
    return ((np.var(Y) - np.cov(X,Y)) / (np.var(X) + np.var(Y) - 2*np.cov(X,Y)))[0,1]

def boot(data, index):
    X = data['horsepower'][index]
    Y = data['mpg'][index]
    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
    return [intercept, slope]


#port = pd.read_csv('Portfolio.csv', usecols=range(1,3), parse_dates=True)
#print alpha(port, range(0,100))
#print alpha(port, np.random.choice(range(0, 100), size=100, replace=True))
#print boot_python(port, alpha, 1000)


auto = pd.read_csv('Auto.csv', usecols=range(0,8), parse_dates=True)
auto_converted = pd.DataFrame({'mpg':auto['mpg'],'horsepower':  pd.to_numeric(auto['horsepower'])})
#print boot(auto_converted, range(1,392))
#print boot(auto_converted, np.random.choice(range(0, 392), size=392, replace=True))
#print boot_python(auto_converted, boot, 1000)

# horsepower
model = smf.glm(formula='mpg ~ horsepower', data=auto_converted)
result = model.fit()
print(result.summary())
print "Incercept t-value"
print (result.params["Intercept"] - result.params["horsepower"])/result.bse["Intercept"]
print "Horsepower t-value"
print (result.params["horsepower"] - 0)/result.bse["horsepower"]

# I(horsepower ^2)
model = smf.glm(formula='mpg ~ horsepower+I(horsepower^2)', data=auto_converted)
result = model.fit()
print(result.summary())
print "Incercept t-value"
print (result.params["Intercept"] - result.params["horsepower"])/result.bse["Intercept"]
print "Horsepower t-value"
print (result.params["horsepower"] - result.params["I(horsepower ^ 2)"])/result.bse["horsepower"]
print "I(Horsepower^2) t-value"
print (result.params["I(horsepower ^ 2)"] - 0)/result.bse["I(horsepower ^ 2)"]