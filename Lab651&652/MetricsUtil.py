import math
import numpy as np

#Mean squared error
def Mse(Y, Y_hat):
    return ((Y_test - Y_hat) ** 2).mean()

#Residual sum of squares
def RSS(Y, Y_hat):
    residual = Y - Y_hat
    return (residual ** 2).sum()

#Residual sum of squares
def A_rsquare(RSS, Y, Y_hat, n, d):
    #Total sum of squares, formula: sum((y-y_mean)^2)
    TSS = ((Y - Y.mean()) ** 2).sum()
    #Formula from slides:  adjusted R^2 = 1-(RSS/(n-d-1)/(TSS/(n-1)))
    return 1.0 - ((RSS / (n - d - 1)) / (TSS / (n - 1)))

def Cp(RSS, d, Y_hat, n, Y):
    # Formula from slides: Cp = 1/n * (RSS + 2* d * sigma^2)
    return  (1.0 / n) * (RSS + 2 * d * Y_hat.var())
              
def Bic(n, RSS, d, Y_hat):
      # Formula from slides: bic = 1/n * (RSS+log(n)*d*sigma^2)
      return (1.0 / n) * (RSS + math.log(n) * d * Y_hat.var())