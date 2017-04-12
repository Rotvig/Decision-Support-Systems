import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

data = np.loadtxt("housing.data")
# Load the diabetes dataset
diabetes = datasets.load_diabetes()
predXData = []
predYData = []
for item in data:
    predXData.append([item[12]])
    predYData.append(item[13])

# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(predXData, predYData)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(predXData) - predYData) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(predXData, predYData))

# Plot outputs
plt.scatter(predXData, predYData,  color='black')
plt.plot(predXData, regr.predict(predXData), color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()