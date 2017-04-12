import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

data = np.loadtxt("housing.data")
# Load the diabetes dataset
diabetes = datasets.load_diabetes()
predXData = []
predYData = []
predZData = []
for item in data:
    predXData.append([item[12]])
    predZData.append([item[0]])
    predYData.append(item[13])

x_train = np.array([predXData, predZData]).reshape(-1,2)
# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(x_train, predYData)

# Plot outputs
plt.scatter(x_train, predYData,  color='black')
plt.plot(x_train, regr.predict(x_train), color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()