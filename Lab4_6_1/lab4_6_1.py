import numpy as np

f = open("Smarket.csv")
f.readline()  # skip the header
data = np.genfromtxt(f, usecols=(1, 2, 3, 4, 5, 6, 7, 8), delimiter=',')

preparedData = []
for i in range(0,8):
    preparedData.append([item[i] for item in data])

result = np.corrcoef(preparedData)

print "Year | Lag1 | Lag2 | Lag3 | Lag4 | Lag5 | Volume | Today"
print result


