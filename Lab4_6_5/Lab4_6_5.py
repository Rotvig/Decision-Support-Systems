import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

data = pd.read_csv('Smarket.csv', usecols=range(1,10), index_col=0, parse_dates=True)

# Traning data before 2005
x_train = data[:'2004'][['Lag1', 'Lag2']]
y_train = data[:'2004']['Direction']

# Test data after 2005
x_test = data['2005':][['Lag1', 'Lag2']]
y_test = data['2005':]['Direction']

neigh = KNeighborsClassifier(n_neighbors=3)
knn_result = neigh.fit(x_train, y_train)

print "Score of train and test data"
print(knn_result.score(x_train, y_train), knn_result.score(x_test, y_test))

print "\n"
print"Confusion matrix:"
print(pd.crosstab(y_test, knn_result.predict(x_test),rownames=['True'], colnames=['Predicted'], margins=True))

print "\n"
print "classification report"
print(metrics.classification_report(y_test, knn_result.predict(x_test)))


'''
import numpy as np
import pylab as pl
from sklearn import neighbors, datasets
# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features.
Y = iris.target

h = .02 # step size in the mesh

# Plot the decision boundary. For that, we will asign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = x_train[:,0].min() - .5, x_train[:,0].max() + .5
y_min, y_max = x_train[:,1].min() - .5, x_train[:,1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = neigh.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
pl.figure(1, figsize=(4, 3))
pl.set_cmap(pl.cm.Paired)
pl.pcolormesh(xx, yy, Z)

# Plot also the training points
pl.scatter(x_train[:,0], x_train[:,1],c=y_train )
pl.xlabel('Sepal length')
pl.ylabel('Sepal width')

pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())
pl.xticks(())
pl.yticks(())

pl.show()
'''