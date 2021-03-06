import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics

data = pd.read_csv('Smarket.csv', usecols=range(1,10), index_col=0, parse_dates=True)

# Traning data before 2005
x_train = data[:'2004'][['Lag1', 'Lag2']]
y_train = data[:'2004']['Direction']

# Test data after 2005
x_test = data['2005':][['Lag1', 'Lag2']]
y_test = data['2005':]['Direction']

lda = LinearDiscriminantAnalysis()
lda_result = lda.fit(x_train, y_train)

print "Score of train and test data"
print(lda_result.score(x_train, y_train), lda_result.score(x_test, y_test))

print "\n"
print"Confusion matrix:"
print(pd.crosstab(y_test, lda_result.predict(x_test),rownames=['True'], colnames=['Predicted'], margins=True))

print "\n"
print "classification report"
print(metrics.classification_report(y_test, lda_result.predict(x_test)))

print "\n"
print "Group Means:"
print lda_result.means_

print "\n"
print "Prior probability of groups::"
print lda_result.priors_