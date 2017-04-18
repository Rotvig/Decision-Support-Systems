# http://www.science.smith.edu/~jcrouser/SDS293/labs/lab4/Lab%204%20-%20Logistic%20Regression%20in%20Python.pdf

import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('Smarket.csv', usecols=range(1,10), index_col=0, parse_dates=True)

glm_results = sm.GLM.from_formula(formula='Direction ~ Lag1 + Lag2', data=df, family=sm.families.Binomial()).fit()
print glm_results.summary()

# First 10 probabilities of the market going 'DOWN'
predictions = glm_results.predict()
print(predictions[0:10])

# Create test data
x_test = df['2005':][['Lag1', 'Lag2']]
y_test = df['2005':]['Direction']

# Create predictions - Transforms 'UP' for all elements which has a probability lower than 0.5 for going down
predictions = glm_results.predict(x_test)
predictions_nominal = ["Up" if x < 0.5 else "Down" for x in predictions]
print classification_report(y_test, predictions_nominal, digits=3)

