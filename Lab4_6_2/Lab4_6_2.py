import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

Smarket = pd.read_csv('Smarket.csv', index_col=0)


star98 = sm.datasets.star98.load_pandas().data
formula = 'Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume'
dta = Smarket[['NABOVE', 'NBELOW', 'LOWINC', 'PERASIAN', 'PERBLACK', 'PERHISP',
              'PCTCHRT', 'PCTYRRND', 'PERMINTE', 'AVYRSEXP', 'AVSALK',
              'PERSPENK', 'PTRATIO', 'PCTAF']]
endog = dta['NABOVE'] / (dta['NABOVE'] + dta.pop('NBELOW'))
del dta['NABOVE']
dta['SUCCESS'] = endog

'''
import pandas as pd
import statsmodels.api as sm
Smarket = pd.read_csv('Smarket.csv', index_col=0)

glm_model = sm.GLM(formula="Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume", data=Smarket, family=sm.families.Binomial())
glm_results = glm_model.fit()

print glm_results.summary()

# Parameter estimates:
print glm_results.params

# The corresponding t-values:
print glm_results.tvalues
'''
