import pandas as pd
import statsmodels.api as sm

Smarket = pd.read_csv('Smarket.csv', index_col=0)

dta = Smarket[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Direction']]

glm_results = sm.GLM.from_formula(formula='Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume', data=dta, family=sm.families.Binomial()).fit()
print glm_results.summary()

# Parameter estimates:
print glm_results.params

# The corresponding t-values:
print glm_results.tvalues
