import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

# Data load
mydata = pd.read_csv('data_regression.csv')
print(mydata.head())

# Data refine
mydata.loc[mydata['SS_DAY'] == -9]
mydata.loc[(mydata['SS_DAY'] != -9) & (mydata['CA_TOT'] == 0.5)] # Day: 20150520/ Average daily total cloud cover: 0.5
mydata.loc[mydata['SS_DAY'] == -9, 'SS_DAY'] = 10.7 # Set average 10.7

# simple linear regression modeling
result = sm.ols(formula='SS_DAY ~ CA_TOT', data=mydata).fit() ### result of regression
print(result.summary())

# detail result
print('< Parameters > \n', result.params) # print result
print('< Prob (Parameters) > \n', result.pvalues) # print P-value of regression
print('< Adj. R-squaured > \n', result.rsquared_adj) # print R-squared
print('< Prob (F-statistic) > \n', result.f_pvalue) # print model's goodness of fit

# Graph
fig, ax = plt.subplots(figsize=(8, 5))
plt.ylabel('SS_DAY', size=12)
plt.xlabel('CA_TOT', size=12)

ax.plot(mydata.CA_TOT.values, mydata.SS_DAY.values, 'o', label='Data')
ax.plot(mydata.CA_TOT.values, result.fittedvalues, 'b-', label='Regression')
ax.legend(loc='best')

### by Korea Meteorological Administration