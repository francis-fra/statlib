import sys
statlib_path = '/home/fra/Project/pyProj/statlib'
sys.path.append(statlib_path

import pandas as pd

data_folder = r'/home/fra/FraDir/learn/Learnpy/Statistical-Rethinking-with-Python-and-PyMC3/Data/'
filename = 'Howell1.csv'
# read data
d = pd.read_csv(data_folder + filename, sep=';', header=0)
# exclude non-adult
d2 = d[d.age >= 18]
d2.shape
d2.head()

# linear model
import pymc3 as pm

# model the height
with pm.Model() as m4_1:
    # priors of params
    mu = pm.Normal('mu', mu=178, sd=20)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    # model
    height = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height)

# sample the posterior
with m4_1:
    trace_4_1 = pm.sample(1000, tune=1000)

# trace_4_1
# pm.traceplot(trace_4_1);

with m4_1:
    means = pm.find_MAP()

means

import numpy as np
np.mean(d2.height)
np.std(d2.height)

# samples of mu and sigma
trace_df = pm.trace_to_dataframe(trace_4_1)
trace_df
trace_df.cov()

pm.summary(trace_4_1).round(2)

# ------------------------------------------------------------
# liner relationship of weight with height
# ------------------------------------------------------------
np.mean(d2.height)
# 154.597

with pm.Model() as m4_3:
    # prior of the linear model parameters
    # intercept
    alpha = pm.Normal('alpha', mu=178, sd=100)
    # slope
    beta = pm.Normal('beta', mu=0, sd=10)
    # params
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    # linear regression model
    mu = alpha + beta * d2.weight
    #mu = pm.Deterministic('mu', alpha + beta * d2.weight) # try uncomenting this line and comenting the above line
    height = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height)
    # param trace
    trace_4_3 = pm.sample(1000, tune=1000)

pm.summary(trace_4_3)

# use centered weight as input
d2 = d2.assign(weight_c=pd.Series(d2.weight - d2.weight.mean()))
d2.head()

# predict height using centered weight
with pm.Model() as m4_4:
    alpha = pm.Normal('alpha', mu=178, sd=100)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    mu = alpha + beta * d2.weight_c
    height = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height)
    trace_4_4 = pm.sample(1000, tune=1000)

pm.summary(trace_4_4)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(d2.weight.to_numpy().reshape(-1,1), d2.height)
lin_reg.intercept_, lin_reg.coef_

import statsmodels.api as sm

X = d2.weight
X = sm.add_constant(X)
ols = sm.OLS(d2.height, X)
ols_result = ols.fit()

ols_result.summary()