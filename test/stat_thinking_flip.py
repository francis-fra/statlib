import sys
statlib_path = '/home/fra/Project/pyProj/statlib'
sys.path.append(statlib_path)

import bayes
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

points = 20
# posterior probility of p, given 6 out of 9 success
w, n = 6, 9
p_grid, posterior = bayes.posterior_grid_approx(points, w, n)
# p_grid, posterior = posterior_grid_approx(points, w, n)
plt.plot(p_grid, posterior, 'o-', label='success = {}\ntosses = {}'.format(w, n))
plt.xlabel('probability of water', fontsize=14)
plt.ylabel('posterior probability', fontsize=14)
plt.title('{} points'.format(points))
plt.legend(loc=0);

# 6 ones out of 9
data = np.repeat((0, 1), (3, 6))
with pm.Model() as normal_aproximation:
    p = pm.Uniform('p', 0, 1)
    w = pm.Binomial('w', n=len(data), p=p, observed=data.sum())
    # MAP (mode of posterior)
    mean_q = pm.find_MAP()
    # standard error
    std_q = ((1/pm.find_hessian(mean_q, vars=[p]))**0.5)[0]

mean_q
std_q
# pm.distributions.continuous.Uniform(0, 1)
# p
# w
# normal_aproximation

pm.hpd(data, alpha=0.5)
# m = pm.Model()

