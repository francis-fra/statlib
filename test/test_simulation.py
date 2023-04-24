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
# p_grid, posterior = bayes.posterior_grid_approx(points, w, n)
p_grid, posterior = bayes.simulate_posterior_grid_approx(points, w, n)
# p_grid, posterior = posterior_grid_approx(points, w, n)
plt.plot(p_grid, posterior, 'o-', label='success = {}\ntosses = {}'.format(w, n))
plt.xlabel('probability of water', fontsize=14)
plt.ylabel('posterior probability', fontsize=14)
plt.title('{} points'.format(points))
plt.legend(loc=0)
plt.show()