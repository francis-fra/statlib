# generate random samples from a given distribution
import math
import numpy as np

#---------------------------------------------------------------------------------
# true mean and std deviation
mu, sigma = 0, 1
N = 20
samples = np.random.normal(mu, sigma, N)

# jackknife (leave one out mean)
sample_mean = np.mean(samples)
sample_mean
# 0.059520064554932514

# variance of the sample mean estimate  (error estimate)
# np.var(samples)
sample_var = np.sum(pow((samples - sample_mean), 2)) / (N-1) / N
sample_var
# 0.043542031704938144
math.sqrt(sample_var)
# 0.20866727511744176

#---------------------------------------------------------------------------------
sample_median = np.median(samples)
sample_median
# -0.021397590956793805

#---------------------------------------------------------------------------------
# Jack knife
#---------------------------------------------------------------------------------
jackknife_vec = np.zeros(N)
total = sum(samples)
for k in range(N):
    jackknife_vec[k] = (total - samples[k]) / (N-1)
    
jackknife_mean = np.mean(jackknife_vec)     
jackknife_mean
np.allclose(sample_mean, jackknife_mean)

#---------------------------------------------------------------------------------
# error variance of the sample estimate
jackknife_mean_var = np.sum(pow((jackknife_vec - jackknife_mean), 2)) * (N-1) / N
jackknife_mean_var
# 0.043542031704938144
np.allclose(sample_var, jackknife_mean_var)

# std deviation
math.sqrt(jackknife_mean_var)

#---------------------------------------------------------------------------------
# median estimate
jackknife_vec = np.zeros(N)
for k in range(N):
    # remove samples
    cloned = list(samples)
    del cloned[k]
    jackknife_vec[k] = np.median(cloned)

jackknife_median = np.mean(jackknife_vec)
jackknife_median
# -0.021397590956793805

np.allclose(sample_median, jackknife_median)

#---------------------------------------------------------------------------------
# median error estimate
jackknife_median_var = np.sum(pow((jackknife_vec - jackknife_median), 2)) * (N-1) / N
jackknife_median_var
# 0.091463402201087285

#---------------------------------------------------------------------------------
# bootstrap
#---------------------------------------------------------------------------------
# num of boostrap
B = 100
n = 200
bootstrap_vec = np.zeros(B)
for k in range(B):
    bootstrap_vec[k] = np.mean(np.random.choice(samples, n))
    

bootstrap_mean = np.mean(bootstrap_vec)    
bootstrap_mean
# 0.058036411668446357

# FIXME: variance much smaller??
bootstrap_mean_var = np.sum(pow((bootstrap_vec - bootstrap_mean), 2)) / (B-1)
bootstrap_mean_var
# 0.0043630172238908289


#---------------------------------------------------------------------------------
# investigate the effect on the size of B
n = 10  # more volatile for small n
B_vec = [20, 50, 100, 200, 500, 2000]
bootstrap_mean_vec = np.zeros(len(B_vec))
bootstrap_mean_var_vec = np.zeros(len(B_vec))

for kk in range(len(B_vec)):
    B = B_vec[kk]
    bootstrap_vec = np.zeros(B)
    for k in range(B):
        bootstrap_vec[k] = np.mean(np.random.choice(samples, n))
        
    bootstrap_mean_vec[kk] = np.mean(bootstrap_vec)
    bootstrap_mean_var_vec[kk] = np.mean(pow((bootstrap_vec - bootstrap_mean_vec[kk]), 2))
    
    
import matplotlib.pyplot as plt
plt.plot(B_vec, bootstrap_mean_vec, 's-')

#---------------------------------------------------------------------------------    

