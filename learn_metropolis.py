# %matplotlib inline
# import pymc3 as pm
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from collections import Counter 

# %config InlineBackend.figure_format = 'retina'
plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

def plot_discrete_pmf(dist):
    "print pmf specified in dictionary form"
    x = dist.keys()
    y = dist.values()
    plt.bar(x, y)

def next_param(param, target_dist):
    "random walk propose function to generate next param state"
    params = target_dist.keys()
    upperbound = max(params)
    lowerbound = min(params)
    
    # decide left or right
    p = stats.uniform.rvs()
    # propose current if at the bounded range
    if p > 0.5:
        proposed = min(param + 1, upperbound)
    else:
        proposed = max(param - 1, lowerbound)
        
    # decide if accept
    p_accept = min(1, target_dist[proposed] / target_dist[param])
    p = stats.uniform.rvs()
           
    return proposed if p < p_accept else param

def run_metropolis(theta, num_steps, target_dist):
    "random walk metropolis algorithm"
    
    hist = np.zeros(num_steps)
    hist[0] = theta
    for idx in range(1, num_steps):
        current_param = hist[idx-1]
        hist[idx] = next_param(current_param, target_dist)
    return hist


def plot_trajectory(hist):
    plt.plot(hist, range(len(hist)))

def get_prob_from_hist(hist):
    "get prob from array of items"
    cnt = Counter(hist)
    return normalize_dist(cnt)

def normalize_dist(dist):
    "normalize relative distributin"
    total = sum(dist.values())
    return {k: v/total for k, v in dist.items()}

# ------------------------------------------------------------
def create_random_pmf(num):
    return dict(zip(range(1, num+1), np.random.permutation(num) + 1))

def create_zero_pmf(num):
    return dict(zip(range(1, num+1), np.zeros(num+1)))


def propagate_distribution(prior_dist, target_dist):
    "one step forward estimated distribution"
    
    # loop each param
    params = target_dist.keys()
    num_params = len(params)
    dist = create_zero_pmf(num_params)
    for p in params:
        dist[p] = calculate_chain_prob(p, prior_dist, target_dist)
    
    return dist


def calculate_chain_prob(param, prior_dist, target_dist):
    "calculate the next stage pmf value for the specified param"
    
    # pmf(param) is the sum of prob jumping from left and right
    upperbound = max(target_dist.keys())
    lowerbound = min(target_dist.keys())
    
    left_param = param -1
    if left_param >= lowerbound:
        # prob if jumping from the left
        left_prob = 0.5 * min(1, target_dist[param] / target_dist[left_param])
        left_prior_prob = prior_dist[left_param]
    else:
        # at bounary, left is the same as current
        left_prob = 0.5
        left_prior_prob = 0
    
    right_param = param + 1
    if right_param <= upperbound:
        right_prob = 0.5 * min(1, target_dist[param] / target_dist[right_param])
        right_prior_prob = prior_dist[right_param]
    else:
        right_prob = 0.5
        right_prior_prob = 0
        
    #--------------------------------------------------------------------
    # stay prob: not going left nor right
    #--------------------------------------------------------------------    
    if left_param >= lowerbound:
        prob_to_left = min(1, target_dist[left_param] / target_dist[param])
    else:
        prob_to_left = 0
        
    if right_param <= upperbound:
        prob_to_right = min(1, target_dist[right_param] / target_dist[param])
    else:
        prob_to_right = 0
    
    stay_prob = 0.5 * (1-prob_to_left) + 0.5 * (1-prob_to_right)
    
    return left_prior_prob * left_prob + right_prior_prob * right_prob + prior_dist[param] * stay_prob

def metroplis_random_walk(step, prior_dist, target_dist):
    for k in range(step):
        prior_dist = propagate_distribution(prior_dist, target_dist)
    return prior_dist

# ------------------------------------------------------------
# 
# ------------------------------------------------------------
def gen_likelihood_function(z, N):
    "Bernoulli distribution"
    return lambda theta: theta**z * (1-theta)**(N-z)

def gen_proposed_function(sd=1):
    return lambda : stats.norm.rvs(0, sd)


def metropolis_stepping(current_param, likelihood, proposed):
    """
        General Metroplis algorithm:
        current_param:
        likelihood: function to evaluate the likelihood
        proposed: function to generate param delta 

    """
    
    delta = proposed()
    proposed_param = current_param + delta

    # FIXME: no prior
    
    # FIXME: hard coded
    upperbound = 1
    lowerbound = 0
    
    if (proposed_param > upperbound) or (proposed_param < lowerbound):
        p_accept = 0
    else:
        p_accept = min(1, likelihood(proposed_param) / likelihood(current_param))
    
    p = stats.uniform.rvs()
    if p < p_accept:
        return proposed_param
    else:
        return current_param