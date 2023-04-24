import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import Counter

# from matplotlib.animation import FuncAnimation
# from IPython.display import HTML

def posterior_grid_approx(grid_points=5, success=6, tosses=9):
    """
        Computing the posterior using a grid approximation.
        using uniform prior and binomial likelihood
    """
    # define grid
    p_grid = np.linspace(0, 1, grid_points)

    # define prior
    prior = np.repeat(5, grid_points)  # uniform
    #prior = (p_grid >= 0.5).astype(int)  # truncated
    #prior = np.exp(- 5 * abs(p_grid - 0.5))  # double exp

    # compute likelihood at each point in the grid
    likelihood = stats.binom.pmf(success, tosses, p_grid)

    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior

    # standardize the posterior, so it sums to 1
    posterior = unstd_posterior / unstd_posterior.sum()
    return p_grid, posterior

# Bermoulli pmf estimation
def sample_bernoulli(n, p):
    """
        n: num samples
        p: probability
    """
    return np.sum([np.random.binomial(1, p) for i in range(n)])

def binomial_pdf(ks, n, p, simulations=1000):
    """
        simulation approach to get binom.pdf
        ks: list or a number (num success)
        n: num trials
        simulations: num samples to draw

        output:
        if ks is a number, just return the prob(k|n)
        if ks is a list, return pmf
    """
    
    # stats.binom.pmf(2, 9, 0.2)
    arr = [sample_bernoulli(n, p) for num in range(simulations)]
    c = Counter(arr) 
    # count occurrence and get frequency
    prob = [c[k]/simulations for k in range(n+1)]
    if isinstance(ks, list):
        return [prob[k] for k in ks ]
    else:
        return prob[ks]

def simulate_posterior_grid_approx(grid_points=5, success=6, tosses=9):
    """
        Computing the posterior using a grid approximation.
        using uniform prior and simulated binomial likelihood
    """
    # define grid
    p_grid = np.linspace(0, 1, grid_points)

    # define prior
    prior = np.repeat(5, grid_points)  # uniform

    # compute likelihood at each point in the grid
    # likelihood = stats.binom.pmf(success, tosses, p_grid)
    likelihood = [binomial_pdf(6, 9, p) for p in p_grid]

    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior

    # standardize the posterior, so it sums to 1
    posterior = unstd_posterior / unstd_posterior.sum()
    
    return p_grid, posterior

def sampling_coin_flipping_dist(k, n, prior, p_grid):
    """
        compute coin flipping posterior using sampling approach
    """
    likelihood = [binomial_pdf(k, n, p) for p in p_grid]
    unstd_posterior = likelihood * prior
    posterior = unstd_posterior / unstd_posterior.sum()
    return posterior

def analytical_coin_flipping_dist(k, n, prior, p_grid):
    "analytical approach"
    # likelihood = [binomial_pdf(k, n, p) for p in p_grid]
    likelihood = stats.binom.pmf(k, n, p_grid)
    unstd_posterior = likelihood * prior
    posterior = unstd_posterior / unstd_posterior.sum()
    return posterior

class UpdateDist(object):
    def __init__(self, ax):

        # data
        self.line, = ax.plot([], [], 'k-')

        self.ax = ax

        # Set up plot parameters
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 0.2)
        self.ax.grid(True)

    def setup(self, grid_points, num_simulations, p):
        # count
        self.success = 0
        self.p = p
        self.num_simulations = num_simulations
        
        #self.grid_points = grid_points
        self.x = np.linspace(0, 1, grid_points)
        self.prior = np.repeat(1, grid_points)  # uniform density
        
        # This vertical line represents the theoretical value, to
        # which the plotted distribution should converge.
        self.ax.axvline(self.p, linestyle='--', color='black')
                
        # sampling
        arr = stats.bernoulli.rvs(self.p, size=self.num_simulations)
        self.arr = arr.cumsum()
            
        # plot data
        self.line.set_data([], [])
        return self.line,

    def start(self):
        self.line.set_data([], [])
        return self.line,

    def __call__(self, i):
        
        k = self.arr[i]
        n = i + 1
        posterior = analytical_coin_flipping_dist(k, n, self.prior, self.x)
        self.prior = posterior
        
        # update y
        self.line.set_data(self.x, posterior)
        
        return self.line,

def density_plot(hist, bins=20):
    "Gaussian KDE plot of histogram"

    density = stats.gaussian_kde(hist)
    n, x, _ = plt.hist(hist,bins=bins, density=True)
    plt.plot(x, density(x), 'r')

from scipy.stats import binom
import math
from scipy.special import beta


# function (p, prior, data) 
# {
#     s = data[1]
#     f = data[2]
#     p1 = p + 0.5 * (p == 0) - 0.5 * (p == 1)
#     like = s * log(p1) + f * log(1 - p1)
#     like = like * (p > 0) * (p < 1) - 999 * ((p == 0) * (s > 
#         0) + (p == 1) * (f > 0))
#     like = exp(like - max(like))
#     product = like * prior
#     post = product/sum(product)
#     return(post)
# }

def pdisc(p, prior, data):
    '''
        posterior probability with discrete prior and binomial likelihood
        :param: p     : vector from 0 to 1
        :param: prior : weighted at the discrete point in p
        :param: data  : number of success and failure
    '''
    
    (s, f) = data
    # boundary at p==0 and p==1
    p1 = p + 0.5 * (p == 0) - 0.5 * (p == 1)
    # log likelihood
    like = s * np.log(p1) + f * np.log(1 - p1)
    # ??
    like = like * (p > 0) * (p < 1) - 999 * ((p == 0) * (s > 0) + (p == 1) * (f > 0))
    # likelihood (avoid overflow) 
    like = np.exp(like - np.max(like))
    # posterior
    product = like * prior
    post = product/sum(product)
    
    return(post)

def plot_pdisc(p, prior, data):
    "plot pror and posterior prob"
    
    post = pdisc(p, prior, data)
    binwidth = p[1] - p[0]
    
    plt.subplot(211)
    plt.bar(p, post, width=binwidth/10)
    plt.title('posterior pmf')
    plt.subplot(212)
    plt.bar(p, prior, width=binwidth/10)
    plt.title('prior pmf')
    
# histprior
# function (p, midpts, prob) 
# {
#     binwidth = midpts[2] - midpts[1]
#     lo = round(10000 * (midpts - binwidth/2))/10000
#     val = 0 * p
#     for (i in 1:length(p)) {
#         val[i] = prob[sum(p[i] >= lo)]
#     }
#     return(val)
# }

def histprior(p, midpts, prob):
    '''
        Computes the density of a probability distribution defined on a
        set of equal-width intervals
        :param: p         : vector from 0 to 1
        :param: midpts    : midpoints
        :param: prob      : discrete probability density
    '''
    
    binwidth = midpts[1] - midpts[0]
    # lower limit of interval
    lo = np.round(10000 * (midpts - binwidth/2)) / 10000
#     val = 0 * p
    val = np.zeros(len(p))
    for i in range(len(p)):
         val[i] = prob[sum(p[i] >= lo)-1]

    return(val)    
     

def plot_histprior(p, midpts, prob):
    "plot pror and posterior prob"
    
    prior = histprior(p, midpts, prob)
    plt.step(midpts, prior, where='mid')
    

# pdiscp
# function (p, probs, n, s) 
# {
#     pred = 0 * s
#     for (i in 1:length(p)) {
#         pred = pred + probs[i] * dbinom(s, n, p[i])
#     }
#     return(pred)
# }

def pdiscp(p, probs, n, s):
    '''
         Computes predictive distribution for number of successes of future
         binomial experiment with a discrete distribution for the
         proportion.
     
        :param: p    : vector from 0 to 1
        :param: probs: vector of  probabilities
        :param: n    : size of future binomial sample
        :param: s    : vector of number of successes for future binomial experiment       
    '''
    
#     pred = 0 * s
    pred = np.zeros(len(s))
    for i in range(len(p)):
        pred = pred + probs[i] * binom.pmf(s, n, p[i])
    
    return(pred)

# pbetap
# function (ab, n, s) 
# {
#     pred = 0 * s
#     a = ab[1]
#     b = ab[2]
#     lcon = lgamma(n + 1) - lgamma(s + 1) - lgamma(n - s + 1)
#     pred = exp(lcon + lbeta(s + a, n - s + b) - lbeta(a, b))
#     return(pred)
# }

def pbetap(ab, n, s):
    '''
         Computes predictive distribution for number of successes of future
         binomial experiment with a discrete distribution for the
         proportion.
     
        :param: ab   : vector of parameters of the beta prior
        :param: n    : size of future binomial sample
        :param: s    : vector of number of successes for future binomial experiment       
    '''
    
    pred = np.zeros(len(s))
    (a, b) = ab
    # FIXME: np.array
    lcon = math.lgamma(n + 1) - math.lgamma(s + 1) - math.lgamma(n - s + 1)
    pred = np.exp(lcon + np.log(beta(s + a, n - s + b)) - np.log(beta(a, b)))
    
    return(pred)

# derived from BernGrid.R
def BernGrid(theta, ptheta, data, isplot=True):
    "plot prior, Bernoulli likelihood and posterior"
    
    # Create summary values of Data
    # number of 1's in Data
    z = sum( data ) 
    N = len( data )
         
    # Compute the Bernoulli likelihood at each value of Theta:
    pDataGivenTheta = np.power(theta,z) * np.power(1-theta, N-z)
    
    # Compute the evidence and the posterior via Bayes' rule:
    pData = np.sum( pDataGivenTheta * ptheta )
    pThetaGivenData = pDataGivenTheta * ptheta / pData
    
    if isplot:
        plt.subplot(311)
        plt.plot(theta, ptheta)
        plt.fill_between(theta, 0, pTheta)        
        plt.subplot(312)
        plt.plot(theta, pDataGivenTheta)
        plt.fill_between(theta, 0, pDataGivenTheta)
        plt.subplot(313)
        plt.plot(theta, pThetaGivenData)
        plt.fill_between(theta, 0, pThetaGivenData)

    return (pThetaGivenData)
    
def prob_estimate_sampling(p_grid, posterior, p1, p2=None, num=1000, isplot=True):    
    '''
        estimate probability under curve
        :params: posterior : posterior density
        :params: p1        : density cutoff point
        :params: p2        : second cutoff point (if interval required)
        :params: num       : number of samples
    '''
    
    samples = np.random.choice(p_grid, p=posterior, size=num, replace=True)
    
    if p2 is None:
        result = sum( samples < p1 ) / num
    else:
        result = sum((samples > p1) & (samples < p2)) / num
    
    if isplot:
        points = sns.kdeplot(samples).get_lines()[0].get_data()
        x = points[0]
        y = points[1]
        if p2 is None:
            plt.fill_between(x, y, where = (x<p1))
        else:
            plt.fill_between(x, y, where = (x<p2) & (x>p1))
         
    return result


def hdp_estimate(p_grid, posterior, alpha=0.05, num=int(1e5), isplot=True):    
    '''
        estimate probability under curve
        :params: posterior : posterior density
        :params: p1        : density cutoff point
        :params: p2        : second cutoff point (if interval required)
        :params: num       : number of samples
    '''
    
    samples = np.random.choice(p_grid, p=posterior, size=num, replace=True)
    
    # HPDI (high probability density interval)
    hdp = pm.hpd(samples, alpha)
    lb = hdp[0]
    ub = hdp[1]
    
    if isplot:
        points = sns.kdeplot(samples).get_lines()[0].get_data()
        x = points[0]
        y = points[1]
        plt.fill_between(x, y, where = (x<ub) & (x>lb))
         
    return hdp

def density_norm_plot(x, ax=None):
    '''
        density compared with normal distribution
        :params: x: samples from a distribution
    '''
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    pm.kdeplot(x, ax=ax, label='Sampled Distribution')
    
    # plot equivalent normal distribution
    mu = np.mean(x)
    sd = np.sqrt(np.var(x))
    x = np.linspace(min(x), max(x), len(x))
    y = mlab.normpdf(x, mu, sd)
    ax.plot(x, y, 'r', label='Quadratic approx')
    
    curr_ylim_max = plt.ylim()[1]
    new_ylim_max = np.max(y)*1.1
    plt.ylim(0, max(curr_ylim_max, new_ylim_max))    
    ax.legend()
    
    return ax

    
def kde_coeff(trace, x, ci=0.1, alpha='alpha', beta='beta'):
    """
        density plot of the coefficients
        :params: trace: chain iterations
        :params: x : value 
        :params: alphal, beta: variables names in string
    """
    
    samples = trace[alpha] + trace[beta] * x
    #density_norm_plot(samples)
    
    # HPDI (high probability density interval)
    hdp = pm.hpd(samples, alpha=ci)
    lb = hdp[0]
    ub = hdp[1]
    points = sns.kdeplot(samples).get_lines()[0].get_data()
    x = points[0]
    y = points[1]
    plt.fill_between(x, y, where = (x<ub) & (x>lb))    
    
    
def evaluate_hdp(trace, model, x, ystr, alpha='alpha', beta='beta', N=10, NN=20, ci=0.1):    
    
        # x vector
        x_seq = np.linspace(min(x), max(x), N)
        mu_pred = np.zeros((len(x_seq), len(trace)*trace.nchains))
    
        # draw samples at each point in x 
        for i, w in enumerate(x_seq):
            mu_pred[i] = trace[alpha] + trace[beta] * w
            
        # mean at each point in x
        mu_mean = mu_pred.mean(1)
        
        # HPD at each point in x
        mu_hpd = pm.hpd(mu_pred.T, alpha=ci)    
        
        # get y prediction
        pred = pm.sample_ppc(trace, 100, model)
        pred_hpd = pm.hpd(pred[ystr])
        
        # reorder the data points in ascending order
        NN = min(NN, len(x))
        idx = np.argsort(x.values[:NN])
        y_ord = x.values[:NN][idx]
        pred_hpd = pred_hpd[idx]
        
        return (x_seq, mu_mean, mu_hpd, y_ord, pred_hpd)
        
        
def plot_simple_regression(trace, x, y, model=None, ystr=None, ci=0.1, N=100, 
                           NN=20, alpha='alpha', beta='beta', showCI=True):
    """
        linear regression evaluation
        :params: trace    : chain iterations
        :params: x,y      : predictor and target
        :params: model    : model object
        :params: ystr     : name of variable y
        :parmas: ci       : confidence interval
        :params: N        : x axis resolution
        :params: NN       : target value resolution
        :params: alpha, beta: variables names in string
    """
    
    # scatter plot
    plt.plot(x, y, '.')
    # regressed line by taking the mean of the coeff distribution
    plt.plot(x, trace[alpha].mean() + trace[beta].mean() * x)

    if showCI:
        (x_seq, mu_mean, mu_hpd, y_ord, pred_hpd) = evaluate_hdp(trace, model, x, ystr, 
                                                                 alpha, beta, N, NN, ci)
        
        # HPDI of mu
        plt.fill_between(x_seq, mu_hpd[:,0], mu_hpd[:,1], color='C2', alpha=0.25)
        # HDPI of y
        plt.fill_between(y_ord, pred_hpd[:,0], pred_hpd[:,1], color='C2', alpha=0.25)
        # fitted lines
        plt.plot(x_seq, mu_mean, 'C2')        
    
    

    
    