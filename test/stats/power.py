import math
from scipy.stats import norm
# dir(norm)


def stderr_prop(p, n=None, se=None):
    '''
        :params: p: mean proportion
        :params: n: number of samples
     '''
    
    assert(p>=0 and p<=1)
    if n is None:
        return (p*(1-p)/(se*se))
    
    if se is None:
        assert(n>0)
        return math.sqrt(p*(1-p)/n)
    

def stderr_two_prop(p, n, se=None):
    '''
        :params: p: mean proportion
        :params: n: number of samples
     '''
    
    p1, p2 = p[0], p[1]
    n1, n2 = n[0], n[1]    
    assert(p1>=0 and p1<=1)
    assert(p2>=0 and p2<=1)
    assert(n1>0 and n2>0)
    
    se = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    return se 
    
    
def norm_cutoff2cumprob(cutoff, sided=2):
    "find cumulative prob given the cutoff"
    
    if sided ==1:
        return (norm.cdf(cutoff))
    else:
        p = norm.cdf(cutoff)
        return (2*p - 1)


def norm_cumprob2cutoff(p, sided=2):
    "find cumulative prob given the cutoff"
    
    if sided ==1:
        return (norm.ppf(p))
    else:
        p = (1-p)/2 + p
        return norm.ppf(p)


def stderrUB(p1, p2, n=None, se=None):
    "find the standard error upper bound with unequal proportions"
    
    phat = 0.5
    factor = phat * math.sqrt(1/p1 + 1/p2)
    if n is None:
        # find sample size
        assert(se is not None)
        result = pow(factor/se, 2)
    else:
        # find std err
        assert(n is not None)
        result = (factor / math.sqrt(n))
    
    return result


def power_two_groups(p1, p2, delta=None, n=None, pw=0.8, ci=0.95):
    '''
        :params: p1, p2 : proportions of treatment and control groups
        :params: delta: mean difference in proportion
        :params: n    : sample size (total)
        :params: ci   : confidence interval
        :params: pw   : power

     '''    
    phat = 0.5
    factor = phat * math.sqrt(1/p1 + 1/p2)
        
    if delta is None:
        # find delta
        assert(n is not None)
        # cutoff of normal density
        cutoff = norm_cumprob2cutoff(ci, 2)
        # power cutoff
        pw_cutoff = norm_cumprob2cutoff(pw, 1)
        result = (cutoff + pw_cutoff) * factor / math.sqrt(n)
    elif n is None:
        # find sample size
        # cutoff of normal density
        cutoff = norm_cumprob2cutoff(ci, 2)
        # power cutoff
        pw_cutoff = norm_cumprob2cutoff(pw, 1)        
        result = math.pow((cutoff + pw_cutoff) * factor / delta, 2)
    else:
        # find power
        # cutoff of normal density
        cutoff = norm_cumprob2cutoff(ci, 2)
        # find power cutoff
        w = math.sqrt(n) * delta / factor - 1.96
        # find power
        result = norm_cutoff2cumprob(w, 1)
    
    return result    
    
    

def power_one_group(delta=None, n=None, pw=0.8, ci=0.95):
    '''
        :params: delta: mean difference in proportion
        :params: n    : sample size
        :params: ci   : confidence interval
        :params: pw   : power
     '''
    
    # the most conservative standard error is at p=0.5
    phat = 0.5
        
    if delta is None:
        # find delta
        assert(n is not None)
        # cutoff of normal density
        cutoff = norm_cumprob2cutoff(ci, 2)
        # power cutoff
        pw_cutoff = norm_cumprob2cutoff(pw, 1)
        result = (cutoff + pw_cutoff) * phat / math.sqrt(n)
    elif n is None:
        # find sample size
        # cutoff of normal density
        cutoff = norm_cumprob2cutoff(ci, 2)
        # power cutoff
        pw_cutoff = norm_cumprob2cutoff(pw, 1)        
        result = math.pow((cutoff + pw_cutoff) * phat / delta, 2)
    else:
        # find power
        # cutoff of normal density
        cutoff = norm_cumprob2cutoff(ci, 2)
        # find power cutoff
        w = math.sqrt(n) * delta / phat - 1.96
        # find power
        result = norm_cutoff2cumprob(w, 1)
    
    return result

# find power
power_one_group(delta=0.1)    
power_one_group(n=96)
power_one_group(delta=0.1, n=196)
power_one_group(delta=0.1, n=96)
    
power_two_groups(0.2, 0.8, delta=0.1)
power_two_groups(0.2, 0.8, n=196)
power_two_groups(0.2, 0.8, n=1226)

#----------------------------------------------------------------
# testing

p1, p2 = 0.5, 0.5
phat = 0.5
phat * math.sqrt(1/p1 + 1/p2)
    
    
stderr_prop(0.6, se=0.05)
stderr_prop(0.6, n=100)

stderrUB(0.8, 0.2, se=0.1)


norm_cutoff2cumprob(1.96, 2)
norm_cutoff2cumprob(1.96, 1)
norm_cutoff2cumprob(1.66, 2)
        
norm_cumprob2cutoff(0.95, 1)
norm_cumprob2cutoff(0.95, 2)

norm_cumprob2cutoff(0.8, 1)


# inverse functions
norm_cumprob2cutoff(norm_cutoff2cumprob(1.95, 2), 2)
norm_cutoff2cumprob(norm_cumprob2cutoff(0.95, 2), 2)

#----------------------------------------------------------------
# DEBUG
#----------------------------------------------------------------

# find cumulative probability given value
norm.cdf(1.96)
# 0.97500210485177952

# find value give cumulative probability
# P(x < ?) = 0.975
norm.ppf(0.975)
# 1.959963984540054
norm.ppf(0.95)
# 1.6448536269514722
norm.ppf(0.9)
# 1.2815515655446004
norm.ppf(0.8)
# 0.8416212335729143

norm.cdf(0.95) - norm.cdf(0.05)


