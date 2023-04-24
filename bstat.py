import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy
import seaborn as sns
from scipy.interpolate import griddata
import math
# import tqdm
import inspect, warnings

from matplotlib import pyplot as plt
import arviz as az

from torch import Tensor as tt
import torch
import pyro
from torch import nn
from torch.distributions import transform_to, constraints
import pyro.distributions as dist

from pyro.nn import PyroModule
from pyro.distributions import Uniform, Normal, Exponential, HalfNormal
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide import (
            AutoLaplaceApproximation, 
            init_to_value, AutoNormal, 
            AutoDiagonalNormal, AutoMultivariateNormal
)
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import SGD, Adam
from pyro.contrib.autoname import name_count, named
from pyro.nn import PyroSample
# import pyro.ops.stats as stats
import pyro.poutine as poutine
from pyro.infer import TracePosterior, TracePredictive, Trace_ELBO
from pyro.ops.welford import WelfordCovariance

# ------------------------------------------------------------
# statistical plots
# ------------------------------------------------------------
def pyro_sample_histplot(samples, data=None, figsize=(15,4)):
    """
        samples: pyro samples
    """
    if data is None:
        num_plots = len(samples.keys())
    else:
        num_plots = len(samples.keys()) + 1
    fig, ax = plt.subplots(1, num_plots, figsize=figsize)
    for idx, param in enumerate(samples.keys()):
        sns.histplot(samples[param], ax=ax[idx])
        ax[idx].set_xlabel(param)
        ax[idx].set(ylabel=None)
    if data is not None:
        sns.histplot(data, ax=ax[idx+1])
        ax[idx+1].set_xlabel('data')
        ax[idx+1].set(ylabel=None)

def hist_with_gaussian_approx(ds, bins=20, step_size=50):
    """
        ds: data series
        num: num points in x axis
    """
    plt.hist(ds, bins=20, density=True, histtype="step")
    x = np.linspace(min(ds), max(ds), step_size)
    m = np.mean(ds)
    std = np.std(ds)
    print(f'mu: {m}, sigma: {std}')
    y = scipy.stats.norm.pdf(x, m, std)
    plt.plot(x, y, label="analytic")


def contour_plots(x_grid, y_grid, post_prob, step_size=100, figsize=(12,4)):
    """
        x_grid:
        y_grid:
        prob: probabilty
        
    """
    _, ax = plt.subplots(1, 2, figsize=figsize)

    xi = np.linspace(x_grid.min(), x_grid.max(), step_size)
    yi = np.linspace(y_grid.min(), y_grid.max(), step_size)
    zi = griddata((x_grid, y_grid), post_prob, (xi[None, :], yi[:, None]))

    ax[0].contour(xi, yi, zi)
    ax[1].imshow(zi, origin="lower", extent=[150.0, 160.0, 7.0, 9.0], aspect="auto")
    ax[1].grid(False)

def sampling_from_2d_grid(x_grid, y_grid, distribution, size=10000):
    """
        x_grid: 1d array of params
        y_grid: 1d array of params
        distribution: 2d grid density (1d array)

        RETURN
        ------
        2d array

    """
    idx = np.random.choice(np.prod(distribution.shape), p=distribution.flatten(), size=size)
    idx = np.vstack([idx//len(x_grid), idx%len(y_grid)])
    samples = np.vstack((x_grid[idx[0]], y_grid[idx[1]]))
    return samples

# ------------------------------------------------------------
# multivariate normal grid posterior
# ------------------------------------------------------------
def normal_grid_posterior(data, mu_lb, mu_ub, 
                            sigma_lb, sigma_ub, 
                            mu_loc, mu_scale, 
                            sigma_loc, sigma_scale, N):
    """
        normal mu with uniform sigma priors and normal likelihood

        data: array like
        mu_lb, mu_ub: range
        sigma_lb, sigma_ub: range
        mu_loc, mu_scale: prior
        sigma_loc, sigma_scale: prior
        N: step size
    
    """
    # mu prior grid
    mu_grid = torch.linspace(mu_lb, mu_ub, N)
    # sigma prior grid
    sigma_grid = torch.linspace(sigma_lb, sigma_ub, N)

    likelihood = Normal(mu_grid.view(-1, 1, 1), 
                        sigma_grid.view(1, -1, 1)).log_prob(data).sum(axis=-1)

    # expand the linear grid for getting the priors
    mu_grid = torch.repeat_interleave(mu_grid, N).numpy()
    sigma_grid = sigma_grid.repeat(1, N).reshape(-1,).numpy()

    # log normal
    mu_prior = scipy.stats.norm.logpdf(mu_grid, loc=mu_loc, scale=mu_scale)
    # log uniform
    sigma_prior = scipy.stats.uniform.logpdf(sigma_grid, loc=sigma_loc, scale=sigma_scale)

    lk = tt(likelihood)
    lk = lk.reshape(-1,).numpy()
    lk = lk + mu_prior + sigma_prior
    # taking exp() to convert log posterior to posterior
    post_prob = np.exp(lk - max(lk))
    post_prob = post_prob / post_prob.sum()
    return (mu_grid, sigma_grid, post_prob)


# ------------------------------------------------------------
# classical linear regression
# ------------------------------------------------------------
def ols(X, y, add_const=True):
    """
        least sq estimate
        X: numpy matrix
        y: numpy array
        add_const: boolean
    """
    if add_const:
        X = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X)).values
    invAtA = np.linalg.inv(np.matmul(X.T,X))
    # estimated coeff
    params = np.matmul(np.matmul(invAtA, X.T), y)
    return params

def fitted_summary(y, X, params, lb, ub):
    "std errors, t values and confidence intervals"
    N = len(y)
    # X2 = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X)).values
    y_pred = np.matmul(X, params)
    degf = len(params)
    
    # error variance of y (N samples but take away the num of freedom used to estimating the params)
    mse = sum((y-y_pred)**2) / (N-degf)
    # error variance for each param
    var_b = mse*(np.linalg.inv(np.dot(X.T,X)).diagonal())
    # standard error of each param
    sd_b = np.sqrt(var_b)
    # t values: standard by the std error of each param
    ts_b = params / sd_b

    # p values : Pr(t>ts_b)
    p_values =[2*(1-scipy.stats.t.cdf(np.abs(i),N-degf)) for i in ts_b]

    # use cdf inverse to find values given the cumulative prob
    lower_bound = [scipy.stats.t.ppf(lb, N - degf, loc=b_hat, scale=stderr) \
                     for b_hat, stderr in zip(params, sd_b)]
    upper_bound = [scipy.stats.t.ppf(ub, N - degf, loc=b_hat, scale=stderr) \
                     for b_hat, stderr in zip(params, sd_b)]

    # rounding for printing
    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)
    lower_bound = np.round(lower_bound,4)
    upper_bound = np.round(upper_bound,4)

    df = pd.DataFrame()
    df["Coefficients"],df["Standard Errors"],df["t values"],df["P>|t|"] = [params,sd_b,ts_b,p_values]
    df['[' + str(lb)] = lower_bound
    df[str(ub)+']'] = upper_bound
    return df


def reg_summary(X, y, lb=0.025, ub=0.975, add_constant=True):
    """
        scipy like regression summary

        X: numpy matrix or data frame
        y: numpy array or data series
        lb, ub: confidence interval bound
        add_constant: boolean
    """
    if type(X) != np.ndarray:
        X = X.values
    if type(y) != np.ndarray:
        y = y.values
    if add_constant:
        X = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X)).values

    params = ols(X, y, False)
    print(fitted_summary(y, X, params, lb, ub))
    y_pred = np.matmul(X, params)
    stat_dict = reg_stat(y, y_pred, params)
    # print(pd.Series(stat_dict))
    return stat_dict

def reg_stat(y, y_pred, params):
    """
        regression statistics
        y: np array
        y_pred: np array
        params: np array

        RETURN
        ------
        dict of R2, loglik, DW, Fstat
    """

    err = y - y_pred
    N = len(y)
    degf = len(params)
    sse =  np.sum(err**2)

    # F statistics
    # ratio of predicted variance and sse
    avg_pred_var = np.sum((y_pred - np.mean(y))**2 / (degf-1))
    avg_sse = sse / (N-degf)
    f_stat = avg_pred_var / avg_sse
    
    # durbin-watson statistics: check if error are lag-1 correlated
    # lag 1 automcorrleation square
    tmp = err[1:] - err[:-1]
    dw_stat = np.sum(tmp**2) / sse
    
    # log likelihood
    # biased sigma sq error variance estimate
    s2 = np.sum(err**2) / N
    llr = np.log(1/np.sqrt(2*math.pi*s2))*N - sse/(2*s2)

    # r2 
    stat = r2(y, y_pred, params)
    d = {'DW': dw_stat, 'loglikelihood': llr, 'F-stat': f_stat}
    stat.update(d)
    return stat

def r2(y, y_pred, params):
    "R2"
    N = len(y)
    num_params = len(params)
    # error (residual)
    err = y - y_pred
    # sum of sq error
    sse = np.sum(err**2)
    # deviation from the mean
    ym = y - np.mean(y)
    # total variance of y
    sst = np.sum(ym**2)
    # r2: explainable variance (total minus unexplainable)
    r2 = 1 - sse / sst
    # adjusted R2
    adj_r2 = 1 - (sse/(N-num_params)) / (sst/(N-1))

    r2 = np.round(r2,4)
    adj_r2 = np.round(adj_r2,4)
    return {'r2': r2, 'adj_r2': adj_r2}

# ------------------------------------------------------------
# bayesian inference
# ------------------------------------------------------------
def make_posterior_inference(create_model, X, y, N, nsteps=1000):
    """
        create_model: function returning guide and model
        X, y: 
        N: num samples
        nstep: num optimization step
    """

    model, delta_guide = create_model()
    # Perform inference
    svi = SVI(model, delta_guide, optim=SGD({"lr": 1e-3}), loss=Trace_ELBO())
    pyro.clear_param_store()
    loss = [svi.step(X, y) for _ in range(nsteps)]
    # instanteiate guide
    guide = delta_guide.laplace_approximation(X, y)
    # sampling from posterior
    samples = {
        k: v.flatten().detach().numpy()
        for k, v in Predictive(model, guide=guide, num_samples=N)(X).items()
    }
    return (samples, loss)

def pytorch_regression(X, y, niterations=1000, lr=0.05):
    """simple linear regression
        X: pytorch tensor
        y: pytorch tensor
    """
    nparams = X.shape[1]
    linear_reg_model = PyroModule[nn.Linear](nparams, 1)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(linear_reg_model.parameters(), lr=lr)

    def train(X, y):
        # run the model forward on the data
        y_pred = linear_reg_model(X).squeeze(-1)
        # calculate the mse loss
        loss = loss_fn(y_pred, y)
        # initialize gradients to zero
        optim.zero_grad()
        # backpropagate
        loss.backward()
        # take a gradient step
        optim.step()
        return loss

    loss = [train(X, y).item() for j in range(niterations)]
    return (linear_reg_model, loss)


# ------------------------------------------------------------
# model posterior distribution evaluation
# ------------------------------------------------------------
def pyro_posterior_samples(model, guide, X, num_samples=800):
    """generate fitted model samples

        model: 
        guide:
        X: pytorch tensor (indepentdent variables)
        num_samples: int
    
    """
    predictive = Predictive(model, guide=guide, num_samples=num_samples)
    return predictive(X)

def pyro_reg_summary(samples, y, lb=0.025, ub=0.975):
    ""

    def _collect_bias_weight(name):
        b = summary['linear.bias'][name].squeeze().detach().numpy()
        W = summary['linear.weight'][name].squeeze().detach().numpy()
        return np.append(b, W)

    summary = {}
    for k, v in samples.items():
        summary[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "lb".format(str(lb)): v.kthvalue(int(len(v) * lb), dim=0)[0],
            "ub".format(str(ub)): v.kthvalue(int(len(v) * ub), dim=0)[0],
        }
    W = _collect_bias_weight('mean')
    W_stderr = _collect_bias_weight('std')
    W_lb = _collect_bias_weight('lb')
    W_ub = _collect_bias_weight('ub')

    # t values
    ts_b = W / W_stderr
    N = len(y)
    degf = W.shape[0]
    # p values
    p_values =[2*(1-scipy.stats.t.cdf(np.abs(i),N-degf)) for i in ts_b]

    # rounding for printing
    W = np.round(W,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    W_stderr = np.round(W_stderr,4)
    W_lb = np.round(W_lb,4)
    W_ub = np.round(W_ub,4)

    df = pd.DataFrame()
    df["Coefficients"] = W
    df["std err"] = W_stderr
    df["t values"] = ts_b
    df["P>|t|"] = p_values
    df["t values"] = ts_b
    df['[' + str(lb)] = W_lb
    df[str(ub)+']'] = W_ub
    return df

def pyro_confidence_interval_plot(samples: dict, prob=0.89):
    """

    """
    precis_df = precis(samples, prob)
    p1, p2 = (1-prob)/2, 1-(1-prob)/2
    p1 = f"{100*p1:.1f}%"
    p2 = f"{100*p2:.1f}%"
    sns.pointplot(x=precis_df["mean"], y=precis_df.index, join=False)
    for i, node in enumerate(precis_df.index):
        sns.lineplot(x=precis_df.loc[node, [p1, p2]], y=[i, i], color="k")


def precis(samples: dict, prob=0.89):
    """Computes some summary statistics
    
        samples: pyro samples
    """

    # prob lower and upper bound
    p1, p2 = (1-prob)/2, 1-(1-prob)/2
    cols = ["mean","stddev",f"{100*p1:.1f}%",f"{100*p2:.1f}%"]
    df = pd.DataFrame(columns=cols, index=samples.keys())
    for k, v in samples.items():
        if type(v) == torch.Tensor:
            v = v.detach().numpy()
        df.loc[k]["mean"] = v.mean()
        df.loc[k]["stddev"] = v.std()
        q1, q2 = np.quantile(v, [p1, p2])
        df.loc[k][f"{100*p1:.1f}%"] = q1
        df.loc[k][f"{100*p2:.1f}%"] = q2

    return df
# ------------------------------------------------------------
# model plot
# ------------------------------------------------------------
def posterior_predictive_plot(x, y, alpha, beta, num_steps=100):
    """
        x, y: actual tensor
        alpha, beta: fitted line coefficient
        num_steps: x axis num steps
    """
    ax = plt.scatter(x, y)
    xx = np.linspace(x.min(), x.max(), num_steps)
    y_pred = alpha + beta * xx
    plt.plot(xx, y_pred, color='k')
    # find fitted line and then find residual
    pred = alpha + beta * x
    residual = y - pred
    for i in range(len(x)):
        xi = x[i]  # x location of line segment
        yi = y[i]  # observed endpoint of line segment
        # draw the line segment
        plt.plot(xi.repeat(2), np.stack([pred[i], yi]), color="k", alpha=0.4)

def pyro_single_variate_reg_line_plot(x, y, posterior_samples, 
                                alpha, beta, x_hat, y_hat,
                                prob=0.89):
    """
        x: numpy
        y: numpy
        posterior_samples: pyro samples dict
        x_hat: name of posterior samples of x 
        y_hat: name of posterior samples of y
        standardize: True if x value is standardized
        prob: confidence interval
    """

    num_steps = posterior_samples[x_hat].shape[-1]
    # x axis
    xx = np.linspace(x.min(), x.max(), num_steps)

    # mean predicted value
    m = [(posterior_samples[alpha] + posterior_samples[beta] * xx[i]).mean() for i in range(num_steps)]

    # get hdpi from posterior samples
    hpdi = dict()
    for k in [x_hat, y_hat]:
        hpdi[k] = np.vstack([az.hdi(posterior_samples[k].squeeze()[:,i], prob) for i in range(num_steps)])

    # Plot
    # plt.scatter(x, y, label="data", facecolor="C1", edgecolor="C1", alpha=1.0)
    plt.scatter(x, y, label="data", color="red", s=8)
    # confidence intervals
    plt.fill_between(xx, *hpdi[x_hat].T, color="black", alpha=0.5, label=x_hat)
    plt.fill_between(xx, *hpdi[y_hat].T, color="gray", alpha=0.5, label=y_hat)
    plt.plot(xx, m, color="black")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("regression line with HPDI bounds")

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class SimpleRegressionModel:
    "simple single variate regression"
    def __init__(self, model, x, y):
        self.x = x
        self.y = y
        self.model = model
        self.name = model.__name__
        
    def __call__(self, x, y=None):
        self.model(x, y)
        
    def train(self, guide, num_steps):
        pyro.clear_param_store()
        # self.guide = AutoMultivariateNormal(self)
        self.guide = guide(self)
        svi = SVI(
            model=self,
            guide=self.guide,
            optim=Adam({"lr": 1e-3}),
            loss=Trace_ELBO(),
        )
        loss = [svi.step(self.x, self.y) for _ in range(num_steps)]
        return loss

    def sampling(self, x, num_samples=1000, num_steps=100, 
                    use_prefix=True, resample=True):
        if resample:
            x = np.linspace(x.min(), x.max(), num_steps)
        samples = {
            k: v.detach().numpy()
            for k, v in Predictive(
                model=self,
                guide=self.guide,
                num_samples=num_samples,
            )(x).items()
        }
        # rename keys
        if use_prefix:
            keys = list(samples.keys())
            for key in keys:
                new_key = f"{self.name}/" + key
                samples[new_key] = samples.pop(key)
        return samples

class SimpleMultipleRegressionModel:
    def __init__(self, model, x, y):
        self.x = x
        self.y = y
        self.N = int(x.shape[-1])
        self.model = model
        self.name = model.__name__
        
    def __call__(self, x, y=None):
        self.model(x, y)
        # nm = self.name + "/" if self.name != "" else ""
        # alpha = pyro.sample(f"{nm}alpha", Normal(0., .2))
        # betas = pyro.sample(f"{nm}betas", Normal(0., .5).expand([self.N]).to_event(1))
        # sigma = pyro.sample(f"{nm}sigma", Exponential(.1))
        # mu = pyro.deterministic(f"{nm}mu", alpha + torch.matmul(x, betas))
        # with pyro.plate("obs", x.shape[0]):
        #     if y is None:
        #         # latent rv sampling
        #         return pyro.sample(f"{nm}D", Normal(mu, sigma))
        #     else:
        #         # observed rv sampling
        #         pyro.sample(f"{nm}D", Normal(mu, sigma), obs=y)
        
    def train(self, guide, num_steps):
        pyro.clear_param_store()
        self.guide = guide(self)
        svi = SVI(
            model=self,
            guide=self.guide,
            optim=Adam({"lr": 1e-3}),
            loss=Trace_ELBO(),
        )
        loss = [svi.step(self.x, self.y) for _ in range(num_steps)]
        return loss

    def sampling(self, X, num_samples=1000, num_steps=100):
        # x = np.linspace(x.min(), x.max(), num_steps)
        samples = {
            k: v.detach().numpy()
            for k, v in Predictive(
                model=self,
                guide=self.guide,
                num_samples=num_samples,
            )(X).items()
        }
        # rename keys
        keys = list(samples.keys())
        for key in keys:
            new_key = f"{self.name}/" + key
            samples[new_key] = samples.pop(key)
        return samples


class BayesianLinearRegression(PyroModule):
    "regression with standardize normal distributed param"
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(Normal(0., 1.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", Uniform(0., 10.))
        mu = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", Normal(mu, sigma), obs=y)
        return mu

def pyro_regression(X, y, niterations=1000, lr=0.05):
    ""
    nparams = X.shape[1]
    model = BayesianLinearRegression(nparams, 1)
    guide = AutoDiagonalNormal(model)
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    pyro.clear_param_store()      
    loss = [svi.step(X, y) for _ in range(niterations)]
    return (model, guide, loss)

def coef(samples):
    """
        samples: pyro sampling dict
    """
    res = {}
    for k in samples:
        res[k] = samples[k].mean()
    return res

# ------------------------------------------------------------
# predictive residual plots 
# ------------------------------------------------------------
def predictor_residual_plot(x1, x2, y, predictor_model, reg_model, 
                            num_samples=1000, num_train_steps=2000, num_steps=100):
    """

    """
    # regress x1 on x2
    mpred = SimpleRegressionModel(predictor_model, x1, x2)
    loss = mpred.train(AutoMultivariateNormal, num_train_steps)
    samples_m = mpred.sampling(x1, num_samples, num_steps, use_prefix=False)
    alpha = coef(samples_m)['alpha']
    beta = coef(samples_m)['beta']

    fig1 = plt.figure("Figure 1")
    posterior_predictive_plot(x1, x2, alpha, beta)

    pred = alpha + beta * x1
    residual = x2 - pred

    model = SimpleRegressionModel(reg_model, residual, y)
    loss = model.train(AutoMultivariateNormal, num_train_steps)
    samples_y = model.sampling(residual, 3000, use_prefix=False)

    fig2 = plt.figure("Figure 2")
    pyro_single_variate_reg_line_plot(residual, y, samples_y, 
                                        'alpha', 'beta', 'mu', 'y')