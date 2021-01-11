import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerErrorbar

# Math functions
def _double_factorial(d):
    if d == 0 or d == 1:
        return 1
    return d * _double_factorial(d-2)

def prob_pt_in_ball(d):
    return (np.pi/2)**(np.floor(d/2)) / _double_factorial(d)

def prob_empty_ball(d, n):
    return binom.pmf(0, n, prob_pt_in_ball(d))

def true_volume(d):
    return 2**d * prob_pt_in_ball(d)

def true_se(d, n):
    p = prob_pt_in_ball(d)
    var_volume_estimator = (1/n) * (4**d) * p * (1-p)
    return var_volume_estimator**0.5

def expected_N_ball(d, n):
    return true_volume(d)*n / 2**d

# Simulation functions
def sample_from_cube(d, n, seed=0):
    np.random.seed(seed)
    rand_pts = np.random.uniform(size=(n,d))
    rand_pts = 2*rand_pts - 1 # rescale to U(-1,1)
    distances = np.linalg.norm(rand_pts, axis=1)
    return distances <= 1

# Statistics functions
def _errors(d, sample_sizes, seed=0):
    relative_errors = []
    volume = true_volume(d)
    for n in sample_sizes:
        is_in_ball = sample_from_cube(d, n, seed=seed)
        N_ball_est = np.sum(is_in_ball)
        volume_est = 2**d * N_ball_est / n
        rel_error = np.abs(volume_est - volume)/volume
        relative_errors.append(rel_error)
    asymptotic_errors = np.ones(len(sample_sizes))/np.sqrt(sample_sizes)
    return np.array(relative_errors), asymptotic_errors

def _estimates(dimensions, n, seed=0):
    volume_mean_ests = []
    volume_se_ests = []
    for d in dimensions:
        is_in_ball = sample_from_cube(d, n, seed=seed)
        mean_est = np.mean(2**d * is_in_ball)
        var_est = 4**d * np.var(is_in_ball, ddof=1) / n
        volume_mean_ests.append(mean_est)
        volume_se_ests.append(np.sqrt(var_est))
    return np.array(volume_mean_ests), np.array(volume_se_ests)

# Plotting functions for 3 kinds of plots
def to_power10(n):
    power = int(np.log10(n))
    return f"$10^{power}$"

def plot_errors(dimensions, sample_sizes, ncols=3, width=4.0, height=4.0, seed=0):
    # dirty but effective function for now. In the future I should generalize it
    nrows = int(np.ceil(len(dimensions) / ncols))
    figsize = (width*ncols, height*nrows)
    _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True)
    if ax.ndim == 1:
        ax = np.expand_dims(ax, axis=0)
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            dim_ind = i*ncols + j
            if dim_ind >= len(dimensions):
                j -= 1
                break
            d = dimensions[dim_ind]
            relative_errors, asymptotic_errors = _errors(d, sample_sizes, seed=seed)
            col.plot(sample_sizes, relative_errors, label="Observed")
            col.plot(sample_sizes, asymptotic_errors, label="Asymptotic")
            col.set_xlabel("Number of points sampled")
            col.set_ylabel("Relative error")
            col.set_title(f"{d}-ball volume estimates")
            col.legend()
    for col_ind in range(j+1, ncols): # don't plot empty axes
        ax[nrows-1, col_ind].set_axis_off()

def plot_errorbars(dimensions, sample_sizes, true_or_est_se, title, seed=0, width=8.0, height=6.0):
    true_means = [true_volume(d) for d in dimensions]
    _, ax = plt.subplots(figsize=(width, height))
    hh = []
    for n in sample_sizes:
        if true_or_est_se == "true":
            means = true_means
            ses = np.array([true_se(d, n) for d in dimensions])
        else:
            means, ses = _estimates(dimensions, n, seed=seed)
        h = ax.errorbar(dimensions, means, yerr=2*ses, label=f"n = {to_power10(n)}", 
                        fmt='o', ms=3)
        hh.append(h)
    h = ax.scatter(dimensions, true_means, c='k', label="True volume")
    hh.append(h)
    ax.legend(hh, [H.get_label() for H in hh], borderpad=1.5,
              handler_map={type(hh[0]): HandlerErrorbar(yerr_size=1.5)},
              labelspacing=2.5)
    ax.set_ylim(bottom=0) # cut off lower error bars at 0 b/c that's impossible
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Volume")
    ax.set_title(title)
    ax.set_xticks(dimensions)

def plot_across(dimensions, sample_sizes, statistic, title, labels, ylabel, width=8.0, height=6.0):
    plt.figure(figsize=(width, height))
    for i, n in enumerate(sample_sizes):
        stats = [statistic(d, n) for d in dimensions]
        plt.plot(dimensions, stats, label=labels[i])
    plt.xlabel("Dimension")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(dimensions)
    plt.legend()