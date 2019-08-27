import numpy as np
import pandas as pd
import math
import random
from matplotlib import pyplot as plt
import matplotlib.colors
import matplotlib.cm

asset_counter = 0
class asset:
    def __init__(self, name, mean, stdDev):
        global asset_counter
        self.id = asset_counter
        self.name = name
        self.mean = mean
        self.std = stdDev
        self.var = stdDev**2
        asset_counter += 1

    def __str__(self):
        return "#####################\n" + \
        "id: %(id)d \nname: %(name)s \nE[r]: %(mean)f \nStd: %(std)f\n" \
            % {'id': self.id, 'name': str(self.name), 'mean': self.mean, 'std': self.std} \
                + "#####################\n"

class universe:
    def __init__(self, names, means, std, cov, rf):
        self.assets = []
        for i in range(len(names)):
            self.assets.append(asset(names[i], means[i], std[i]))
        self.rf = rf
        self.mean_vec = means
        self.sigma = cov
        self.excess_mean_vec = self.mean_vec - rf
    
portfolio_count = 0
class portfolio:
    def __init__(self, universe, weights):
        assert len(universe.mean_vec) == len(weights), "number of weights don't match number of assets"
        global portfolio_count
        self.id = portfolio_count 
        self.assets = universe.assets
        self.weights = weights
        self.mean = np.matmul(universe.mean_vec, weights)
        self.var = np.matmul(weights.T, np.matmul(sigma, weights))
        self.std = math.sqrt(self.var)
        self.sharpe = (self.mean - universe.rf)/self.std
        portfolio_count += 1

    def __str__(self):
        buffer = "#####################\n"
        returns = str(round(self.mean, 2)) + " = "
        std = str(round(self.std, 2)) + "**2 = "
        for i in range(len(self.weights)):
            returns += str(round(self.weights[i], 2)) +  "x" + str(round(self.assets[i].mean, 2)) \
                + "(" + self.assets[i].name + ")"
            std += str(round(self.weights[i], 2)) +  "x" + str(round(self.assets[i].std, 2)) \
                + "**2(" + self.assets[i].name + ") + "
            if i != len(self.weights) - 1:
                returns += " + "
        std += "...\n"
        return  buffer + returns + "\n" + std + buffer

def random_weights(asset_num):
    cuts = []
    for i in range(asset_num-1):
        cuts.append(random.random())
    cuts.sort()
    weights = []
    start = 0
    for cut in cuts:
        weights.append(cut-start)
        start = cut
    weights.append(1-cuts[-1])
    return np.array(weights)

def random_portfolios(univ, port_num):
    exp = []
    stdev = []
    for i in range(port_num):
        p = portfolio(univ, random_weights(len(univ.assets)))
        exp.append(p.mean)
        stdev.append(p.std)
    return np.array(stdev), np.array(exp)

def base_portfolios(univ):
    mean = []
    std = []
    num = len(univ.assets)
    for i in range(num):
        weights = [0] * num
        weights[i] = 1
        p = portfolio(univ, np.array(weights))
        mean.append(p.mean)
        std.append(p.std)
    return np.array(std),np.array(mean)

def Markowitz_Risk_Min(univ, expected_return):
    mu = univ.mean_vec
    sigma = univ.sigma
    sigma_inv = np.linalg.inv(sigma)
    one = np.array([1]*len(univ.mean_vec)).T
    
    a = -1 * np.matmul(one.T, np.matmul(sigma_inv, one))
    b = -1 * np.matmul(mu.T, np.matmul(sigma_inv, one))
    c = b
    d = -1 * np.matmul(mu.T, np.matmul(sigma_inv, mu))

    A = np.array([[a,b],[c,d]])
    A_inv = np.linalg.inv(A)
    y = np.array([1, expected_return])
    Lagrangian_mult = np.matmul(A_inv, y)
    weights = -1* Lagrangian_mult[0] * np.matmul(sigma_inv, one) - Lagrangian_mult[1]* np.matmul(sigma_inv, mu)
    return portfolio(univ, weights)

def Markowitz_Risk_Min_Vec(univ, expected_return):
    p = Markowitz_Risk_Min(univ, expected_return)
    return p.std, p.mean, p.sharpe

def Min_Var_Mean(univ):
    one = np.array([1]*len(univ.assets))
    s_inv = np.linalg.inv(sigma)
    numerator = np.matmul(mu, np.matmul(s_inv, one))
    denominator = np.matmul(one, np.matmul(s_inv, one))
    return numerator/denominator

def Min_Var_Portfolio(univ):
    return Markowitz_Risk_Min(univ, Min_Var_Mean(univ))

def Min_Var_Portfolio_Vec(univ):
    return Markowitz_Risk_Min_Vec(univ, Min_Var_Mean(univ))

def Market_Portfolio(univ):
    sigma_inv = np.linalg.inv(univ.sigma)
    one = np.array([1]*len(univ.assets))
    numerator = np.matmul(sigma_inv, univ.excess_mean_vec)
    denominator = np.matmul(one, np.matmul(sigma_inv, univ.excess_mean_vec))
    weights = np.array(numerator/denominator)
    return portfolio(univ, weights)

def Market_Portfolio_Vec(univ):
    p = Market_Portfolio(univ)
    return p.std, p.mean, p.sharpe

def bullet_curve(universe, resolution, maximum, minimum):
    assert resolution > 0, "resolution can't be non-negative"
    assert minimum<maximum, "cannot have minimum>=maximum"
    min_mean = minimum
    max_mean = maximum
    mean_range = max_mean - min_mean 
    returns = []
    for i in range(resolution+1):
        returns.append(min_mean + i* mean_range/resolution)
    exp = []
    std = []
    sharpe = []
    for i in returns:
        p = Markowitz_Risk_Min(univ, i)
        exp.append(p.mean)
        std.append(p.std)
        sharpe.append(p.sharpe)
    return np.array(std), np.array(exp), np.array(sharpe)

def efficient_frontier(universe, resolution, maximum):
    minimum = Min_Var_Mean(universe)
    return bullet_curve(universe, resolution, maximum, minimum)

def capital_allocation_line(universe):
    b = universe.rf
    run, rise = Market_Portfolio_Vec(universe)[:2]
    m = (rise-b)/run
    x = np.linspace(0, max([asset.std for asset in universe.assets]), 100)
    y = m*x+b
    return x, y

###################
# Plotting
###################
def plot_assets(universe):
    std, exp = base_portfolios(universe)
    plt.scatter(std, exp, c = 'cadetblue')

def plot_bullet_curve(universe):
    market_mean = Market_Portfolio(universe).mean
    upperbound = max(max(asset.mean for asset in universe.assets)*1.5, market_mean+0.05)
    std, exp, sharpe = bullet_curve(universe, 1000, upperbound, 0.0001)
    plt.scatter(std, exp, c=sharpe, cmap = 'jet', marker = ".", norm = matplotlib.colors.PowerNorm(5))
    plt.colorbar(label = 'Sharpe Ratio',ticks = [1.6, 1.575, 1.55, 1.50, 1.40, 1.3, 1.2, 1.1]).ax.set_ylabel('Sharpe Ratio', rotation = 270, labelpad=25)

def plot_efficient_frontier(universe, heat_map = False):
    market_mean = Market_Portfolio(universe).mean
    upperbound = max(max(asset.mean for asset in universe.assets)*1.5, market_mean+0.05)
    ef_std, ef_exp, ef_sharpe = efficient_frontier(univ, 1000, upperbound)
    if heat_map == True:
        resolution = 100
        cmap = matplotlib.cm.Blues(np.linspace(0,1,resolution))
        cmap = matplotlib.colors.ListedColormap(cmap[int(resolution/4):,:-1])
        minimum = min(ef_sharpe)
        maximum = max(ef_sharpe)
        norm = matplotlib.colors.Normalize(vmin=minimum, vmax=maximum)
        plt.scatter(ef_std, ef_exp, c=ef_sharpe, cmap = cmap, marker = ".", norm = norm)
        plt.colorbar().ax.set_ylabel('Sharpe Ratio', rotation = 270, labelpad=25)
    else:
        plt.scatter(ef_std, ef_exp, marker = ".", c = "darkslateblue")

def plot_CAL(universe):
    x,y = capital_allocation_line(universe)
    plt.plot(x, y, zorder = 1)

def plot_Market_Portfolio(universe):
    std, mean = Market_Portfolio_Vec(universe)[:2]
    plt.scatter(std, mean, c='teal', zorder=2)

def plot_Min_Var_Portfolio(universe):
    std, mean = Min_Var_Portfolio_Vec(universe)[:2]
    plt.scatter(std, mean, c='teal', zorder=2)

#################
#   Main
#################
from stockDataClean import stocks, mu, std, sigma

univ = universe(stocks, mu, std, sigma, 0.02)

plt.style.use('seaborn')
plt.title("Efficient Frontier")
plt.xlabel('Standard Deviation')
plt.ylabel('Expected Return')

plot_assets(univ)
plot_efficient_frontier(univ, heat_map = True)
plot_CAL(univ)
plot_Market_Portfolio(univ)
plot_Min_Var_Portfolio(univ)

plt.show()

