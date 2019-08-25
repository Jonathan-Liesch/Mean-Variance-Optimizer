import numpy as np
import pandas as pd
import math
import random
from matplotlib import pyplot as plt

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
    def __init__(self, names, means, std, cov):
        self.assets = []
        for i in range(len(names)):
            self.assets.append(asset(names[i], means[i], std[i]))
        
        self.mean_vec = means
        self.sigma = cov
    
    
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

def efficient_frontierMC(universe, resolution = 5):
    assert resolution >0
    min_mean = min(univ.mean_vec)
    max_mean = max(univ.mean_vec)
    mean_range = max_mean - min_mean 
    returns = []

    for i in range(resolution+1):
        returns.append(min_mean + i* mean_range/resolution)
    
    exp = []
    std = []
    min_var = 100000
    min_var_ret = 0
    for i in returns:
        p = Markowitz_Risk_Min(univ, i)
        if p.std < min_var:
            min_var = p.std
            min_var_ret = p.mean
        exp.append(p.mean)
        std.append(p.std)
    print(Markowitz_Risk_Min(univ, min_var_ret))
    print(Markowitz_Risk_Min(univ, max_mean))
    print(Markowitz_Risk_Min(univ, min_mean))
    return np.array(std), np.array(exp)







#################
#   Main
#################

from stockDataClean import stocks, mu, std, sigma


univ = universe(stocks, mu, std, sigma)

ef_std, ef_exp = efficient_frontierMC(univ, 1000)


std, exp = random_portfolios(univ, 100)


s, e = base_portfolios(univ)
plt.style.use('seaborn')
plt.xlabel('std')
plt.ylabel('return')

plt.scatter(ef_std, ef_exp)
plt.scatter(s, e)
plt.scatter(std, exp)

plt.show()

