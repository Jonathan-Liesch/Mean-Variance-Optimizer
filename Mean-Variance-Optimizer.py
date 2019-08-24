import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt

asset_counter = 0

class asset:
    def __init__(self, name, mean, stdDev):
        global asset_counter
        self.id = asset_counter
        self.name = name
        self.mean = mean
        self.std = stdDev
        asset_counter += 1

    def __str__(self):
        return "#####################\n" + \
        "id: %(id)d \nname: %(name)s \nE[r]: %(mean)f \nStd: %(std)f\n" \
            % {'id': self.id, 'name': str(self.name), 'mean': self.mean, 'std': self.std} \
                + "#####################\n"

def asset_mean_vec(assets):
    returns = []
    for asset in assets:
        returns.append(asset.mean)
    return np.asarray(returns)

def port_mean(asset_means_vec, weights):
    return np.matmul(asset_mean_vec(assets), weights)

def port_std(weights,sigma):
    return round(math.sqrt(np.matmul(weights.T, np.matmul(sigma, weights))), 4)


portfolio_count = 0
class portfolio:
    def __init__(self, assets, weights, covar):
        global portfolio_count
        self.id = portfolio_count 
        self.assets = assets
        self.weights = weights
        self.sigma = covar
        self.mean = port_mean(asset_mean_vec(assets), weights)
        self.std = port_std(weights, sigma)
        portfolio_count += 1

    def __str__(self):
        return "#####################\n" + \
        "id: %(id)d \nweights: %(weights)s \nE[R]: %(mean)f \nStd: %(std)f\n" \
            % {'id': self.id, 'weights': str(self.weights), 'mean': self.mean, 'std': self.std} \
                + "#####################\n"

#################
#   Main
#################

a1 = asset("goog", 10, 16)
a2 = asset("sp", 8, 12)
assets = [a1, a2]

sigma = np.array([[16**2, 0],\
                  [0, 12**2]])

f_weight = np.arange(0, 1.000000001, 0.001)

expected_return = []
standard_deviation = []

for w in f_weight:
    weight = np.array([w, 1-w])
    p = portfolio(assets, weight, sigma) 
    expected_return.append(p.mean)
    standard_deviation.append(p.std)

expected_return = np.asarray(expected_return)
standard_deviation = np.asarray(standard_deviation)

plt.style.use('seaborn')
plt.scatter(standard_deviation, expected_return)

plt.show()
