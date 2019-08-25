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

#################
#   Main
#################

from stockDataClean import stocks, mu, std, sigma
import random

univ = universe(stocks, mu, std, sigma)

p1 = portfolio(univ, np.array([0,0,0,0,0,1]))

print(stocks, mu, std, sep='\n')

print(p1.mean, p1.std)
print(p1)


# plt.style.use('seaborn')
# plt.scatter(standard_deviation, expected_return)

# plt.show()
