# -*- coding: utf-8 -*-
""" nearest_neighbor.py
    Compute the nearest neighbor algorithm for the TSP.
"""

__all__ = ['NearestNeighbor', 'nearest_neighbor', 'dynamic_nearest_neighbor']
__version__ = '0.2'
__author__ = 'Emanuele Lovera'

from random import randint

from utils.tsp_utils import get_weights_matrix, static_objfun, dynamic_objfun, get_fengyun_weights

import numpy as np


class NearestNeighbor():
    def __init__(self, filename, card_max=None, is_static=True, is_fengyun=False):
        if is_fengyun:
            self.weights, self.N_CITIES = get_fengyun_weights(filename)
        else:
            self.weights, self.N_CITIES = get_weights_matrix(filename, card_max=card_max, is_static=is_static)

        self.is_static = is_static
        self.best_solution = None
        self.best_cost = float('Inf')

    def run_nn(self, iterations=5):
        for i in range(iterations):
            if self.is_static:
                sol = nearest_neighbor(self.weights, self.N_CITIES)
                cost = static_objfun(sol, self.weights, self.N_CITIES)
            else:
                sol = dynamic_nearest_neighbor(self.weights, self.N_CITIES)
                cost = dynamic_objfun(sol, self.weights, self.N_CITIES)

            if cost < self.best_cost:
                self.best_sol = sol
                self.best_cost = cost
        return self.best_cost, self.best_sol


def nearest_neighbor(weights, N_CITIES):
    """Find the best solution for the static TSP with the nearest neighbor algorithm."""
    nn_solution = []
    available = set(range(N_CITIES))
    nn_solution.append(randint(0, N_CITIES - 1))

    available.remove(nn_solution[-1])
    for i in range(N_CITIES - 1):
        city = nn_solution[-1]
        minim = np.amin(weights[city, list(available)])
        next_city = -1
        for iterator in available:
            if minim == weights[city, iterator]:
                next_city = iterator
                break

        nn_solution.append(next_city)
        available.remove(next_city)
    return nn_solution


def dynamic_nearest_neighbor(weights, N_CITIES):
    """Find the best solution for the dynamic TSP with the nearest neighbor algorithm."""
    nn_solution = []
    available = set(range(N_CITIES))

    nn_solution.append(randint(0, N_CITIES - 1))
    available.remove(nn_solution[-1])
    t = 1
    for i in range(N_CITIES - 1):
        city = nn_solution[-1]
        minim = np.min(weights[t, city, list(available)])
        next_city = -1
        for iterator in available:
            if minim == weights[t, city, iterator]:
                next_city = iterator
                break
        nn_solution.append(next_city)
        available.remove(next_city)
        t = t + 1
    return nn_solution
