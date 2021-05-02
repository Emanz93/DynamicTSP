# -*- coding: utf-8 -*-
""" ant_static_main.py
Main for ant algorithms.
"""
__author__ = 'Emanuele Lovera'

import numpy as np

from time import time
from ants.ant_colony_system import AntColonySystem
from utils.nearest_neighbors import NearestNeighbor

if __name__ == '__main__':
    filename_weights = "./instances/iridium_weights_matricies.npz"
    acs = AntColonySystem(filename_weights)
    best_cost, best_solution = acs.run_ants()

    nn = NearestNeighbor(filename_weights, is_static=True)
    nn_cost, nn_solution = nn.run_nn()
    print("NN SOLUTION")
    print("  solution={}".format(nn_solution))
    print("  cost   = {}".format(nn_cost))

    print("STATIC ANT BEST")
    print("  solution={}".format(best_solution))
    print("  cost   = {}".format(best_cost))
    print("\n")
