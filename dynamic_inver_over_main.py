# -*- coding: utf-8 -*-
""" dynamic_inver_over_main.py
    Main for dynamic inver over algorithm.
"""
__author__ = 'Emanuele Lovera'

import numpy as np
from time import time

from inver_over.dynamic_inver_over import DynamicInverOver
from utils.nearest_neighbors import NearestNeighbor


if __name__ == '__main__':
    #filename = "C:/Users/Emanuele Lovera/Documents/Python/DynamicTSP/instances/iridium_weights_matricies.npz"
    filename = "./instances/iridium_weights_matricies.npz"
    t1 = time()
    dio = DynamicInverOver(filename)
    best_cost, best_solution = dio.run_dio()
    t2 = time()

    t3 = time()
    nn = NearestNeighbor(filename)
    nn_cost, nn_sol = nn.run_nn()
    t4 = time()

    print("NearestNeighbor: ")
    print(" best_cost={}".format(nn_cost))
    print(" best_sol={}".format(nn_sol))
    print(" computed in {}".format(t3 - t4))

    print("GLOBAL BEST")
    print("  solution={}".format(best_solution))
    print("  cost   = {}".format(best_cost))
    print(" computed in {}".format(t2-t1))
    print("\n")
