# -*- coding: utf-8 -*-
""" nn_main.py
    Main for the NearestNeighbor class
"""

from utils.nearest_neighbors import NearestNeighbor

__all__ = ['NearestNeighbor']
__version__ = '0.1'
__author__ = 'Emanuele Lovera'

if __name__ == '__main__':
    filename = "./instances/iridium_weights_matricies.npz"
    nn = NearestNeighbor(filename)
    best_cost, best_sol = nn.run_nn()
    print("NearestNeighbor: ")
    print(" best_cost={}".format(best_cost))
    print(" best_sol={}".format(best_sol))
