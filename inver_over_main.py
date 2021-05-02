# -*- coding: utf-8 -*-
""" inver_over_main.py
    Main for the inver over class.
"""
__author__ = 'Emanuele Lovera'

from inver_over.inver_over import InverOver
from time import time

def inver_over_result_collector(filename):
    """Collect some results from multiple runs."""
    with open("InverOver_results.txt", "w") as f:
        for i in range(25):
            io = InverOver(filename)
            best_cost, best_solution, nn_cost, nn_solution = io.run_io()
            f.write("Inver Over\n")
            f.write("  Iteration {}\n".format(i))
            f.write("    nn_cost={}\n".format(str(nn_cost)))
            f.write("    nn_solution={}\n".format(str(nn_solution)))
            f.write("    best_cost={}\n".format(str(best_cost)))
            f.write("    best_solution={}\n".format(str(best_solution)))

        f.write("\n\n\n")

if __name__ == '__main__':
    filename = "./instances/iridium_weights_matricies.npz"

    for i in range(1):
        t1 = time()
        io = InverOver(filename, population_size=20, timer_limit=2000)
        best_cost, best_sol, nn_cost, nn_solution = io.run_io()
        t2 = time()

        print("NN cost =", nn_cost)
        print("NN sol =", nn_solution)
        print("Best cost =", best_cost)
        print("Best sol =", best_sol)
        print(" in {:6.2f} seconds.".format(t2-t1))
