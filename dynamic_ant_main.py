# -*- coding: utf-8 -*-
""" dynamic_ant_main.py
Main for ant algorithms.
"""
__author__ = 'Emanuele Lovera'

from time import time
import numpy as np
from ants.S_ACO import DynamicAntColonySystem
from utils.nearest_neighbors import NearestNeighbor
from utils.statistics import compute_statistics


def parameter_discoverer(filename, card_max):
    """Discover the (optimal) parameter of the ACO with a brute force approach.
    Parameters:
        filename : path of the weights matrix.
        card_max : number of cities.
    """
    for beta in range(1, 7):
        for alpha in range(1, 7):
            for type in ["MMAS", "AS"]:
                c = []
                for i in range(5):
                    acs = DynamicAntColonySystem(filename, card_max=card_max, beta=beta, alpha=alpha, type=type)
                    best_cost, best_solution = acs.run_ants()
                    c.append(best_cost)
                print("alpha={}, beta={}, type={}".format(alpha, beta, type))
                print("  cost = {}\n".format(min(c)))


def statistical_collector(n, card_max, is_fengyun=False):
    """Perform the runs of the ACO. It can run also multiple time and with a variable subset of cities.
    Parameters:
        n : number of runs.
        card_max : number of cities.
    """
    costs = []
    sols = []
    times = []
    best_performances = []
    avg_performances = []
    worst_performances = []
    glob_best_change = []
    p_bi = []
    p_gi = []
    n_bi = []
    n_gi = []
    diversity = []
    avg_time = []
    ls_num_improvements = []
    ls_rate_improvements = []
    acs_objs = []

    for i in range(n):
        t1 = time()
        acs = DynamicAntColonySystem(filename, card_max=card_max, is_fengyun=is_fengyun)
        best_cost, best_solution = acs.run_ants()
        t2 = time()
        print("{}: S_ACO SOLUTION".format(i))
        print("  solution={}".format(best_solution))
        print("  cost   = {}".format(best_cost))
        print("  computed in {}".format(t2 - t1))

        costs.append(best_cost)
        sols.append(best_solution)
        times.append(t2 - t1)

        bst, avg, wst = acs.get_performance_arr()
        best_performances.append(bst)
        avg_performances.append(avg)
        worst_performances.append(wst)
        glob_best_change.append(acs.get_change_global_min())

        loc_p_bi, loc_p_gi, loc_n_bi, loc_n_gi = acs.get_bi_gi()
        p_bi.append(loc_p_bi)
        p_gi.append(loc_p_gi)
        n_bi.append(loc_n_bi)
        n_gi.append(loc_n_gi)

        diversity.append(acs.get_diversity())

        avg_time.append(acs.get_avg_time())

        loc_ls_num_improvements, loc_ls_rate_improvements = acs.get_ls_improvements()
        ls_num_improvements.append(loc_ls_num_improvements)
        ls_rate_improvements.append(loc_ls_rate_improvements)

        min_idx = np.argmin(np.array(costs))

        acs_objs.append(acs)

    print("\n\nFINAL RESULTS:")
    print("  best_cost={}, best_sol={}".format(costs[min_idx], sols[min_idx]))
    print("  avg_time={}, min_time={}, max_time={}".format(sum(times) / len(times), min(times), max(times)))

    best_performances, avg_performances, best_robusteness, avg_robusteness = compute_statistics(best_performances,
                                                                                                avg_performances,
                                                                                                worst_performances)

    # save it in a file.
    np.savez_compressed("./FEN1000_S_ACO_aio.npz", costs=np.array(costs), sols=np.array(sols),
                        times=np.array(times), p_bi=np.array(p_bi), p_gi=np.array(p_gi), n_bi=np.array(n_bi), n_gi=np.array(n_gi),
                        ls_num_improvements=np.array(ls_num_improvements), ls_rate_improvements=np.array(ls_rate_improvements),
                        glob_best_change=np.array(glob_best_change), diversity=np.array(diversity), avg_time=np.array(avg_time),
                        best_performances=np.array(best_performances), avg_performances=np.array(avg_performances),
                        best_robusteness=np.array(best_robusteness), avg_robusteness=np.array(avg_robusteness))


def run(n, card_max):
    for i in range(n):
        t1 = time()
        acs = DynamicAntColonySystem(filename, card_max=card_max)
        best_cost, best_solution = acs.run_ants()
        t2 = time()
        print("{}: S_ACO SOLUTION".format(i))
        print("  solution={}".format(best_solution))
        print("  cost   = {}".format(best_cost))
        print("  computed in {}".format(t2 - t1))


if __name__ == '__main__':
    filename = "./instances/fengyun_1000.npz"
    card_max = 1000

    t1 = time()
    nn = NearestNeighbor(filename, card_max=card_max, is_static=False)
    nn_cost, nn_solution = nn.run_nn()
    t2 = time()
    print("NN SOLUTION")
    print("  solution={}".format(nn_solution))
    print("  cost   = {}".format(nn_cost))
    print("  computed in {}".format(t2 - t1))
    print("\n")

    # parameter_discoverer(filename)

    statistical_collector(1, card_max)

    # run(10, card_max)
