# -*- coding: utf-8 -*-

""" tsp_utils.py
This file contains general and useful functions for managing tsp.
"""
__all__ = ['static_objfun', 'dynamic_objfun', 'get_weights_matrix', 'get_candidate_list', 'brute_force', 'reverse_path',
           'swap', 'init_population', 'opt_3', 'get_best_solution', 'diversity', 'inver_over', 'get_fengyun_weights' ]
__author__ = 'Emanuele Lovera'

from copy import copy
from itertools import permutations
from random import random, sample, choice

import numpy as np


def static_objfun(tour, weights, length):
    """Static calculation of the objective function. It doesn't use the time feature.
    Parameters:
        tour : list of cities.
        weights : weight matrix.
        length : lenght of the tour.
    Returns:
        cost : static cost of the tour.
    """
    cost = 0
    i = 0
    while i < length:
        cost = cost + weights[tour[i], tour[(i + 1) % length]]
        i = i + 1
    return cost


def dynamic_objfun(tour, weights, length):
    """Dynamic objective function.
    Parameters:
        tour : tour.
        weights : weight matrix.
        length : length of the tour.
    Returns:
        cost : dynamic cost of the tour.
    """
    i = 0  # index.
    cost = 0
    while i < length:
        cost = cost + weights[i, tour[i], tour[(i + 1) % length]]
        i = i + 1
    return cost


def get_weights_matrix(filename, card_max=None, is_static=False, time=0):
    """Get the weight matrix from the filename.
    Parameter:
        filename : Filename of the npz file.
        static : True if the tsp is static, False otherwise.
        time : which time to select if  hte problem is static.
    Returns:
        weights : weights matrix.
    """
    # test the filename type.
    if filename.endswith(".npz") or filename.endswith(".npy"):
        loaded = np.load(filename)
        weights = loaded['w']  # weight matrix
        loaded.close()

        # if I want to use a dynamic tsp as static
        if is_static:
            weights = weights[time]
    else:
        weights = np.loadtxt(filename, dtype=int)

    if not card_max is None and card_max < np.shape(weights)[1] and card_max > 0:
        N_CITIES = card_max
        if is_static is True:
            weights = weights[:card_max, :card_max]
        else:
            weights = weights[:card_max, :card_max, :card_max]
    else:
        N_CITIES = np.shape(weights)[1]

    return weights, N_CITIES


def get_fengyun_weights(filename):
    weights = np.memmap(filename, dtype=np.float32, mode='r+', shape=(1550, 1550, 1550))
    return weights, 1550

def get_candidate_list(filename, static=False):
    """Retreive the candidate list from the memory.
    Parameters:
        filename : name of the cl file.
        static : if True it takes only the cl at time 0. False otherwise takes all.
    Returns:
        cl : confusion matrix.
    """
    loaded = np.load(filename)
    cl = loaded['cl']  # weight matrix
    loaded.close()

    if static:
        cl = cl[0]
    return cl


def brute_force(weights):
    """From a weights matrix fount via brute force approach the best solution.
    Not feasible for more than 100 cities.
    Parameters:
        weights : weights matrix.
    """
    opt_sol = None
    opt_cost = float('Inf')
    n_city = np.shape(weights)[1]
    for sol in permutations(range(n_city)):
        cost = static_objfun(sol, weights, n_city)
        if opt_cost > cost:
            print("Found new solution:")
            print("  sol={}".format(sol))
            print("  cost={}".format(cost))
            opt_cost = cost
            opt_sol = sol
    print("opt_sol={}".format(opt_sol))
    print("opt_cost={}".format(opt_cost))


def reverse_path(tour, start, stop, N_CITIES):
    """Revert the a sequence of cities from start index to end index.
    The input list tour is considered as a circular array.
    Parameter:
        tour : circular array.
        start : starting index of the swap.
        stop : stopping index.
        N_CITIES : length of the tour.
    """
    if start < stop:
        while start < stop:
            tmp = tour[start]
            tour[start] = tour[stop]
            tour[stop] = tmp
            start = ((start + 1) % N_CITIES)
            stop = ((stop - 1) % N_CITIES)
    else:
        while True:
            tmp = tour[start]
            tour[start] = tour[stop]
            tour[stop] = tmp
            if start == stop or ((start + 1) % N_CITIES) == stop:
                break
            start = ((start + 1) % N_CITIES)
            stop = ((stop - 1) % N_CITIES)


def swap(arr, elem_1, elem_2):
    """Swap the position of the two elements inside the array.
    Parameter:
        arr : input list.
        elem_1 : first elem to swap.
        elem_2 : second elem to swap.
    """
    idx1 = arr.index(elem_1)
    idx2 = arr.index(elem_2)
    tmp = arr[idx1]
    arr[idx1] = arr[idx2]
    arr[idx2] = tmp


def init_population(POP_SIZE, N_CITIES):
    """Init randomly the population. The size of the population is expressed by
    POP_SIZE and the range with N_CITIES.
    Parameters:
        POP_SIZE : (int) population size
        N_CITIES : (int) number of cities.
    Returns:
        population : population.
    """
    population = []
    r = list(range(N_CITIES))
    for i in range(POP_SIZE):
        population.append(sample(r, N_CITIES))
    return population


def opt_3(sol, weights, N_CITIES, is_static=True):
    """Perform the 3-opt local search.

    Parameters:
        sol : initial solution. (list)
        weights : weight matrix.
        N_CITIES : length of the list.
        is_static : True if the fitness si static. False otherwise.

    Returns:
        cost : cost of the best local solution
        local_sol : best local solution
    """
    # get 6 random indices and take them sorted.
    indices = sample(range(N_CITIES), 6)
    indices.sort()
    [i, j, k, l, m, n] = indices

    opts = []
    opts_costs = []

    # opt1
    opts.append(copy(sol))
    reverse_path(opts[0], j, k, N_CITIES)  # swap(j, k)

    # opt2
    opts.append(copy(sol))
    reverse_path(opts[1], l, m, N_CITIES)  # swap(m, l)

    # opt3
    opts.append(copy(sol))
    reverse_path(opts[2], n, i, N_CITIES)  # swap(n, i)

    # opt4
    opts.append(copy(opts[0]))  # swap(j, k)
    reverse_path(opts[3], m, l, N_CITIES)  # swap(m, l)

    # opt5
    opts.append(copy(opts[1]))  # swap(m, l)
    reverse_path(opts[4], n, i, N_CITIES)  # swap(n, i)

    # opt6
    opts.append(copy(opts[2]))  # swap(n, i)
    reverse_path(opts[5], j, k, N_CITIES)  # swap(j, k)

    # opt7
    opts.append(copy(opts[0]))  # swap(j, k)
    reverse_path(opts[6], m, l, N_CITIES)  # swap(m, l)
    reverse_path(opts[6], n, i, N_CITIES)  # swap(n, i)

    for i in range(7):
        if is_static:
            opts_costs.append(static_objfun(opts[i], weights, N_CITIES))
        else:
            opts_costs.append(dynamic_objfun(opts[i], weights, N_CITIES))

    idx = np.argmin(opts_costs)

    return opts_costs[idx], opts[idx]


def get_best_solution(tours, N, weights, N_CITIES, is_static=True):
    """Get the best solution among N others from the list.

    Parameters:
        tours : List of the possible solutions.
        N : number of elements to search.
        weights : weight matrix.
        N_CITIES : length of the list.
        is_static : True if the fitness si static. False otherwise.

    Returns:
        cost : cost of the best local solution
        sol : best local solution

    """
    best_cost = float('Inf')
    best_sol = None

    for i in range(N):
        if is_static:
            cost = static_objfun(tours[i], weights, N_CITIES)
        else:
            cost = dynamic_objfun(tours[i], weights, N_CITIES)

        if cost < best_cost:
            best_cost = cost
            best_sol = tours[i]

    return best_cost, best_sol


def _ce(sol1, sol2, N_CITIES):
    """This function count the number of common edges between the two solutions.
    Parameters:
        sol1 : first solution.
        sol2 : second solution.
        N_CITIES : number of cities. Length of a solution.
    Returns:
        counter : counter of the common edges.
    """
    counter = 0
    for r in range(N_CITIES):
        s = (r + 1) % N_CITIES
        if sol1[r] == sol2[r] and sol1[s] == sol2[s]:
            counter += 1
    return counter


def diversity(population, POP_SIZE, N_CITIES):
    """Compute the diversity measure. 0.0 if all the solution in the population are equal.
    Parameters:
        population : list of solutions.
        POP_SIZE : length of the population.
        N_CITIES : number of cities. Length of a solution.
    Returns:
        diversity_value : returns the value of diversity of the population. 0.0 means that all the solutions in the
                        population are the same.
    """
    diversity_value = 0
    for i in range(POP_SIZE):
        for j in range(POP_SIZE):
            if i != j:
                c = (1 - _ce(population[i], population[j], N_CITIES) / N_CITIES)
                diversity_value += c
    return diversity_value / (POP_SIZE * (POP_SIZE - 1))


def inver_over(seq, p, population, weights, N_CITIES):
    """Inver over function.
    Parameters:
        seq : tour.
        p : probability of using the GI or BI inversion.
        population : population of the sequence.
        weights : weights matrix.
        N_CITIES : number of cities of the problem. Length of the seq.
    Returns:
        seq : return the (modified) sequence.
    """
    s_tmp = copy(seq)

    c = choice(s_tmp)

    if random() <= p: # Blind inversion.
        c_prime = c
        while c_prime == c:
            c_prime = choice(seq)
    else: # Guided inversion.
        s_2 = choice(population)
        c_prime = c
        while c_prime == c:
            c_prime = choice(s_2)

    reverse_path(s_tmp, (s_tmp.index(c) + 1) % N_CITIES, s_tmp.index(c_prime), N_CITIES)

    if dynamic_objfun(s_tmp, weights, N_CITIES) < dynamic_objfun(seq, weights, N_CITIES):
        seq = s_tmp
    return seq


def init_ant_position(N_ANTS, N_CITIES):
        """Position randomly the ants. Reinitialize the new_solutions array.
        Parameters:
            N_ANTS : number of ants.
            N_CITIES : number of cities.
        Returns:
            new_solutions : list of the new solutions.
            available : list of sets of available cities.
        """
        available = init_available(N_ANTS, N_CITIES)
        new_solutions = []
        k = 0
        for new in sample(range(N_CITIES), N_ANTS):
            available[k].remove(new)
            new_solutions.append([new])
            k = k + 1
        return new_solutions, available

def init_available(N_ANTS, N_CITIES):
        """Initialize the available list of sets. It creates a set for each ant.
        Achtung! It is a list of sets.
        Parameters:
            N_ANTS : number of ants.
            N_CITIES : number of cities.
        Returns:
            available : set of available cities.
        """
        available = []
        for k in range(N_ANTS):
            available.append(set(range(N_CITIES)))
        return available
