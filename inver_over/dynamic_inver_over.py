# -*- coding: utf-8 -*-
""" dynamic_inver_over.py
    Contains the class InverOver that apply the homonym algorithm to solve
    the classic TSP.
"""

__all__ = ['DynamicInverOver']
__version__ = '0.1'
__author__ = 'Emanuele Lovera'

import numpy as np

from random import random, sample, choice, seed

from scipy.sparse.csgraph import minimum_spanning_tree

from utils.tsp_utils import get_weights_matrix, init_population, dynamic_objfun, reverse_path, swap


class DynamicInverOver():
    def __init__(self, filename, population_size=100, mutation_probability=0.1, pool_probability=0.2,
                 timer_limit=20000):
        # Matrices
        self.weights = get_weights_matrix(filename, static=False)
        #self.weights = self.weights[:50, :50, :50]

        # constants
        self.N_CITIES = np.shape(self.weights)[1]
        self.TIMER_LIMIT = timer_limit
        self.MUTATION_PROBABILITY = mutation_probability
        self.POOL_PROBABILITY = pool_probability
        self.POP_SIZE = population_size
        self.SENSIBILITY = 0.1

        # population
        self.population = init_population(self.POP_SIZE, self.N_CITIES)
        self.population_cost = None # it is a numpy array.
        self.compute_population_cost()

        # gene_pool
        self.K = 6  # cardinality of the gene pool
        self.gene_pool = []
        self.init_gene_pool()

        # solutions:
        self.best_cost = float('Inf')
        self.best_solution = None

    def run_dio(self):
        """Main function of the class. Iterate over the inver over algorithm."""
        timer = 0
        while timer < self.TIMER_LIMIT:
            print(timer)
            best_loc_cost, best_loc_sol = self.inver_over_algorithm()
            self.dynamic_change()

            if best_loc_cost < self.best_cost:
                self.best_cost = best_loc_cost
                self.best_solution = best_loc_sol

            timer = timer + 1

        return self.best_cost, self.best_solution

    def inver_over_algorithm(self):
        """Main of the inver over algorithm."""
        # local best.
        timer = 0

        seed()
        for idx in range(self.POP_SIZE):
            individual_tmp = self.population[idx].copy()
            g = choice(individual_tmp)  # select randomly a gene from the individual_tmp

            length = len(individual_tmp)

            while timer < 100:
                # select the gene g_prime.
                g_prime = self.city_selection(g, individual_tmp)

                # if g is close to g_prime
                if individual_tmp[(individual_tmp.index(g) + 1) % length] == g_prime or individual_tmp[
                    (individual_tmp.index(g) - 1) % length] == g_prime:
                    break
                else:
                    self.inversion(individual_tmp, g, g_prime)

                # check if the while has produced some better solution.
                cost = dynamic_objfun(individual_tmp, self.weights, self.N_CITIES)
                if cost < self.population_cost[idx]:
                    self.population[idx] = individual_tmp
                    self.population_cost[idx] = cost
                    timer = 0
                else:
                    timer += 1

        best_loc_idx = np.argmin(self.population_cost)
        return self.population_cost[best_loc_idx], self.population[best_loc_idx]

    def city_selection(self, g, individual_tmp):
        """Function that select the g_prime city."""
        # select g_prime
        rnd = random()
        if rnd < self.MUTATION_PROBABILITY:  # Mutation Case
            g_prime = g
            while g_prime == g:
                g_prime = choice(individual_tmp)  # select randomly a gene from individual_tmp
        elif rnd < self.MUTATION_PROBABILITY + self.POOL_PROBABILITY:
            g_prime = g
            while g_prime == g:
                g_prime = self.get_from_gene_pool(g)
        else:  # select randomly a gene from a random individual_j
            g_prime = g
            while g_prime == g:
                g_prime = choice(choice(self.population))  # individual_j = choice(self.population)
        return g_prime

    def inversion(self, individual_tmp, g, g_prime):
        """Reverse the path between g and g_prime."""
        g_idx = individual_tmp.index(g)
        g_prime_idx = individual_tmp.index(g_prime)

        if g_idx < g_prime_idx:
            reverse_path(individual_tmp, g_idx + 1, g_prime_idx)
        else:
            reverse_path(individual_tmp, g_prime_idx, g_idx - 1)
        swap(individual_tmp, g, individual_tmp[g_prime_idx])

    def get_population(self):
        """Return the population list."""
        return self.population.copy()

    def set_population(self, population):
        """Set the population list."""
        self.population = population

    def get_from_gene_pool(self, g):
        """Get a gene from the gene pool. Choose randomly the gene prime among all
        contained inside the g gene pool."""
        return choice(self.gene_pool[g])

    def dynamic_change(self):
        """"Detect the moved cities and reset them inside each individual."""
        for individual in self.population:
            for c in self.find_moved_cities(individual):  # get the list of all moved cities
                print("Detected moved city c = {}".format(c))
                individual.remove(c)
                c1_idx = self.find_closest_cities(individual, c)
                individual.insert(c1_idx, c)

    def find_closest_cities(self, individual, c):
        """"Return the index of the nearest city."""
        t = individual.index(c)
        min_c = float('Inf')
        min_idx = -1

        length = len(individual)

        for idx in range(length):
            cost = self.weights[t, c, individual[idx % length]] + self.weights[
                t, c, individual[(idx + 1) % length]]
            if cost < min_c:
                min_c = cost
                min_idx = idx
        return min_idx

    def find_moved_cities(self, individual):
        """Find the cities that are moving more than the sensibility value."""
        moved_cities = []  # accumulator of all moved cities.

        t = 0  # time iterator.
        for idx in range(self.N_CITIES - 1):  # loop over all cities of the tour.
            c_i = individual[idx]  # first city.
            c_j = individual[idx + 1]  # second city.
            # compute the difference  between w[t] and w[t+1]
            diff = np.abs(self.weights[t, c_i, c_j] - self.weights[t + 1, c_i, c_j])
            limit = self.SENSIBILITY * self.weights[t, c_i, c_j]
            if diff > limit:
                moved_cities.append(c_i)
            t = t + 1
        return moved_cities

    def alpha_distances(self, t, c, c2, mst=None):
        """Compute the alpha distance at time t for city c and edge (c,c2)"""
        # delete row and column from the weights
        w_t = self.weights[t].copy()
        w_t = np.delete(w_t, c, 0)
        w_t = np.delete(w_t, c, 1)

        # compute the minimum spanning tree
        if mst is None:
            mst = minimum_spanning_tree(w_t)
        mst_graph = np.array(mst.toarray().astype(int))
        mst_graph = np.insert(mst_graph, c, np.zeros(self.N_CITIES - 1), axis=0)
        mst_graph = np.insert(mst_graph, c, np.zeros(self.N_CITIES), axis=1)

        # find the verticies with degree 1
        mask = mst_graph != 0
        verticies = []
        for i in range(np.shape(mst_graph)[0]):
            if np.count_nonzero(mst_graph[i]) == 1:
                verticies.append(i)

        # find the two closest cities
        closest_city1 = np.argmin(self.weights[t, c, verticies])
        verticies.remove(verticies[closest_city1])
        closest_city2 = np.argmin(self.weights[t, c, verticies])

        # link c with closest verticies.
        one_tree = mst_graph.copy()
        one_tree[c, closest_city1] = self.weights[t, c, closest_city1]
        one_tree[c, closest_city2] = self.weights[t, c, closest_city2]
        l_one_tree = np.sum(one_tree)

        one_tree_plus = mst_graph
        one_tree_plus[c, closest_city1] = self.weights[t, c, closest_city1]
        one_tree_plus[c, closest_city2] = self.weights[t, c, c2]
        l_one_tree_plus = np.sum(one_tree_plus)

        return l_one_tree, l_one_tree_plus

    def init_gene_pool(self):
        """"Create the gene pool of cardinality k."""
        for v_1 in range(self.N_CITIES):
            alpha_distances = []
            for v_2 in range(self.N_CITIES):
                l1, l2 = self.alpha_distances(0, v_1, v_2)
                alpha_distances.append(abs(l1 - l2))

            gp = []
            for i in range(self.K):
                gp.append(np.argmin(alpha_distances))
                alpha_distances.remove(alpha_distances[gp[-1]])
            self.gene_pool.append(gp)

    def check_valid_individual(self, individual):
        try:
            ind = individual.copy()
            for e in range(self.N_CITIES):
                ind.remove(e)
        except ValueError:
            print("Element e {} not present!".format(e))

    def compute_population_cost(self):
        self.population_cost = []
        for individual in self.population:
            self.population_cost.append(dynamic_objfun(individual, self.weights, self.N_CITIES))
        self.population_cost = np.array(self.population_cost)
