# -*- coding: utf-8 -*-
""" inver_over.py
    Contains the class InverOver that apply the homonym algorithm to solve
    the classic TSP.
"""

__all__ = ['InverOver']
__version__ = '0.2'
__author__ = 'Emanuele Lovera'

from random import seed, random, choice, sample

import numpy as np

from utils.tsp_utils import get_weights_matrix, static_objfun, dynamic_objfun, reverse_path, nearest_neighbor, dynamic_nearest_neighbor, swap


class InverOver():
    def __init__(self, filename, weights=None, population=None, is_static=True, population_size=100, mutation_probability=0.05, timer_limit=20000):
        self.is_static = is_static
        # Matrices
        if weights is None:
            self.weights = get_weights_matrix(filename, static=self.is_static)
        else:
            self.weights = weights
            #self.weights = self.weights[:25, :25]

        # population
        if population is None:
            self.population = None
            self.POP_SIZE = population_size
        else:
            self.population = population
            self.POP_SIZE = len(self.population)
        self.population_cost = None
        self.compute_population_cost()

        # constants
        self.N_CITIES = np.shape(self.weights)[1]
        self.TIMER_LIMIT = timer_limit
        self.MUTATION_PROBABILITY = mutation_probability  # mutation probability

        # solutions:
        self.best_cost = float('Inf')
        self.best_solution = None
        self.nn_solution = []
        self.nn_cost = float('Inf')

    def run_io(self):
        """Main function of the class. Iterate over the inver over algorithm."""
        self.init_population()
        timer = 0
        while timer < self.TIMER_LIMIT:
            best_loc_cost, best_loc_sol = self.inver_over_algorithm()

            if best_loc_cost < self.best_cost:
                self.best_cost = best_loc_cost
                self.best_solution = best_loc_sol

            timer = timer + 1

        return self.best_cost, self.best_solution, self.nn_cost, self.nn_solution

    def inver_over_algorithm(self):
        """Main of the inver over algorithm."""
        # local best.
        best_loc_sol = None
        best_loc_cost = float('Inf')

        seed()
        for idx in range(self.POP_SIZE):
            individual_i = self.population[idx]
            individual_tmp = individual_i.copy()
            g = choice(individual_tmp) # select randomly a gene from the individual_tmp

            while True:
                # select the gene g_prime.
                g_prime = self.city_selection(g, individual_tmp)

                # if g is close to g_prime
                if individual_tmp[(individual_tmp.index(g) + 1) % self.N_CITIES] == g_prime or individual_tmp[(individual_tmp.index(g) - 1) % self.N_CITIES] == g_prime:
                    break
                else:
                    self.inversion(individual_tmp, g, g_prime)

                # check if the while has produced some better solution.
                if self.is_static:
                    cost_tmp = static_objfun(individual_tmp, self.weights, self.N_CITIES)
                else:
                    cost_tmp = dynamic_objfun(individual_tmp, self.weights, self.N_CITIES)

                if cost_tmp < static_objfun(individual_i, self.weights, self.N_CITIES):
                    self.population[idx] = individual_tmp
                    individual_i = individual_tmp
                    cost_i = cost_tmp
                    self.population_cost[idx] = cost_tmp

                # update local best
                if cost_i < best_loc_cost:
                    best_loc_sol = individual_i
                    best_loc_cost = cost_i

        return best_loc_cost, best_loc_sol

    def city_selection(self, g, individual_tmp):
        """Function that select the g_prime city."""
        # select g_prime
        if random() < self.MUTATION_PROBABILITY: # Mutation Case
            g_prime = g
            while g_prime == g:
                g_prime = choice(individual_tmp) # select randomly a gene from individual_tmp
        else:  # select randomly a gene from a random individual_j
            g_prime = g
            while g_prime == g:
                g_prime = choice(choice(self.population)) # individual_j = choice(self.population)
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

    def init_population(self):
        """Init randomly the population. THe size of the population is expressed by POP_SIZE and the range
        with n."""
        self.population = []
        for i in range(self.POP_SIZE):
            self.population.append(sample(range(self.N_CITIES), self.N_CITIES))
        self._compute_nn()

    def get_population(self):
        """Return the population list."""
        return self.population.copy()

    def set_population(self, population):
        """Set the population list."""
        self.population = population.copy()
        self._compute_nn()

    def _compute_nn(self):
        """Inner function: update the nn_solution."""
        if self.is_static:
            self.nn_solution = nearest_neighbor(self.weights)
            self.nn_cost = static_objfun(self.nn_solution, self.weights, self.N_CITIES)
        else:
            self.nn_solution = dynamic_nearest_neighbor(self.weights)
            self.nn_cost = dynamic_objfun(self.nn_solution, self.weights, self.N_CITIES)

    def compute_population_cost(self):
        self.population_cost = []
        if self.is_static:
            for individual in self.population:
                self.population_cost.append(static_objfun(individual, self.weights, self.N_CITIES))
            self.population_cost = np.array(self.population_cost)
        else:
            for individual in self.population:
                self.population_cost.append(dynamic_objfun(individual, self.weights, self.N_CITIES))
            self.population_cost = np.array(self.population_cost)

    # save the results.
    print("Saving...")
    savefilename = "parameter_hard_test.npz"
    np.savez(savefilename, pop_cost=cost_arr_popul, mutat_cost=cost_arr_mutat,
             timer_cost=cost_arr_timer, cost_matrix=cost_matrix)
    print("Finish!")
