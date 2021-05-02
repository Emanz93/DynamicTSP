# -*- coding: utf-8 -*-
""" ant_colony_system
    Contains the class AntColonySystem that apply the homonym algorithm to solve
    the classic TSP.
"""

__all__ = ['AntColonySystem']
__version__ = '0.2'
__author__ = 'Emanuele Lovera'


import numpy as np
from random import random, sample, choice

from utils.tsp_utils import static_objfun, get_weights_matrix, opt_3
from utils.nearest_neighbors import nearest_neighbor




class AntColonySystem:

    def __init__(self, filename, n_ants=10, alpha=0.1, beta=6, rho=0.1, q0=0.98, max_iteration=100):
        # Matrices
        self.weights = get_weights_matrix(filename, static=True)
        #self.weights = self.weights[:20, :20]

        # Constants
        self.BETA = beta
        self.N_CITIES = np.shape(self.weights)[1]
        self.N_ANTS = n_ants
        self.MAX_ITERATION = max_iteration
        self.TAO_0 = 0
        self.RHO = rho
        self.ALPHA = alpha
        self.Q0 = q0

        # Solutions
        self.solutions = []
        self.new_solutions = None

        # global solution
        self.global_best_solution = None
        self.global_best_cost = float('Inf')

        # indexes
        self.iteration = 0
        # self.start_idx = 0

        # pheromone
        self.pheromone_matrix = None
        self.init_pheromone_matrix()

    def run_ants(self):
        """Main function. Run the ants."""
        while self.iteration < self.MAX_ITERATION:
            #seed()  # new seed for each new iteration.
            # (re)init new solutions.
            self.init_available()
            self.init_ant_position()

            for n in range(1, self.N_CITIES):

                self.state_transition_rule()

                self.local_pheromone_update()

            self.local_search()

            self.global_pheromone_update()

            # scope of the next group of solutions.
            for new_sol in self.new_solutions:
                self.solutions.append(new_sol)
            self.iteration = self.iteration + 1

        return self.global_best_cost, self.global_best_solution

    def state_transition_rule(self):
        """Make a probabilistic step.
            tour: array containing all cities visited from the ant.
            weights: weights matrix.
            pheromone_matrix: matrix containing all the pheromone.

           Returns:
            tour : array containing the updated tour for the ant.
        """
        for k in range(self.N_ANTS):
            # get the current city.
            r = self.new_solutions[k][-1]

            rnd = random()
            if rnd > self.Q0:
                next_city = self.random_proportional_rule2(r, self.available[k])
            else:  # if q <= Q0
                next_city = self.pseudo_random_proportional_rule2(r, self.available[k])

            self.available[k].remove(next_city)  # remove the city just visited
            self.new_solutions[k].append(next_city) # append the new city.

    def init_ant_position(self):
        """Position randomly the ants. Reinitialize the new_solutions array."""
        self.new_solutions = []
        k = 0
        for new in sample(range(self.N_CITIES), self.N_ANTS):
            self.available[k].remove(new)
            self.new_solutions.append([new])
            k = k + 1

    def init_available(self):
        """Initialize the available list of sets. It creates a set for each ant.
        Achtung! It is a list of sets.
        """
        self.available = []
        for k in range(self.N_ANTS):
            self.available.append(set(range(self.N_CITIES)))

    def init_pheromone_matrix(self):
        """Init the pheromone matrix with nearest neighbor algorithm."""
        self.nn_solution = nearest_neighbor(self.weights)
        self.nn_cost = static_objfun(self.nn_solution, self.weights, self.N_CITIES)

        self.TAO_0 = 1 / (self.nn_cost * self.N_CITIES)
        self.pheromone_matrix = np.full((self.N_CITIES, self.N_CITIES), self.TAO_0)

    def random_proportional_rule2(self, r, available):
        """ Optimized version of the random_proportional_rule. For each available
         city returns the best city according to the random probability rule."""
        available = np.array(list(available))

        product = self.pheromone_matrix[r, available] * np.power((1 / self.weights[r, available]), self.BETA)
        summa = np.sum(product)

        if summa == 0:
            return choice(available)
        else:
            p = product / summa
            ret_val = np.random.choice(available, p=p)
            return ret_val

    def pseudo_random_proportional_rule2(self, r, available):
        """Optimized version of pseudo_random_proportional_rule2. For each available
         city returns the best city according to the pseudo random probability rule."""
        available = list(available)
        p = self.pheromone_matrix[r, available] * np.power((1 / self.weights[r, available]), self.BETA)
        return available[np.argmax(p)]

    def local_pheromone_update(self):
        for k in range(self.N_ANTS):
            r = self.new_solutions[k][-2]
            s = self.new_solutions[k][-1]
            self.pheromone_matrix[r, s] = self.pheromone_matrix[s, r] = (1 - self.RHO) * self.pheromone_matrix[
                r, s] + self.RHO * self.TAO_0

    def global_pheromone_update(self):
        """Global pheromone update."""
        best_new_sol = self.new_solutions[0]
        best_new_cost = static_objfun(best_new_sol, self.weights, self.N_CITIES)
        for sol in self.new_solutions:
            cost = static_objfun(sol, self.weights, self.N_CITIES)
            # update local best for global pheromone update.
            if best_new_cost < cost:
                best_new_cost = cost
                best_new_sol = sol

            # update global best
            if cost < self.global_best_cost:
                self.global_best_cost = cost
                self.global_best_solution = sol

        # to do: FIX GLOBAL PHEROMONE UPDATE
        for sol in self.new_solutions:
            for idx in range(self.N_CITIES):
                r = sol[idx]
                s = sol[(idx + 1) % self.N_CITIES]
                self.pheromone_matrix[r, s] = self.pheromone_matrix[s, r] = (1 - self.ALPHA) * self.pheromone_matrix[r, s]

        for idx in range(self.N_CITIES):
            r = best_new_sol[idx]
            s = best_new_sol[(idx + 1) % self.N_CITIES]
            self.pheromone_matrix[r, s] = self.pheromone_matrix[s, r] = self.pheromone_matrix[
                                                                                r, s] + self.ALPHA * 1 / best_new_cost

    def local_search(self):
        """Perform the local search with the 3-opt heuristic."""
        for k in range(self.N_ANTS):
            opt_cost, opt = opt_3(self.new_solutions[k], self.weights, self.N_CITIES)
            cost = static_objfun(self.new_solutions[k], self.weights, self.N_CITIES)
            if opt_cost < cost:
                self.new_solutions[k] = opt