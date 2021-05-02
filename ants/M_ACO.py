# -*- coding: utf-8 -*-
""" dynamic_ant_colony_system
    Implementation of the population based ACO version.
"""

__all__ = ['DynamicAntColonySystem']
__version__ = '0.1'
__author__ = 'Emanuele Lovera'

from random import seed, random, sample, choice

import numpy as np

from utils.nearest_neighbors import dynamic_nearest_neighbor
from utils.tsp_utils import dynamic_objfun, get_weights_matrix, get_best_solution


class DynamicAntColonySystem:

    def __init__(self, filename, card_max=None, n_ants=5, alpha=1, beta=6, q0=0.9, pop_size=100, max_generation=1000):
        """Constructor of the class Dynamic Ant for the Population Based Ant Colony Optimization.
        Parameters:
            filename : string that contains the path of the weights file.
            card_max : (optional - default None) integer that represent the maximum number of cities to use.
            n_ants : (optional - default 5) integer. The number of the ants. .
            alpha : (optional - default 1) pheromone trail influence. The greater the more important is the pheromone
                    trail.
            beta : (optional - dafault 1) heuristic information influence. The greater the more important is the
                   heuristic.
            q0 : (optional - default 0.1) probability threshold of choosing the next city with the probabilistic rule.
            max_generation : (optional - default 1000) maximum number of the generations.
        """
        # Weights Matrix
        self.weights = get_weights_matrix(filename)
        if card_max is not None and card_max < np.shape(self.weights)[1] and card_max > 0:
            self.N_CITIES = card_max
            self.weights = self.weights[:card_max, :card_max, :card_max]
        else:
            self.N_CITIES = np.shape(self.weights)[1]

        # Constants
        self.BETA = beta
        self.N_ANTS = n_ants
        self.MAX_GENERATION = max_generation
        self.TAO_0 = 0  # minimum pheromone level. initialized later.
        self.ALPHA = alpha
        self.Q0 = q0  # to be global constant
        self.MAX_POP = pop_size

        # Precompute the heuristic information based on the weights.
        self.heuristic_info = np.power(np.divide(1, self.weights, where=self.weights != 0), self.BETA)

        # Time iterator for the dynamic application.
        self.t = 0  # current time
        self.t0 = 0  # initial time
        self.T = np.shape(self.weights)[0] - 1  # final time (index)

        # Solutions
        self.solutions = []
        self.new_solutions = None
        self.nn_solution = []
        self.nn_cost = float('Inf')

        # global solution
        self.global_best_solution = None
        self.global_best_cost = float('Inf')

        # indexes
        self.iteration = 0

        # pheromone
        self.TAO_0 = None  # minimum pheromone level.
        self.TAO_MAX = None  # maximum pheromone level.
        self.TAO_DELTA = None  # delta step for the pheromone.
        self.pheromone_matrix = None
        self.init_pheromone_matrix()

        # list of length k of sets of available cities
        self.available = None

        # population: managed with the age mechanism.
        self.population = []
        self.pop_iter = 0
        self.pop_iter_age = 0

        self.overflow = 0

    def run_ants(self):
        """Main function. Run the ants."""
        while self.iteration < self.MAX_GENERATION:
            seed()  # new seed for each new iteration.
            # (re)init new solutions.
            self.t = 0
            self.init_available()
            self.init_ant_position()

            for n in range(1, self.N_CITIES):
                self.state_transition_rule()
                self.t = self.t + 1

            # add the best in solution.
            self.include_in_population()

            # scope of the next group of solutions.
            for new_sol in self.new_solutions:
                self.solutions.append(new_sol)
            self.iteration = self.iteration + 1

        if self.overflow != 0:
            print("Counted {} overflows.".format(self.overflow))
        return self.global_best_cost, self.global_best_solution

    def state_transition_rule(self):
        """For each ant k, choose the next city."""
        for k in range(self.N_ANTS):
            # get the current city.
            r = self.new_solutions[k][-1]

            # pick a random number.
            rnd = random()
            if rnd < self.Q0:
                next_city = self.random_proportional_rule(r, self.available[k])
            else:
                next_city = self.pseudo_random_proportional_rule(r, self.available[k])

            self.available[k].remove(next_city)  # remove the city just visited
            self.new_solutions[k].append(next_city)  # append the new city.

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
        self.nn_solution = dynamic_nearest_neighbor(self.weights)
        self.nn_cost = dynamic_objfun(self.nn_solution, self.weights, self.N_CITIES)

        self.TAO_0 = 1 / (self.nn_cost * self.N_CITIES)
        self.TAO_MAX = 100
        self.TAO_DELTA = abs(self.TAO_MAX - self.TAO_0) / self.MAX_POP
        self.pheromone_matrix = np.full((self.N_CITIES, self.N_CITIES), self.TAO_0)

    def random_proportional_rule(self, r, available):
        """ Optimized version of the random_proportional_rule. Make the probabilistic choice of the next city."""
        available = list(available)

        product = self.pheromone_matrix[r, available] * self.heuristic_info[self.t, r, available]
        summa = np.sum(product)

        if summa == 0:
            return choice(available)
        else:
            p = product / summa
            s = np.sum(p)

            try:
                ret_val = np.random.choice(available, p=p)
            except ValueError:
                p = abs(p)
                s = np.sum(p)
                ret_val = np.random.choice(available, p=p)
                self.overflow += 1

            return ret_val

    def pseudo_random_proportional_rule(self, r, available):
        """Optimized version of pseudo_random_proportional_rule. Choose the next city which has the maximum probability.
        """
        available = list(available)
        p = np.power(self.pheromone_matrix[r, available], self.ALPHA) * self.heuristic_info[self.t, r, available]
        return available[np.argmax(p)]

    def include_in_population(self):
        """Include the new best local solution inside the population list."""
        # Find the best among the new solutions.
        best_loc_cost, best_loc_sol = get_best_solution(self.new_solutions, self.N_ANTS, self.weights, self.N_CITIES,
                                                        is_static=False)

        # compute the global best.
        if best_loc_cost < self.global_best_cost:
            self.global_best_cost = best_loc_cost
            self.global_best_solution = best_loc_sol

        # append or include the best solution for this generation.
        if self.pop_iter < self.MAX_POP:  # If there is enough space, simply append it.
            self.population.append(best_loc_sol)
            self.pop_iter = self.pop_iter + 1  # increment the iterator
        else:
            self.helper_insert_aging(best_loc_sol)

    def helper_insert_aging(self, sol):
        """Insert a new solution in population using the aging technique."""

        # remove the oldest solution and apply the evaporation.
        removed = self.population.pop(self.pop_iter_age)
        self.pheromone_modification(removed, removing=True)

        self.population.insert(self.pop_iter_age, sol)
        self.pheromone_modification(sol, removing=False)

        self.pop_iter_age = (self.pop_iter_age + 1) % self.MAX_POP  # increase the circular array index.

    def pheromone_modification(self, sol, removing=True):
        """Evaporate or deposit the pheromone matrix of the removed solution."""
        for i in range(self.N_CITIES):
            r = sol[i]
            s = sol[(i + 1) % self.N_CITIES]

            if removing:
                # if removing is true than decrease the pheromone value.
                self.pheromone_matrix[r, s] = self.pheromone_matrix[s, r] = self.pheromone_matrix[r, s] - self.TAO_DELTA
                if self.pheromone_matrix[r, s] < self.TAO_0:
                    self.pheromone_matrix[r, s] = self.pheromone_matrix[s, r] = self.TAO_0
            else:
                # Otherwise increase of delta.
                self.pheromone_matrix[r, s] = self.pheromone_matrix[s, r] = self.pheromone_matrix[r, s] + self.TAO_DELTA
