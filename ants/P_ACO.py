# -*- coding: utf-8 -*-
""" dynamic_ant_colony_system
    Implementation of the population based ACO version.
"""

__all__ = ['DynamicAntColonySystem']
__version__ = '0.2'
__author__ = 'Emanuele Lovera'

from random import seed, random, sample, choice

import numpy as np
from time import time

from utils.nearest_neighbors import dynamic_nearest_neighbor
from utils.tsp_utils import dynamic_objfun, get_weights_matrix, get_fengyun_weights, init_ant_position, inver_over, opt_3, diversity


class DynamicAntColonySystem:

    def __init__(self, filename, card_max=None, n_ants=5, alpha=1, beta=6, q0=0.9, pop_size=100, max_generations=1000, is_fengyun=False):
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
            max_generations : (optional - default 1000) maximum number of the generations.
            is_fengyun : (optional - default False) If set to true change the ACO behaviour in order to manage its huge dataset.

        """
        # Weights Matrix
        if is_fengyun:
            self.weights, self.N_CITIES = get_fengyun_weights(filename)
        else:
            self.weights, self.N_CITIES = get_weights_matrix(filename, card_max=card_max, is_static=False)

        # Constants
        self.BETA = beta
        self.N_ANTS = n_ants
        self.MAX_GENERATIONS = max_generations
        self.TAO_0 = 0  # minimum pheromone level. initialized later.
        self.ALPHA = alpha
        self.Q0 = q0  # to be global constant
        self.MAX_POP = pop_size
        self.is_fengyun = is_fengyun


        # Precompute the heuristic information based on the weights.
        if not self.is_fengyun:
            self.heuristic_info = np.power(np.divide(1, self.weights, where=self.weights != 0), self.BETA)

        # Time iterator for the dynamic application.
        self.t = 0  # current time
        self.t0 = 0  # initial time
        self.T = np.shape(self.weights)[0] - 1  # final time (index)

        # Solutions
        self.solutions = []
        self.new_solutions = None  # new solutions.
        self.new_costs = None  # costs of the new solutions.
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

        # population: managed with the age mechanism.
        self.population = []
        self.pop_iter = 0
        self.pop_iter_age = 0

        # list of length k of sets of available cities
        self.available = None

        # aio local search
        self.IO_MAX_ITER = 10
        self.p_bi = 0.5
        self.p_gi = 0.5

        # statistic variables
        self.best_performance = np.zeros(self.MAX_GENERATIONS)
        self.avg_performance = np.zeros(self.MAX_GENERATIONS)
        self.worst_performance = np.zeros(self.MAX_GENERATIONS)
        self.change_global_min = np.zeros(self.MAX_GENERATIONS)
        self.global_p_bi = np.zeros(self.MAX_GENERATIONS)
        self.global_p_gi = np.zeros(self.MAX_GENERATIONS)
        self.global_n_bi_inversions = np.zeros(self.MAX_GENERATIONS)
        self.global_n_gi_inversions = np.zeros(self.MAX_GENERATIONS)
        self.relative_diversity = np.zeros(self.MAX_GENERATIONS)
        self.avg_time_per_solution = np.zeros(self.MAX_GENERATIONS)
        self.ls_number_of_improvements = np.zeros((self.N_ANTS, self.MAX_GENERATIONS))
        self.ls_rate_of_improvements = np.zeros((self.N_ANTS, self.MAX_GENERATIONS))

    def run_ants(self):
        """Main function. Run the ants."""
        while self.iteration < self.MAX_GENERATIONS:
            print(self.iteration)
            seed()  # new seed for each new iteration.
            # (re)init new solutions.
            self.new_solutions, self.available = init_ant_position(self.N_ANTS, self.N_CITIES)
            self.times = np.zeros(self.N_ANTS)

            self.new_costs = np.zeros(self.N_ANTS)
            for k in range(self.N_ANTS):
                t1 = time()
                self.state_transition_rule(k)
                self.times[k] = time() - t1

                self.new_costs[k] = dynamic_objfun(self.new_solutions[k], self.weights, self.N_CITIES)

            idx_loc_min = np.argmin(self.new_costs)
            # Perform the local search on the best new solution.
            #self.adaptive_inver_over(idx_loc_min)
            #self.local_search(idx_loc_min)

            # Pheromone update
            # add the best in solution.
            self.include_in_population()

            # scope of the next group of solutions.
            for new_sol in self.new_solutions:
                self.solutions.append(new_sol)

            # Update the new global max
            if self.new_costs[idx_loc_min] < self.global_best_cost:
                self.global_best_cost = self.new_costs[idx_loc_min]
                self.global_best_solution = self.new_solutions[idx_loc_min]

            # update the statistics
            self.update_statistics()

            self.iteration = self.iteration + 1

        return self.global_best_cost, self.global_best_solution

    def update_statistics(self):
        # compute the costs for all the solutions.
        self.best_performance[self.iteration] = np.min(self.new_costs)
        self.worst_performance[self.iteration] = np.max(self.new_costs)
        self.avg_performance[self.iteration] = np.average(self.new_costs)
        self.change_global_min[self.iteration] = self.global_best_cost
        self.relative_diversity[self.iteration] = diversity(self.new_solutions, self.N_ANTS, self.N_CITIES)
        self.avg_time_per_solution[self.iteration] = np.average(self.times)

    def get_performance_arr(self):
        return self.best_performance, self.avg_performance, self.worst_performance

    def get_change_global_min(self):
        return self.change_global_min

    def get_bi_gi(self):
        return self.global_p_bi, self.global_p_gi, self.global_n_bi_inversions, self.global_n_gi_inversions

    def get_diversity(self):
        return self.relative_diversity

    def get_avg_time(self):
        return self.avg_time_per_solution

    def get_ls_improvements(self):
        return self.ls_number_of_improvements, self.ls_rate_of_improvements

    def state_transition_rule(self, k):
        """For each ant k, choose the next city."""
        self.t = 0
        for i in range(1, self.N_CITIES):
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
            self.t = self.t + 1

    def init_pheromone_matrix(self):
        """Init the pheromone matrix with nearest neighbor algorithm."""
        self.nn_solution = dynamic_nearest_neighbor(self.weights, self.N_CITIES)
        self.nn_cost = dynamic_objfun(self.nn_solution, self.weights, self.N_CITIES)

        self.TAO_0 = 1 / (self.nn_cost * self.N_CITIES)
        self.TAO_MAX = 100
        self.TAO_DELTA = abs(self.TAO_MAX - self.TAO_0) / self.MAX_POP
        self.pheromone_matrix = np.full((self.N_CITIES, self.N_CITIES), self.TAO_0)

    def random_proportional_rule(self, r, available):
        """ Optimized version of the random_proportional_rule. Make the probabilistic choice of the next city."""
        available = list(available)

        if not self.is_fengyun:
            product = self.pheromone_matrix[r, available] * self.heuristic_info[self.t, r, available]
        else:
            heuristic_info = np.power(np.divide(1, self.weights[self.t], where=self.weights[self.t] != 0), self.BETA)
            product = self.pheromone_matrix[r, available] * heuristic_info[r, available]
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
        if not self.is_fengyun:
            p = np.power(self.pheromone_matrix[r, available], self.ALPHA) * self.heuristic_info[self.t, r, available]
        else:
            heuristic_info = np.power(np.divide(1, self.weights[self.t], where=self.weights[self.t] != 0), self.BETA)
            p = np.power(self.pheromone_matrix[r, available], self.ALPHA) * heuristic_info[r, available]
        return available[np.argmax(p)]

    def include_in_population(self):
        """Include the new best local solution inside the population list."""
        # Find the best among the new solutions.
        idx_loc_best = np.argmin(self.new_costs)
        best_loc_cost, best_loc_sol = self.new_costs[idx_loc_best], self.new_solutions[idx_loc_best]

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

    def adaptive_inver_over(self, idx=None):
        """Perform the local search with the Adaptive Inver-Over."""
        if idx is not None:
            self.helper_aio(idx)
        else:
            for k in range(self.N_ANTS):
                self.helper_aio(k)

    def helper_aio(self, k):
        original_cost = self.new_costs[k]
        epsilon_bi = epsilon_gi = 0

        for io_iteration in range(self.IO_MAX_ITER):
            c_old_best = self.new_costs[k]
            if random() < self.p_bi:
                new_best = inver_over(self.new_solutions[k], 1.0, self.new_solutions, self.weights, self.N_CITIES)
                c_new_best = dynamic_objfun(new_best, self.weights, self.N_CITIES)
                epsilon_bi = abs(c_new_best - c_old_best) / c_old_best
                self.global_n_bi_inversions[self.iteration] += 1  # Increment the statistics.
            else:
                new_best = inver_over(self.new_solutions[k], 0.0, self.new_solutions, self.weights, self.N_CITIES)
                c_new_best = dynamic_objfun(new_best, self.weights, self.N_CITIES)
                epsilon_gi = abs(c_new_best - c_old_best) / c_old_best
                self.global_n_gi_inversions[self.iteration] += 1  # Increment the statistics.

            # May I have update the sol?
            if c_new_best < c_old_best:
                self.new_solutions[k] = new_best
                self.new_costs[k] = c_new_best

        # update the statistics.
        if self.new_costs[k] < original_cost:
            self.ls_number_of_improvements[k, self.iteration] += 1  # increment the number of improvements.
            self.ls_rate_of_improvements[k, self.iteration] = abs(original_cost - self.new_costs[k])

        # Recompute the p_bi and p_gi
        self.p_bi = self.global_p_bi[self.iteration] = (self.p_bi + epsilon_bi) / \
                                                       (self.p_bi + epsilon_bi + self.p_gi + epsilon_gi)
        self.p_gi = self.global_p_gi[self.iteration] = 1 - self.p_bi

    def helper_local_search(self, k):
        opt_cost, opt_sol = opt_3(self.new_solutions[k], self.weights, self.N_CITIES, is_static=False)

        # if the new cost is better, than replace the old solution with the new one.
        if self.new_solutions[k] != opt_sol and opt_cost < self.new_costs[k]:
            self.ls_number_of_improvements[k, self.iteration] += 1  # increment the number of improvements.
            self.ls_rate_of_improvements[k, self.iteration] = abs(opt_cost - self.new_costs[k])
            self.new_solutions[k] = opt_sol
            self.new_costs[k] = opt_cost

    def local_search(self, idx=None):
        """Perform the local search with the 3-opt heuristic.
        Parameters:
            idx : if it is not None, than perform the local search only on the selected solution with index idx.
        """
        if idx is not None:
            self.helper_local_search(idx)
        else:
            for k in range(self.N_ANTS):
                self.helper_local_search(k)


def are_sol_equal(sol1, sol2):
    for i in range(len(sol1)):
        if sol1[i] != sol2[i]:
            return False
    return True
