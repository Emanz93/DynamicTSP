# -*- coding: utf-8 -*-
""" dioea.py
    Contains the class InverOver that apply the homonym algorithm to solve
    the dynamic TSP.
"""

__all__ = ['Dioea']
__version__ = '0.2'
__author__ = 'Emanuele Lovera'

import numpy as np

from inver_over.inver_over import InverOver

from utils.tsp_utils import dynamic_objfun


class Dioea():
    def __init__(self, filename, population_size=100, timer_limit=20000):
        self.weights = get_weights_matrix(filename, static=False)

        self.best_solution = None
        self.best_cost = float('Inf')

        self.POP_SIZE = population_size
        self.population = None
        self.init_population()
        print("BEST INITIAL COST = {}".format(self.best_cost))
        print("BEST INITIAL SOLUTION = {}".format(self.best_solution))

        self.inver_over = None
        self.d_list = None


    def run_dioea():
        """Main function for the dioea algorithm."""
        self.get_d_list()
        while timer < self.TIMER_LIMIT: # I'm not sure of the blocking condition.
            self.inver_over = InverOver(None, weights=self.weights, population=self.population, is_static=False)
            self.inver_over.inver_over_algorithm()
            self.population = self.inver_over.get_population()

            idx_d_list = 0 # index of d_list
            timer = 0
            while idx_d_list < len(self.d_list[0]):
                city = (self.d_list[0, idx_d_list], self.d_list[1, idx_d_list])
                self.change(city)
                idx_d_list = idx_d_list + 1
            self.t = self.t + 1

    def change(self, city):
        for element in self.population:
            #delete(element, city)
            index_c = element.index(city)


            #insert(element, city)



    def get_d_list(self):
        """Compute the d_list from the current population."""
        try:
            diff_matrix = abs(self.weights[self.t, :, :] - self.weights[self.t + 1, :, :])
            limit = self.matrix[self.t, :, :] * self.SENSIBILITY
            mask = diff_matrix < limit
            (x, y) = np.nonzero(mask) # indicies of all the cities that have moved over time.
            self.d_list = (x, y)
        except IndexError:
            print("IndexError. self.t={}".format(self.t))

    def init_population(self):
        """Init randomly the population. THe size of the population is expressed by POP_SIZE and the range
        with n."""
        self.population = []
        for i in range(self.POP_SIZE):
            self.population.append(random.sample(
                range(self.N_CITIES), self.N_CITIES))
            cost = dynamic_objfun(self.weights, self.population[-1])
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = self.population[-1]

def delete(element, city):
    pass

def insert(element, city):
    pass
