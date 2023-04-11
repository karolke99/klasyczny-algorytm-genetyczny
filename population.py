import random

from individual import *
import numpy as np


class Population:

    def __init__(self, a=None, b=None, pop_size=None, fitness_function=None, value=None):
        self.a = a
        self.b = b
        self.decoded_population = None
        self.evaluated_population = None


        self.fitness_function = fitness_function
        self.chromosome_size = fitness_function.get_chromosome_size()

        if value is None:
            self.population = []
            for i in range(pop_size):
                self.population.append(Individual(2, self.chromosome_size))

            self.population = np.array(self.population)

        else:
            self.population = value

        self.size = len(self.population)

    def __str__(self):
        population_string = f'Population size: {self.size} \n'

        for individual in self.population:
            population_string += str(individual) + '\n\n'

        return population_string

    def evaluate(self):

        self.decoded_population = []

        for individual in self.population:
            decoded_individual = individual.decode(self.a, self.b, self.chromosome_size)
            self.decoded_population.append(decoded_individual)

        self.evaluated_population = np.array([self.fitness_function.compute(individual)
                                              for individual in self.decoded_population])

        return self.evaluated_population

    def select_best(self, percent):
        sorted_indices = np.argsort(self.evaluated_population)[::-1]
        selected_indices = sorted_indices[:int(self.size * percent)]
        return self.population[selected_indices]

    def select_roulette(self, percent):
        temp_population = np.array([1 / i for i in self.evaluated_population])
        probabilities = np.array([(i / sum(temp_population)) for i in temp_population])
        distribution = probabilities.cumsum()
        new_pop = np.array([])

        for i in range(int(self.size * percent)):
            spin = np.random.uniform(0., 1.)
            for index in range(0, len(self.evaluated_population)):
                if spin < distribution[index]:
                    new_pop = np.append(new_pop, self.population[index])
                    break

        return new_pop

    def select_tournament(self, tournament_size):
        selected_indices = np.array([], dtype=np.int64)
        permuted_indices = np.random.permutation(self.size)
        elements_in_tournament = int(self.size / tournament_size)

        for i in range(0, self.size, elements_in_tournament):
            tournament_indices = permuted_indices[i:i + elements_in_tournament]

            if len(tournament_indices) != elements_in_tournament:
                break

            tournament = self.evaluated_population[tournament_indices]
            best = np.argmin(tournament)
            selected_indices = np.append(selected_indices, tournament_indices[best])

        return self.population[selected_indices]

    def single_point_cross(self, probability, selected_population):

        new_pop = np.array([])
        while new_pop.size < self.size:
            value = np.random.uniform(0., 1.)

            if value < probability:
                index_of_crossing = int(np.random.randint(self.chromosome_size * 2))
                elements_to_cross = np.random.choice(selected_population, size=2, replace=False)

                new_element1 = Individual(2, self.chromosome_size,
                                          np.append(elements_to_cross[0].get_individual()[:index_of_crossing],
                                                    elements_to_cross[1].get_individual()[index_of_crossing:]))
                new_element2 = Individual(2, self.chromosome_size,
                                          np.append(elements_to_cross[1].get_individual()[:index_of_crossing],
                                                    elements_to_cross[0].get_individual()[index_of_crossing:]))

                new_pop = np.append(new_pop, new_element1)
                new_pop = np.append(new_pop, new_element2)

        return new_pop

    def double_point_cross(self, probability, selected_population):
        new_pop = np.array([])
        while new_pop.size < self.size:
            value = np.random.uniform(0., 1.)

            if value < probability:
                ind_of_crossing = np.sort(np.random.choice(self.chromosome_size * 2, size=2, replace=False))

                elements_to_cross = np.random.choice(selected_population, size=2, replace=False)

                new_element1 = Individual(2, self.chromosome_size, np.concatenate((
                    elements_to_cross[0].get_individual()[:ind_of_crossing[0]],
                    elements_to_cross[1].get_individual()[ind_of_crossing[0]:ind_of_crossing[1]],
                    elements_to_cross[0].get_individual()[ind_of_crossing[1]:]
                )))

                new_element2 = Individual(2, self.chromosome_size, np.concatenate((
                    elements_to_cross[1].get_individual()[:ind_of_crossing[0]],
                    elements_to_cross[0].get_individual()[ind_of_crossing[0]:ind_of_crossing[1]],
                    elements_to_cross[1].get_individual()[ind_of_crossing[1]:]
                )))

                new_pop = np.append(new_pop, new_element1)
                new_pop = np.append(new_pop, new_element2)

        return new_pop

