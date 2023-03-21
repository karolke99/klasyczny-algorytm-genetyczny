from individual import *
import numpy as np


class Population:

    def __init__(self, a=None, b=None, pop_size=None, fitness_function=None):
        self.a = a
        self.b = b
        self.decoded_population = None
        self.evaluated_population = None
        self.population = []

        self.fitness_function = fitness_function

        self.chromosome_size = fitness_function.get_chromosome_size()

        for i in range(pop_size):
            self.population.append(Individual(2, self.chromosome_size))

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
