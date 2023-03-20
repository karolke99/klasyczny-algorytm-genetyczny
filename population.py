from individual import *
import numpy as np


class Population:

    def __init__(self, chromosome_size=None, pop_size=None, variables_number=2):
        self.decoded_population = None
        self.population = []
        population_array = np.random.choice([0, 1], size=(pop_size, variables_number, chromosome_size))
        # print(population_array)
        # print("--------------------------")
        for genome in population_array:
            individual = Individual(genome)
            self.population.append(individual)

        self.size = len(self.population)
        self.chromosome_size = chromosome_size

    def decode(self):

        self.decoded_population = []

        for individual in self.population:
            # print(individual)
            decoded_individual = individual.decode(-10, 10, self.chromosome_size)
            self.decoded_population.append(decoded_individual)

        # print(self.decoded_population)

        return np.array(self.decoded_population)
