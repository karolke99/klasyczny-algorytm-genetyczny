from fitness_function import *
from population import *

import settings
if __name__ == '__main__':

    fitness_fun = FitnessFunction(a=settings.A, b=settings.B, precision=settings.PRECISION)

    population = Population(a=settings.A, b=settings.B, pop_size=settings.POPULATION_SIZE, fitness_function=fitness_fun)

    print(population)

    print(population.evaluate())