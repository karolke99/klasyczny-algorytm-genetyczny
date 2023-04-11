from fitness_function import *
from population import *

import settings

if __name__ == '__main__':
    fitness_fun = FitnessFunction(a=settings.A, b=settings.B, precision=settings.PRECISION)

    individual = Individual(2, fitness_fun.chromosome_size)

    population = Population(a=settings.A, b=settings.B, pop_size=settings.POPULATION_SIZE, fitness_function=fitness_fun)
    print(population.evaluate())

    selected_tournament = population.select_tournament(3)

    population.double_point_cross(0.8, selected_tournament)


    # for i in range(settings.EPOCH_NUM):
    #
    #     selected_tournament = population.select_tournament(15)
    #     population = population.single_point_cross(0.9, selected_tournament)
    #
    #     population = Population(a=settings.A, b=settings.B, pop_size=settings.POPULATION_SIZE,
    #                          fitness_function=fitness_fun,
    #                          value=population)
    #
    #     print(population.evaluate())

    # selected_best = population.select_best(0.5)
    # selected_roulette = population.select_roulette(0.5)
    # selected_tournament = population.select_tournament(3)





