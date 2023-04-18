from fitness_function import *
from population import *
from real_individual import *
import settings

# Percent podajemy w ułamku np. 50% = 0.5 tj. select_best(0.5) oznacza, że wybranych zostanie 50% osobników

if __name__ == '__main__':
    fitness_fun = FitnessFunction(a=settings.A, b=settings.B)

    population = Population(a=settings.A, b=settings.B, pop_size=10, fitness_function=fitness_fun , value=None)

    print(population)

    evaluated_pop = population.evaluate_real()
    print(evaluated_pop)

    selected_best = population.select_best(0.5)
    for i in selected_best:
        print(i.chromosomes)

    print(" ")

    selected_roulette = population.select_roulette()
    for i in selected_roulette:
        print(i.chromosomes)

    print(" ")
    selected_tournament = population.select_tournament(2)
    for i in selected_tournament:
        print(i.chromosomes)





