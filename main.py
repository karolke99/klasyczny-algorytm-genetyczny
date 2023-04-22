from fitness_function import *
from population import *
import settings

# Percent podajemy w ułamku np. 50% = 0.5 tj. select_best(0.5) oznacza, że wybranych zostanie 50% osobników

if __name__ == '__main__':
    fitness_fun = FitnessFunction(a=settings.A, b=settings.B)

    population = Population(a=settings.A, b=settings.B, pop_size=10, fitness_function=fitness_fun , value=None)

    print(population)

    evaluated_pop = population.evaluate_real()
    print(evaluated_pop)

    selected_best = population.select_best(0.5)

    selected_roulette = population.select_roulette()

    # selected_tournament = population.select_tournament(2)

    print(" ")

    new_pop = population.blend_cross_alpha_beta(1, selected_roulette, 0.25, 0.25, "%", 0)

    new_pop = population.gauss_mutation(1, new_pop)

    for i in new_pop:
        print(i)






