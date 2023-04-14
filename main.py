from fitness_function import *
from population import *

import settings

# Percent podajemy w ułamku np. 50% = 0.5 tj. select_best(0.5) oznacza, że wybranych zostanie 50% osobników

if __name__ == '__main__':
    fitness_fun = FitnessFunction(a=settings.A, b=settings.B, precision=settings.PRECISION)

    population = Population(a=settings.A, b=settings.B, pop_size=settings.POPULATION_SIZE, fitness_function=fitness_fun)
    print(population.evaluate())

    print(" ")

    print("Populacja")

    for i in population.population:
        print(i)

    print(" ")

    print("Rozmiary populacji po selekcji")
    print(f"Najlepszych: {settings.POPULATION_SIZE}")
    selected_tournament = population.select_tournament(3)
    print(f"Turniejowa: {len(selected_tournament)}")
    selected_roulette = population.select_roulette(0.8)
    print(f"Ruletka: {len(selected_roulette)}")
    selected_best = population.select_best(0.5)
    print(f"Najlepszych: {len(selected_best)}")

    print(" ")

    print("Rozmiary populacji po krzyżowaniu")
    new_pop_single_cross = population.homogeneous_cross(0.8, selected_tournament)
    new_pop_double_cross = population.single_point_cross(0.8, selected_roulette)
    new_pop_triple_cross = population.triple_point_cross(0.8, selected_best)
    new_pop_homogeneous_cross = population.homogeneous_cross(0.8, selected_tournament)
    print(f'Single point: {len(new_pop_single_cross)}')
    print(f'Double point: {len(new_pop_double_cross)}')
    print(f'Triple point: {len(new_pop_triple_cross)}')
    print(f'Homogeneous: {len(new_pop_homogeneous_cross)}')

    print(" ")
    print("Rozmiary populacji po mutacjach")
    new_pop_boundary_mutation = population.boundary_mutation(0.2, new_pop_single_cross)
    new_pop_single_mutation = population.single_point_mutation(0.2, new_pop_homogeneous_cross)
    new_pop_double_mutation = population.double_point_mutation(0.2, new_pop_triple_cross)
    print(f'Boundary: {len(new_pop_boundary_mutation)}')
    print(f'Single point: {len(new_pop_single_mutation)}')
    print(f'Double point: {len(new_pop_double_mutation)}')

    print(" ")
    new_pop_inversion = population.inversion(0.3, new_pop_double_mutation)
    print(f"Rozmiar populacji po inwersji: {len(new_pop_inversion)}")





    #population = Population(a=settings.A, b=settings.B, pop_size=settings.POPULATION_SIZE, fitness_function=fitness_fun)
    #evaluated_pop = population.evaluate()

    # for i in range(settings.EPOCH_NUM):
    #
    #     selected_population = population.select_tournament(5)
    #     crossed_population = population.single_point_cross(0.8, selected_population)
    #     mutated_population = population.boundary_mutation(0.2, crossed_population)
    #     inverted_population = population.inversion(0.3, mutated_population)

    ##### Przed przejściem do kolejnej epoki należy podać jako value w konstruktorze Population nową populację
    ##### (populację po dokonaniu wszystkich operacji krzyżowania, mutacji, inwersji)

    #     population = Population(a=settings.A, b=settings.B, pop_size=settings.POPULATION_SIZE,
    #                          fitness_function=fitness_fun,
    #                          value=inverted_population)
    #
    #     print(population.evaluate())






