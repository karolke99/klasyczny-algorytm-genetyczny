from individual import *
from real_individual import *
import numpy as np

ELITE_STRATEGY = True


class Population:

    def __init__(self, a=None, b=None, pop_size=None, fitness_function=None, value=None):
        self.a = a
        self.b = b
        self.decoded_population = None
        self.evaluated_population = None

        self.fitness_function = fitness_function

        if value is None:
            self.population = []
            for i in range(pop_size):
                self.population.append(RealIndividual(2, self.a, self.b, None))

            self.population = np.array(self.population)

        else:
            self.population = value

        self.size = len(self.population)

    def __str__(self):
        population_string = f'Population size: {self.size} \n'

        for individual in self.population:
            population_string += str(individual.chromosomes) + '\n'

        return population_string

    def evaluate_real(self):
        self.evaluated_population = np.array([self.fitness_function.compute(individual.chromosomes)
                                              for individual in self.population])

        return self.evaluated_population

    def select_best(self, percent):
        sorted_indices = np.argsort(self.evaluated_population)
        selected_indices = sorted_indices[:int(self.size * percent)]
        return self.population[selected_indices]

    def select_roulette(self, percent=1):
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

    def arithmetic_cross(self, probability, selected_population, elite_strategy_type, elite_strategy_value, k=None):

        new_pop = np.array([])

        if k is None:
            k = np.random.uniform(0., 1.)

        if ELITE_STRATEGY:
            if elite_strategy_type == "%":
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=None,
                                                               percent=elite_strategy_value)
                np.append(new_pop, elite_individuals)
            else:
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=elite_strategy_value,
                                                               percent=None)
                np.append(new_pop, elite_individuals)

        while new_pop.size < self.size:
            value = np.random.uniform(0., 1.)
            elements_to_cross = np.random.choice(selected_population, size=2, replace=False)

            if value < probability:
                new_individual_1 = RealIndividual(2, a=self.a, b=self.b, value=np.array([
                    (k * elements_to_cross[0].chromosomes[0]) + ((1 - k) * elements_to_cross[1].chromosomes[0]),
                    (k * elements_to_cross[0].chromosomes[1]) + ((1 - k) * elements_to_cross[1].chromosomes[1])])
                                                  )
                new_individual_2 = RealIndividual(2, a=self.a, b=self.b, value=np.array([
                    ((1 - k) * elements_to_cross[0].chromosomes[0]) + (k * elements_to_cross[1].chromosomes[0]),
                    ((1 - k) * elements_to_cross[0].chromosomes[1]) + (k * elements_to_cross[1].chromosomes[1])])
                                                  )
            else:
                new_individual_1 = elements_to_cross[0]
                new_individual_2 = elements_to_cross[1]

            if np.any(new_individual_1.chromosomes < self.a) or np.any(new_individual_1.chromosomes > self.b) or \
                    np.any(new_individual_2.chromosomes < self.a) or np.any(new_individual_2.chromosomes > self.b):
                continue

            new_pop = np.append(new_pop, new_individual_1)
            new_pop = np.append(new_pop, new_individual_2)

        return new_pop

    def linear_cross(self, probability, selected_population, elite_strategy_type, elite_strategy_value):
        new_pop = np.array([])

        if ELITE_STRATEGY:
            if elite_strategy_type == "%":
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=None,
                                                               percent=elite_strategy_value)
                np.append(new_pop, elite_individuals)
            else:
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=elite_strategy_value,
                                                               percent=None)
                np.append(new_pop, elite_individuals)

        while new_pop.size < self.size:
            value = np.random.uniform(0., 1.)
            elements_to_cross = np.random.choice(selected_population, size=2, replace=False)

            if value < probability:

                z = np.array([
                    (0.5 * elements_to_cross[0].chromosomes[0]) + (0.5 * elements_to_cross[1].chromosomes[0]),
                    (0.5 * elements_to_cross[0].chromosomes[1]) + (0.5 * elements_to_cross[1].chromosomes[1])
                ])

                v = np.array([
                    ((3. / 2.) * elements_to_cross[0].chromosomes[0]) - (0.5 * elements_to_cross[1].chromosomes[0]),
                    ((3. / 2.) * elements_to_cross[0].chromosomes[1]) - (0.5 * elements_to_cross[1].chromosomes[1])
                ])

                w = np.array([
                    (-0.5 * elements_to_cross[0].chromosomes[0]) + ((3. / 2.) * elements_to_cross[1].chromosomes[0]),
                    (-0.5 * elements_to_cross[0].chromosomes[1]) + ((3. / 2.) * elements_to_cross[1].chromosomes[1])
                ])

                evaluated = [self.fitness_function.compute(z), self.fitness_function.compute(v),
                             self.fitness_function.compute(w)]
                max_idx = evaluated.index(max(evaluated))

                if max_idx == 0:
                    new_individual_1 = RealIndividual(2, a=self.a, b=self.b, value=v)
                    new_individual_2 = RealIndividual(2, a=self.a, b=self.b, value=w)
                elif max_idx == 1:
                    new_individual_1 = RealIndividual(2, a=self.a, b=self.b, value=z)
                    new_individual_2 = RealIndividual(2, a=self.a, b=self.b, value=w)
                elif max_idx == 2:
                    new_individual_1 = RealIndividual(2, a=self.a, b=self.b, value=z)
                    new_individual_2 = RealIndividual(2, a=self.a, b=self.b, value=v)

            else:
                new_individual_1 = elements_to_cross[0]
                new_individual_2 = elements_to_cross[1]

            if np.any(new_individual_1.chromosomes < self.a) or np.any(new_individual_1.chromosomes > self.b) or \
                    np.any(new_individual_2.chromosomes < self.a) or np.any(new_individual_2.chromosomes > self.b):
                continue

            new_pop = np.append(new_pop, new_individual_1)
            new_pop = np.append(new_pop, new_individual_2)

        return new_pop

    def average_cross(self, probability, selected_population, elite_strategy_type, elite_strategy_value):
        new_pop = np.array([])

        if ELITE_STRATEGY:
            if elite_strategy_type == "%":
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=None,
                                                               percent=elite_strategy_value)
                np.append(new_pop, elite_individuals)
            else:
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=elite_strategy_value,
                                                               percent=None)
                np.append(new_pop, elite_individuals)

        while new_pop.size < self.size:
            value = np.random.uniform(0., 1.)
            elements_to_cross = np.random.choice(selected_population, size=2, replace=False)

            if value < probability:
                new_individual = RealIndividual(2, a=self.a, b=self.b, value=np.array([
                    (elements_to_cross[0].chromosomes[0] + elements_to_cross[1].chromosomes[0]) / 2.,
                    (elements_to_cross[0].chromosomes[1] + elements_to_cross[1].chromosomes[1]) / 2.
                ]))
            else:
                new_individual = np.random.choice(elements_to_cross, size=1)

            if np.any(new_individual.chromosomes < self.a) or np.any(new_individual.chromosomes > self.b):
                continue

            new_pop = np.append(new_pop, new_individual)

        return new_pop

    def blend_cross_alpha(self, probability, selected_population, alpha, elite_strategy_type, elite_strategy_value):

        new_pop = np.array([])

        if alpha is None:
            alpha = np.random.uniform(0., 1.)

        if ELITE_STRATEGY:
            if elite_strategy_type == "%":
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=None,
                                                               percent=elite_strategy_value)
                np.append(new_pop, elite_individuals)
            else:
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=elite_strategy_value,
                                                               percent=None)
                np.append(new_pop, elite_individuals)

        while new_pop.size < self.size:
            value = np.random.uniform(0., 1.)
            elements_to_cross = np.random.choice(selected_population, size=2, replace=False)

            if value < probability:
                min_x = np.min([elements_to_cross[0].chromosomes[0], elements_to_cross[1].chromosomes[0]])
                min_y = np.min([elements_to_cross[0].chromosomes[1], elements_to_cross[1].chromosomes[1]])

                dx = np.abs(elements_to_cross[0].chromosomes[0] - elements_to_cross[1].chromosomes[0])
                dy = np.abs(elements_to_cross[0].chromosomes[1] - elements_to_cross[1].chromosomes[1])

                x1_new = np.random.uniform(min_x - (alpha * dx), min_x + (alpha * dx))
                y1_new = np.random.uniform(min_y - (alpha * dy), min_y + (alpha * dy))
                new_individual_1 = RealIndividual(2, self.a, self.b, np.array([x1_new, y1_new]))

                x2_new = np.random.uniform(min_x - (alpha * dx), min_x + (alpha * dx))
                y2_new = np.random.uniform(min_y - (alpha * dy), min_y + (alpha * dy))
                new_individual_2 = RealIndividual(2, self.a, self.b, np.array([x2_new, y2_new]))
            else:
                new_individual_1 = elements_to_cross[0]
                new_individual_2 = elements_to_cross[1]

            if np.any(new_individual_1.chromosomes < self.a) or np.any(new_individual_1.chromosomes > self.b) or \
                    np.any(new_individual_2.chromosomes < self.a) or np.any(new_individual_2.chromosomes > self.b):
                continue

            new_pop = np.append(new_pop, new_individual_1)
            new_pop = np.append(new_pop, new_individual_2)

        return new_pop

    def blend_cross_alpha_beta(self, probability, selected_population, alpha, beta, elite_strategy_type,
                               elite_strategy_value):

        new_pop = np.array([])

        if alpha is None:
            alpha = np.random.uniform(0., 1.)
        if beta is None:
            beta = np.random.uniform(0., 1.)

        if ELITE_STRATEGY:
            if elite_strategy_type == "%":
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=None,
                                                               percent=elite_strategy_value)
                np.append(new_pop, elite_individuals)
            else:
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=elite_strategy_value,
                                                               percent=None)
                np.append(new_pop, elite_individuals)

        while new_pop.size < self.size:
            value = np.random.uniform(0., 1.)
            elements_to_cross = np.random.choice(selected_population, size=2, replace=False)

            if value < probability:
                min_x = np.min([elements_to_cross[0].chromosomes[0], elements_to_cross[1].chromosomes[0]])
                min_y = np.min([elements_to_cross[0].chromosomes[1], elements_to_cross[1].chromosomes[1]])

                dx = np.abs(elements_to_cross[0].chromosomes[0] - elements_to_cross[1].chromosomes[0])
                dy = np.abs(elements_to_cross[0].chromosomes[1] - elements_to_cross[1].chromosomes[1])

                x1_new = np.random.uniform(min_x - (alpha * dx), min_x + (beta * dx))
                y1_new = np.random.uniform(min_y - (alpha * dy), min_y + (beta * dy))
                new_individual_1 = RealIndividual(2, self.a, self.b, np.array([x1_new, y1_new]))

                x2_new = np.random.uniform(min_x - (alpha * dx), min_x + (beta * dx))
                y2_new = np.random.uniform(min_y - (alpha * dy), min_y + (beta * dy))
                new_individual_2 = RealIndividual(2, self.a, self.b, np.array([x2_new, y2_new]))
            else:
                new_individual_1 = elements_to_cross[0]
                new_individual_2 = elements_to_cross[1]

            if np.any(new_individual_1.chromosomes < self.a) or np.any(new_individual_1.chromosomes > self.b) or \
                    np.any(new_individual_2.chromosomes < self.a) or np.any(new_individual_2.chromosomes > self.b):
                continue

            new_pop = np.append(new_pop, new_individual_1)
            new_pop = np.append(new_pop, new_individual_2)

        return new_pop

    def regular_mutation(self, probability, population):
        new_pop = np.array([])

        for i in population:
            value = np.random.uniform(0., 1.)

            if value < probability:
                i.regular_mutate()

            new_pop = np.append(new_pop, RealIndividual(2, self.a, self.b, i.chromosomes))

        return new_pop

    def gauss_mutation(self, probability, population):
        new_pop = np.array([])

        for i in population:
            value = np.random.uniform(0., 1.)

            if value < probability:
                i.gauss_mutate()

            new_pop = np.append(new_pop, RealIndividual(2, self.a, self.b, i.chromosomes))

        return new_pop

    ############################################################

    def evaluate(self):

        self.decoded_population = []

        for individual in self.population:
            decoded_individual = individual.decode(self.a, self.b, self.chromosome_size)
            self.decoded_population.append(decoded_individual)

        self.evaluated_population = np.array([self.fitness_function.compute(individual)
                                              for individual in self.decoded_population])

        return self.evaluated_population

    def get_elite_individuals(self, evaluated_pop, percent=None, number=None):

        sorted_indices = np.argsort(evaluated_pop)

        if percent is not None:
            smallest_indices = sorted_indices[:int(percent * evaluated_pop.size)]

        if number is not None:
            smallest_indices = sorted_indices[:number]

        return self.population[smallest_indices]

    def single_point_cross(self, probability, selected_population, elite_strategy_type, elite_strategy_value):

        new_pop = np.array([])

        if ELITE_STRATEGY:
            if elite_strategy_type == "%":
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=None,
                                                               percent=elite_strategy_value)
                np.append(new_pop, elite_individuals)
            else:
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=elite_strategy_value,
                                                               percent=None)
                np.append(new_pop, elite_individuals)

        while new_pop.size < self.size:
            value = np.random.uniform(0., 1.)
            elements_to_cross = np.random.choice(selected_population, size=2, replace=False)

            if value < probability:
                index_of_crossing = int(np.random.randint(self.chromosome_size * 2))

                new_element1 = Individual(2, self.chromosome_size,
                                          np.append(elements_to_cross[0].get_individual()[:index_of_crossing],
                                                    elements_to_cross[1].get_individual()[index_of_crossing:]))
                new_element2 = Individual(2, self.chromosome_size,
                                          np.append(elements_to_cross[1].get_individual()[:index_of_crossing],
                                                    elements_to_cross[0].get_individual()[index_of_crossing:]))
            else:
                new_element1 = elements_to_cross[0]
                new_element2 = elements_to_cross[1]

            new_pop = np.append(new_pop, new_element1)
            new_pop = np.append(new_pop, new_element2)

        return new_pop

    def double_point_cross(self, probability, selected_population, elite_strategy_type, elite_strategy_value):
        new_pop = np.array([])

        if ELITE_STRATEGY:
            if elite_strategy_type == "%":
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=None,
                                                               percent=elite_strategy_value)
                np.append(new_pop, elite_individuals)
            else:
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=elite_strategy_value,
                                                               percent=None)
                np.append(new_pop, elite_individuals)

        while new_pop.size < self.size:
            value = np.random.uniform(0., 1.)
            elements_to_cross = np.random.choice(selected_population, size=2, replace=False)

            if value < probability:
                ind_of_crossing = np.sort(np.random.choice(self.chromosome_size * 2, size=2, replace=False))

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

            else:
                new_element1 = elements_to_cross[0]
                new_element2 = elements_to_cross[1]

            new_pop = np.append(new_pop, new_element1)
            new_pop = np.append(new_pop, new_element2)

        return new_pop

    def triple_point_cross(self, probability, selected_population, elite_strategy_type, elite_strategy_value):
        new_pop = np.array([])

        if ELITE_STRATEGY:
            if elite_strategy_type == "%":
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=None,
                                                               percent=elite_strategy_value)
                np.append(new_pop, elite_individuals)
            else:
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=elite_strategy_value,
                                                               percent=None)
                np.append(new_pop, elite_individuals)

        while new_pop.size < self.size:
            value = np.random.uniform(0., 1.)

            elements_to_cross = np.random.choice(selected_population, size=2, replace=False)

            if value < probability:
                ind_of_crossing = np.sort(np.random.choice(self.chromosome_size * 2, size=3, replace=False))
                new_element1 = Individual(2, self.chromosome_size, np.concatenate((
                    elements_to_cross[0].get_individual()[:ind_of_crossing[0]],
                    elements_to_cross[1].get_individual()[ind_of_crossing[0]:ind_of_crossing[1]],
                    elements_to_cross[0].get_individual()[ind_of_crossing[1]:ind_of_crossing[2]],
                    elements_to_cross[1].get_individual()[ind_of_crossing[2]:]
                )))

                new_element2 = Individual(2, self.chromosome_size, np.concatenate((
                    elements_to_cross[1].get_individual()[:ind_of_crossing[0]],
                    elements_to_cross[0].get_individual()[ind_of_crossing[0]:ind_of_crossing[1]],
                    elements_to_cross[1].get_individual()[ind_of_crossing[1]:ind_of_crossing[2]],
                    elements_to_cross[0].get_individual()[ind_of_crossing[2]:]
                )))
            else:
                new_element1 = elements_to_cross[0]
                new_element2 = elements_to_cross[1]

            new_pop = np.append(new_pop, new_element1)
            new_pop = np.append(new_pop, new_element2)

        return new_pop

    def homogeneous_cross(self, probability, selected_population, elite_strategy_type, elite_strategy_value):
        new_pop = np.array([])

        if ELITE_STRATEGY:
            if elite_strategy_type == "%":
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=None,
                                                               percent=elite_strategy_value)
                np.append(new_pop, elite_individuals)
            else:
                elite_individuals = self.get_elite_individuals(self.evaluated_population,
                                                               number=elite_strategy_value,
                                                               percent=None)
                np.append(new_pop, elite_individuals)

        while new_pop.size < self.size:
            value = np.random.uniform(0., 1.)
            elements_to_cross = np.random.choice(selected_population, size=2, replace=False)
            if value < probability:

                new_element1 = np.zeros(self.chromosome_size * 2, dtype=int)
                new_element2 = np.zeros(self.chromosome_size * 2, dtype=int)

                for i in range(self.chromosome_size * 2):
                    if i % 2 == 0:
                        new_element1[i] = elements_to_cross[0].get_individual()[i]
                        new_element2[i] = elements_to_cross[1].get_individual()[i]
                    else:
                        new_element1[i] = elements_to_cross[1].get_individual()[i]
                        new_element2[i] = elements_to_cross[0].get_individual()[i]

                new_individual1 = Individual(2, self.chromosome_size, new_element1)
                new_individual2 = Individual(2, self.chromosome_size, new_element2)

            else:
                new_individual1 = elements_to_cross[0]
                new_individual2 = elements_to_cross[1]

            new_pop = np.append(new_pop, new_individual1)
            new_pop = np.append(new_pop, new_individual2)

        return new_pop

    def boundary_mutation(self, probability, population):
        new_pop = np.array([])

        for i in population:
            value = np.random.uniform(0., 1.)

            if value < probability:
                i.boundary_mutate()

            new_pop = np.append(new_pop, Individual(2, self.chromosome_size, i.get_individual()))

        return new_pop

    def single_point_mutation(self, probability, population):
        new_pop = np.array([])

        for i in population:
            value = np.random.uniform(0., 1.)

            if value < probability:
                i.single_point_mutate()

            new_pop = np.append(new_pop, Individual(2, self.chromosome_size, i.get_individual()))

        return new_pop

    def double_point_mutation(self, probability, population):
        new_pop = np.array([])

        for i in population:
            value = np.random.uniform(0., 1.)

            if value < probability:
                i.double_point_mutate()

            new_pop = np.append(new_pop, Individual(2, self.chromosome_size, i.get_individual()))

        return new_pop

    def inversion(self, probability, population):
        new_pop = np.array([])

        for i in population:
            value = np.random.uniform(0., 1.)

            if value < probability:
                i.invert()

            new_pop = np.append(new_pop, Individual(2, self.chromosome_size, i.get_individual()))

        return new_pop
