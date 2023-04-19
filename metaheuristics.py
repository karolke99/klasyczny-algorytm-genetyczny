
import random
from fitness_function import FitnessFunction


class Metaheuristics(FitnessFunction):

    def __init__(self):
        super().__init__()

    def objective(self, x: list):
        return self.compute(x)

    def random_sampling(self):
        best = list()
        best.append(random.uniform(-32.768, 32.768))
        best.append(random.uniform(-32.768, 32.768))
        for x in range(0, 1000000):
            tmp = list()
            tmp.append(random.uniform(-32.768, 32.768))
            tmp.append(random.uniform(-32.768, 32.768))
            if self.objective(tmp) < self.objective(best):
                best = tmp
            # print(self.objective(tmp))

        print('Result:')
        print('x:', best)
        print('Fitness:', self.objective(best))

    def random_walk(self):
        best = list()
        best.append(random.uniform(-32.768, 32.768))
        best.append(random.uniform(-32.768, 32.768))
        step_size = 0.5

        for x in range(0, 10000):
            tmp = list()
            tmp.append(best[0] + random.uniform(-32.768, 32.768)* step_size)
            tmp.append(best[1] + random.uniform(-32.768, 32.768)* step_size)
            best = tmp
            print(self.objective(tmp))

        print('Result:')
        print('x:', best)
        print('Fitness:', self.objective(best))

    def hill_climbing(self):
        pass

    def simulated_annealing(self):
        pass


