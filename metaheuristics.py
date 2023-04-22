from numpy import exp
import random
from fitness_function import FitnessFunction
from numpy import asarray
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed

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
        bounds = asarray([[-32.768, 32.768]])
        n_iterations = 5000
        step_size = 0.1
        solution = list()
        solution.append(bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
        solution.append(bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
        solution_eval = self.objective(solution)
        scores = list()
        scores.append(solution_eval)
        for i in range(n_iterations):
            candidate = list()
            candidate.append(solution[0] + randn(len(bounds)) * step_size)
            candidate.append(solution[1] + randn(len(bounds)) * step_size)
            candidte_eval = self.objective(candidate)
            if candidte_eval <= solution_eval:
                solution, solution_eval = candidate, candidte_eval
                scores.append(solution_eval)
                # print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
        print(f"Best value: {solution_eval}")
        return [solution, solution_eval, scores]

    def simulated_annealing(self):
        # define range for input
        bounds = asarray([[-32.768, 32.768]])
        # define the total iterations
        n_iterations = 50000
        # define the maximum step size
        step_size = 0.4
        # initial temperature
        temp = 10
        best = list()
        best.append(bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
        best.append(bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
        # evaluate the initial point
        best_eval = self.objective(best)
        # current working solution
        curr, curr_eval = best, best_eval
        scores = list()
        # run the algorithm
        for i in range(n_iterations):
            candidate = list()
            candidate.append(curr[0] + randn(len(bounds)) * step_size)
            candidate.append(curr[1] + randn(len(bounds)) * step_size)
            # evaluate candidate point
            candidate_eval = self.objective(candidate)
            # check for new best solution
            if candidate_eval < best_eval:
                # store new best point
                best, best_eval = candidate, candidate_eval
                # keep track of scores
                scores.append(best_eval)
                # report progress
                # print('>%d f(%s) = %.5f' % (i, best, best_eval))
            # difference between candidate and current point evaluation
            diff = candidate_eval - curr_eval
            # calculate temperature for current epoch
            t = temp / float(i + 1)
            # calculate metropolis acceptance criterion
            metropolis = exp(-diff / t)
            # check if we should keep the new point
            if diff < 0 or rand() < metropolis:
                # store the new current point
                curr, curr_eval = candidate, candidate_eval
        print(f"Best value: {curr_eval}")
        return [best, best_eval, scores]



