import numpy as np


class RealIndividual:

    def __init__(self, individual_size=2, a=-10, b=10, value=None):
        self.size = individual_size

        self.a = a
        self.b = b

        if value is None:
            self.chromosomes = np.random.uniform(low=a, high=b, size=(2,))
        else:
            self.chromosomes = value

    def __str__(self):
        return str(self.chromosomes)

    def regular_mutate(self):
        idx = np.random.randint(2)
        new_chromosome_value = np.random.uniform(low=self.a, high=self.b)
        self.chromosomes[idx] = new_chromosome_value

    def gauss_mutate(self):
        n1 = 1000
        while (self.chromosomes[0] + n1) < self.a or ((self.chromosomes[0] + n1) > self.b):
            n1 = np.random.normal()

        self.chromosomes[0] = self.chromosomes[0] + n1

        n2 = 1000
        while (self.chromosomes[1] + n2) < self.a or ((self.chromosomes[1] + n2) > self.b):
            n2 = np.random.normal()

        self.chromosomes[1] = self.chromosomes[1] + n2
