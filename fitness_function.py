import math

import numpy as np


class FitnessFunction:

    def __init__(self, a=None, b=None, precision=None):
        self.a = a
        self.b = b
        self.precision = precision

        self.chromosome_size = math.ceil(
            math.log2((self.b - self.a) * (10 ** precision)) + math.log2(1)
        )

    def get_chromosome_size(self):
        return self.chromosome_size

    def compute(self, params):
        return -20 * np.exp(-0.2 * np.sqrt((1 / 2) * (params[0] ** 2 + params[1] ** 2))) - np.exp(
            (1 / 2) * (np.cos(2 * np.pi * params[0]) + np.cos(2 * np.pi * params[1]))) + 20 + np.exp(1)

