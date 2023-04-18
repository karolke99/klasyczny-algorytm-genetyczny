import math

import numpy as np


class FitnessFunction:

    def __init__(self, a=None, b=None, precision=None):
        self.a = a
        self.b = b

    def compute(self, params):
        return -20 * np.exp(-0.2 * np.sqrt((1 / 2) * (params[0] ** 2 + params[1] ** 2))) - np.exp(
            (1 / 2) * (np.cos(2 * np.pi * params[0]) + np.cos(2 * np.pi * params[1]))) + 20 + np.exp(1)

