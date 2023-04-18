import numpy as np


class RealIndividual:

    def __init__(self, individual_size=2, a=-10, b=10, value=None):
        self.size = individual_size

        if value is None:
            self.chromosomes = np.random.uniform(low=a, high=b, size=(2,))
