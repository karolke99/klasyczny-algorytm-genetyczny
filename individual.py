import numpy as np
from numpy import float32


class Individual:

    def __init__(self, genome):
        self.genome = genome

    def __str__(self):
        return str(self.genome)

    def decode(self, a=None, b=None, chromosome_size=None):

        decoded_array = []

        for chromosome in self.genome:
            binary_string_chromosome = ''.join(chromosome.astype(str))
            # print(binary_string_chromosome)
            decimal_val = int(binary_string_chromosome, 2)
            # print(decimal_val)
            decoded_chromosome = a + decimal_val * (b-a)/((2 ** chromosome_size) - 1)
            # print(decoded_chromosome)
            decoded_array.append(decoded_chromosome)


        # tutaj problem z precyzją, większa jest dla listy pythonowej niż dla ndarray
        # return decoded_array
        return np.array(decoded_array)



    # def evaluate(self):
