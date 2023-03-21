import numpy as np


class Individual:

    def __init__(self, individual_size=2, chromosome_size=None):
        self.chromosomes = np.random.choice([0, 1], size=(individual_size, chromosome_size))

    def __str__(self):
        return str(self.chromosomes)

    def decode(self, a=None, b=None, chromosome_size=None):

        decoded_array = []

        for chromosome in self.chromosomes:
            binary_string_chromosome = ''.join(chromosome.astype(str))
            # print(binary_string_chromosome)
            decimal_val = int(binary_string_chromosome, 2)
            # print(decimal_val)
            decoded_chromosome = a + decimal_val * (b-a)/((2 ** chromosome_size) - 1)
            # print(decoded_chromosome)
            decoded_array.append(decoded_chromosome)

        return np.array(decoded_array)



