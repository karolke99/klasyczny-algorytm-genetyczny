import numpy as np


class Individual:

    def __init__(self, individual_size=2, chromosome_size=None, value=None):

        if value is None:
            self.chromosomes = np.random.choice([0, 1], size=(individual_size * chromosome_size))
        else:
            self.chromosomes = value
        self.chromosome_size = chromosome_size

    def __str__(self):
        return str(self.chromosomes)

    def get_individual(self):
        return self.chromosomes

    def decode(self, a=None, b=None, chromosome_size=None):
        decoded_array = []

        for chromosome in np.split(self.chromosomes, 2):
            binary_string_chromosome = ''.join(chromosome.astype(str))
            decimal_val = int(binary_string_chromosome, 2)
            decoded_chromosome = a + decimal_val * (b - a) / ((2 ** chromosome_size) - 1)
            decoded_array.append(decoded_chromosome)

        return np.array(decoded_array)

    def boundary_mutate(self):
        self.chromosomes[-1] = ~self.chromosomes.astype(bool)[-1]

    def single_point_mutate(self):
        index = np.random.randint(0, self.chromosome_size * 2)
        print(index )
        self.chromosomes[index] = ~self.chromosomes.astype(bool)[index]
