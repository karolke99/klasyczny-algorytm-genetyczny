class Configuration:

    population_size = None
    accuracy = None
    epoch = None
    selection_type = None
    tournament_size = None
    elite_group_type = None
    elite_group_size = None
    crossing_type = None
    crossing_probability = None
    mutation_type = None
    mutation_probability = None
    inversion_probability = None

    def get_population_size(self):
        return self.population_size

    def get_accuracy(self):
        return self.accuracy

    def get_epoch(self):
        return self.epoch

    def get_selection_type(self):
        return self.selection_type

    def get_tournament_size(self):
        return self.tournament_size

    def get_elite_group_type(self):
        return self.elite_group_type

    def get_elite_group_size(self):
        return self.elite_group_size

    def get_crossing_type(self):
        return self.crossing_type

    def get_crossing_probability(self):
        return self.crossing_probability

    def get_mutation_type(self):
        return self.mutation_type

    def get_mutation_probability(self):
        return self.mutation_probability

    def get_inversion_probability(self):
        return self.inversion_probability

    def set_population_size(self, value):
        size = int(value.get())
        if size <= 0:
            raise Exception("Population size must be greater than 0")
        self.population_size = size

    def set_accuracy(self, value):
        accuracy = int(value.get())
        if accuracy <= 0:
            raise Exception("Accuracy must be greater than 0")
        self.accuracy = accuracy

    def set_epoch(self, value):
        epoch = int(value.get())
        if epoch <= 0:
            raise Exception("Epoch number must be greater than 0")
        self.epoch = epoch

    def set_selection_type(self, value):
        self.selection_type = value

    def set_tournament_size(self, value):
        size = int(value.get())
        if size < 0:
            raise Exception("Tournament size must be greater than 0")
        self.tournament_size = size

    def set_elite_group_type(self, value):
        self.elite_group_type = value

    def set_elite_group_size(self, value):
        size = int(value.get())
        if size < 0:
            raise Exception("Elite group size must be equal or greater than 0")
        self.elite_group_size = size

    def set_crossing_type(self, value):
        self.crossing_type = value

    def set_crossing_probability(self, value):
        probability = float(value.get())
        if probability < 0 or probability > 1:
            raise Exception("Probability must fit into <0, 1> range")
        self.crossing_probability = probability

    def set_mutation_type(self, value):
        self.mutation_type = value

    def set_mutation_probability(self, value):
        probability = float(value.get())
        if probability < 0 or probability > 1:
            raise Exception("Probability must fit into <0, 1> range")
        self.mutation_probability = probability

    def set_inversion_probability(self, value):
        probability = float(value.get())
        if probability < 0 or probability > 1:
            raise Exception("Probability must fit into <0, 1> range")
        self.inversion_probability = probability

    def __str__(self) -> str:
        return str(self.population_size) + "\n "\
                + str(self.accuracy) + "\n " \
                + str(self.epoch) + "\n " + self.selection_type + "\n " + str(self.tournament_size) + "\n " \
                + self.elite_group_type + "\n " + str(self.elite_group_size) + "\n " + self.crossing_type + "\n " \
                + str(self.crossing_probability) + "\n " + self.mutation_type + "\n " + str(self.mutation_probability) \
                + "\n " + str(self.inversion_probability)
#





