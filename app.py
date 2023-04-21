import tkinter
import tkinter.messagebox

import customtkinter
import numpy as np

from configuration import Configuration
from util import EliteStrategyType, SelectionType, CrossingType, MutationType

from fitness_function import *
from population import *
import pandas as pd
import time
import matplotlib.pyplot as plt

customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    configuration = Configuration()

    def __init__(self):
        super().__init__()

        self.title("Algorytm genetyczny")
        self.geometry(f"{750}x{550}")

        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1)

        # population
        self.evolution_frame = customtkinter.CTkFrame(self)
        self.evolution_frame.grid(row=0, column=0, sticky="nesw")

        self.population_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Population")
        self.population_label.grid(row=0, column=1, columnspan=2, padx=10, pady=20)

        self.population_size_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Population size:")
        self.population_size_label.grid(row=1, column=1, padx=10, pady=5)
        self.population_size_var = tkinter.StringVar(value='0')
        self.population_size_entry = customtkinter.CTkEntry(self.evolution_frame,
                                                            textvariable=self.population_size_var)
        self.population_size_entry.grid(row=1, column=2, padx=10, pady=5)

        self.accuracy_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Accuracy:")
        self.accuracy_label.grid(row=2, column=1, padx=10, pady=5)
        self.accuracy_var = tkinter.StringVar(value='0')
        self.accuracy_entry = customtkinter.CTkEntry(self.evolution_frame, textvariable=self.accuracy_var)
        self.accuracy_entry.grid(row=2, column=2, padx=10, pady=5)

        # Epochs
        self.epochs_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Epochs")
        self.epochs_label.grid(row=3, column=1, columnspan=2, padx=10, pady=20)

        self.epochs_number_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Epoch number:")
        self.epochs_number_label.grid(row=4, column=1, padx=10, pady=5)
        self.epochs_number_var = tkinter.StringVar(value='0')
        self.epochs_entry = customtkinter.CTkEntry(self.evolution_frame, textvariable=self.epochs_number_var)
        self.epochs_entry.grid(row=4, column=2, padx=20, pady=5)

        # selection
        self.selection_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Selection")
        self.selection_label.grid(row=5, column=1, columnspan=3, padx=10, pady=20)

        self.selection_type_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Selection type:")
        self.selection_type_label.grid(row=6, column=1, padx=10, pady=5)
        self.selection_dropdown = customtkinter.CTkOptionMenu(self.evolution_frame,
                                                              values=list(SelectionType.values()),
                                                              command=self.change_selection_event)
        self.selection_dropdown.grid(row=6, column=2, pady=5, padx=20)

        self.tournament_size = customtkinter.CTkLabel(master=self.evolution_frame, text="Selection size:")
        self.tournament_size.grid(row=7, column=1, padx=10, pady=5)
        self.tournament_size_var = tkinter.StringVar(value='0')
        self.tournament_size_entry = customtkinter.CTkEntry(self.evolution_frame, textvariable=self.tournament_size_var)
        self.tournament_size_entry.grid(row=7, column=2, padx=20, pady=5)

        # elite strategy
        self.elite_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Elite strategy")
        self.elite_label.grid(row=8, column=1, columnspan=2, padx=10, pady=20)

        self.elite_group_size_var = tkinter.StringVar(value='0')
        self.elite_group_size_entry = customtkinter.CTkEntry(self.evolution_frame,
                                                             textvariable=self.elite_group_size_var)
        self.elite_group_size_entry.grid(row=9, column=1, padx=20, pady=5)

        self.elite_dropdown = customtkinter.CTkOptionMenu(self.evolution_frame,
                                                          values=list(EliteStrategyType.values()),
                                                          command=self.change_elite_strategy_event)
        self.elite_dropdown.grid(row=9, column=2, pady=5, padx=20)

        # crossing
        self.genetic_operators_frame = customtkinter.CTkFrame(self)
        self.genetic_operators_frame.grid(row=0, column=1, rowspan=3, sticky="nsew")
        self.Crossing_label = customtkinter.CTkLabel(master=self.genetic_operators_frame, text="Crossing")
        self.Crossing_label.grid(row=0, column=1, columnspan=2, padx=10, pady=20)

        self.Crossing_type_label = customtkinter.CTkLabel(master=self.genetic_operators_frame, text="Crossing type:")
        self.Crossing_type_label.grid(row=1, column=1, padx=10, pady=5)
        self.crossing_dropdown = customtkinter.CTkOptionMenu(self.genetic_operators_frame,
                                                             values=list(CrossingType.values()),
                                                             command=self.change_crossing_type_event)
        self.crossing_dropdown.grid(row=1, column=2, pady=5, padx=20, sticky="n")

        self.crossing_probability_var = tkinter.StringVar(value='0')
        self.crossing_probability_label = customtkinter.CTkLabel(master=self.genetic_operators_frame,
                                                                 text="Probability:")
        self.crossing_probability_label.grid(row=2, column=1, padx=10, pady=5)
        self.crossing_probability_entry = customtkinter.CTkEntry(self.genetic_operators_frame,
                                                                 textvariable=self.crossing_probability_var)
        self.crossing_probability_entry.grid(row=2, column=2, padx=20, pady=5, sticky="nsew")

        self.alpha_var = tkinter.StringVar(value='0')
        self.alpha_label = customtkinter.CTkLabel(master=self.genetic_operators_frame, text="Alpha:")
        self.alpha_label.grid(row=3, column=1, padx=10, pady=5)
        self.alpha_entry = customtkinter.CTkEntry(self.genetic_operators_frame, textvariable=self.alpha_var, width=10)
        self.alpha_entry.grid(row=3, column=2, padx=20, pady=5, sticky="nsew")

        self.beta_var = tkinter.StringVar(value='0')
        self.beta_label = customtkinter.CTkLabel(master=self.genetic_operators_frame, text="Beta:")
        self.beta_label.grid(row=4, column=1, padx=10, pady=5)
        self.beta_entry = customtkinter.CTkEntry(self.genetic_operators_frame, textvariable=self.beta_var)
        self.beta_entry.grid(row=4, column=2, padx=20, pady=5, sticky="nsew")

        self.mutating_label = customtkinter.CTkLabel(master=self.genetic_operators_frame, text="Mutating")
        self.mutating_label.grid(row=5, column=1, columnspan=2, padx=10, pady=20)

        self.mutation_type_label = customtkinter.CTkLabel(master=self.genetic_operators_frame, text="Mutating type:")
        self.mutation_type_label.grid(row=6, column=1, padx=10, pady=5)
        self.mutation_type_dropdown = customtkinter.CTkOptionMenu(self.genetic_operators_frame,
                                                                  values=list(MutationType.values()),
                                                                  command=self.change_mutation_type_event)
        self.mutation_type_dropdown.grid(row=6, column=2, pady=5, padx=20, sticky="n")

        self.mutation_probability_var = tkinter.StringVar(value='0')
        self.mutation_probability_label = customtkinter.CTkLabel(master=self.genetic_operators_frame,
                                                                 text="Probability:")
        self.mutation_probability_label.grid(row=7, column=1, padx=10, pady=5)
        self.mutation_probability_entry = customtkinter.CTkEntry(self.genetic_operators_frame,
                                                                 textvariable=self.mutation_probability_var)
        self.mutation_probability_entry.grid(row=7, column=2, padx=20, pady=5, sticky="nsew")

        # results
        self.start_button = customtkinter.CTkButton(master=self.genetic_operators_frame, fg_color="transparent",
                                                    border_width=2, text_color=("gray10", "#DCE4EE"), text="Start",
                                                    command=self.start)
        self.start_button.grid(row=10, column=1, padx=20, pady=(30, 10))

        self.result_label = customtkinter.CTkLabel(master=self.genetic_operators_frame, text="x, y:")
        self.result_label.grid(row=11, column=1, padx=20, pady=5)
        self.result_textbox = customtkinter.CTkTextbox(self.genetic_operators_frame, width=150, height=20)
        self.result_textbox.grid(row=11, column=2, padx=(0, 0), pady=(20, 0))

        self.fx_label = customtkinter.CTkLabel(master=self.genetic_operators_frame, text="F(x):")
        self.fx_label.grid(row=12, column=1, padx=20, pady=5)
        self.fx_textbox = customtkinter.CTkTextbox(self.genetic_operators_frame, width=150, height=20)
        self.fx_textbox.grid(row=12, column=2, pady=(20, 0))

        # defaults
        self.tournament_size_entry.configure(state="readonly")
        self.fx_textbox.configure(state="disabled")
        self.result_textbox.configure(state="disabled")

        self.configuration.set_selection_type(SelectionType['ROULETTE'])
        self.configuration.set_elite_group_type(EliteStrategyType['PERCENT'])
        self.configuration.set_crossing_type(CrossingType['ARITHMETIC_POINT'])
        self.configuration.set_mutation_type(MutationType['EVEN'])

    def start(self):
        try:
            self.configuration.set_population_size(self.population_size_var)
            self.configuration.set_epoch(self.epochs_number_var)
            self.configuration.set_accuracy(self.accuracy_var)
            self.configuration.set_tournament_size(self.tournament_size_var)
            self.configuration.set_elite_group_size(self.elite_group_size_var)
            self.configuration.set_mutation_probability(self.mutation_probability_var)
            self.configuration.set_crossing_probability(self.crossing_probability_var)
            self.configuration.set_alpha(self.alpha_var)
            self.configuration.set_beta(self.beta_var)

            x, fx = self.run_algorithm()

            self.fx_textbox.configure(state="normal")
            self.result_textbox.configure(state="normal")

            self.result_textbox.insert("0.0", str(x))
            self.fx_textbox.insert("0.0", str(fx))

            self.fx_textbox.configure(state="disabled")
            self.result_textbox.configure(state="disabled")

        except Exception as e:
            self.open_input_dialog_event(e)

    def run_algorithm(self):
        evaluation_pop_list = list()
        mean = list()
        standard_deviation = list()
        best_value = list()
        fitness_fun = FitnessFunction(self.configuration.A, self.configuration.B,
                                      precision=self.configuration.accuracy)

        population = Population(a=self.configuration.A, b=self.configuration.B,
                                pop_size=self.configuration.population_size,
                                fitness_function=fitness_fun)

        evaluation_pop_list.append(population.evaluate_real())
        evaluated_pop = population.evaluate_real()
        mean.append(np.mean(evaluated_pop))
        standard_deviation.append(np.std(evaluated_pop))
        best_value.append(np.max(evaluated_pop))
        time_start = time.time()

        for i in range(self.configuration.epoch):

            # selection
            if self.configuration.selection_type == SelectionType["ROULETTE"]:
                selected_pop = population.select_roulette()
            elif self.configuration.selection_type == SelectionType["TOURNAMENT"]:
                selected_pop = population.select_tournament(self.configuration.tournament_size)
            elif self.configuration.selection_type == SelectionType["BEST"]:
                selected_pop = population.select_best(self.configuration.tournament_size)

            # crossover
            if self.configuration.crossing_type == CrossingType["ARITHMETIC_POINT"]:
                crossover_pop = population.arithmetic_cross(self.configuration.crossing_probability,
                                                            selected_pop,
                                                            self.configuration.get_elite_group_type(),
                                                            self.configuration.get_elite_group_size())
            elif self.configuration.crossing_type == CrossingType["LINEAR_POINTS"]:
                crossover_pop = population.linear_cross(self.configuration.crossing_probability,
                                                        selected_pop,
                                                        self.configuration.get_elite_group_type(),
                                                        self.configuration.get_elite_group_size())
            elif self.configuration.crossing_type == CrossingType["BLEND_ALFA"]:
                crossover_pop = population.blend_cross_alpha(self.configuration.crossing_probability,
                                                             selected_pop,
                                                             self.configuration.get_alpha(),
                                                             self.configuration.get_elite_group_type(),
                                                             self.configuration.get_elite_group_size())
            elif self.configuration.crossing_type == CrossingType["BLEND_ALFA_BETA"]:
                crossover_pop = population.blend_cross_alpha_beta(self.configuration.crossing_probability,
                                                                  selected_pop,
                                                                  self.configuration.get_alpha(),
                                                                  self.configuration.get_beta(),
                                                                  self.configuration.get_elite_group_type(),
                                                                  self.configuration.get_elite_group_size())
            elif self.configuration.crossing_type == CrossingType["AVERAGE"]:
                crossover_pop = population.average_cross(self.configuration.crossing_probability,
                                                         selected_pop,
                                                         self.configuration.get_elite_group_type(),
                                                         self.configuration.get_elite_group_size())

            # mutation
            if self.configuration.mutation_type == MutationType["EVEN"]:
                mutated_pop = population.regular_mutation(self.configuration.mutation_probability, crossover_pop)
            elif self.configuration.mutation_type == MutationType["GAUSS"]:
                mutated_pop = population.gauss_mutation(self.configuration.mutation_probability, crossover_pop)

            population = Population(self.configuration.A,
                                    b=self.configuration.B,
                                    pop_size=self.configuration.population_size,
                                    fitness_function=fitness_fun,
                                    value=mutated_pop)

            evaluated_pop = population.evaluate_real()
            mean.append(np.mean(evaluated_pop))
            standard_deviation.append(np.std(evaluated_pop))
            best_value.append(np.amin(evaluated_pop))
            evaluation_pop_list.append(population.evaluate_real())
            # print(evaluated_pop)

        time_end = time.time()

        print(f'Calculation time: {time_end - time_start}')

        # min_individual = np.argmin(evaluated_pop)

        min_individual = population.population[np.argmin(evaluated_pop)]
        self.generate_mean_plot(mean)
        self.generate_standard_deviation_plot(standard_deviation)
        self.generate_best_value_plot(best_value)
        df = pd.DataFrame(evaluation_pop_list)
        df.to_csv("Data.csv")
        return (min_individual, np.amin(evaluated_pop))

    def generate_mean_plot(self, mean_list):
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot(range(len(mean_list)), mean_list)
        plt.xlabel("epoch")
        plt.ylabel("mean")
        fig.savefig('mean.png')  # save the figure to file
        plt.close(fig)

    def generate_standard_deviation_plot(self, standard_deviation_list):
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot(range(len(standard_deviation_list)), standard_deviation_list)
        plt.xlabel("epoch")
        plt.ylabel("standard deviation")
        fig.savefig('standard_deviation.png')  # save the figure to file
        plt.close(fig)

    def generate_best_value_plot(self, best_value):
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot(range(len(best_value)), best_value)
        plt.xlabel("epoch")
        plt.ylabel("best value")
        fig.savefig('best_value.png')  # save the figure to file
        plt.close(fig)

    def open_input_dialog_event(self, error):
        customtkinter.CTkInputDialog(text=error, title="Error occurred")

    def change_selection_event(self, value: str):
        self.configuration.set_selection_type(value)
        if value == SelectionType["TOURNAMENT"] or value == SelectionType["BEST"]:
            self.tournament_size_entry.configure(state="normal")
        else:
            self.tournament_size_entry.configure(state="disabled")

    def change_elite_strategy_event(self, value: str):
        self.configuration.set_elite_group_type(value)

    def change_crossing_type_event(self, value: str):
        self.configuration.set_crossing_type(value)

    def change_mutation_type_event(self, value: str):
        self.configuration.set_mutation_type(value)


if __name__ == "__main__":
    app = App()
    app.mainloop()
