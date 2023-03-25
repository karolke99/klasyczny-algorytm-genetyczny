import tkinter
import tkinter.messagebox

import customtkinter

from configuration import Configuration
from util import EliteStrategyType, SelectionType, CrossingType, MutationType

customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    configuration = Configuration()

    def __init__(self):
        super().__init__()

        self.title("Algorytm genetyczny")
        self.geometry(f"{750}x{580}")

        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1)

        # population
        self.evolution_frame = customtkinter.CTkFrame(self)
        self.evolution_frame.grid(row=0, column=0, sticky="nesw")

        self.population_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Population")
        self.population_label.grid(row=0, column=1, columnspan=2, padx=10, pady=10)

        self.population_size_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Population size:")
        self.population_size_label.grid(row=1, column=1, padx=10, pady=10)
        self.population_size_var = tkinter.StringVar(value='0')
        self.population_size_entry = customtkinter.CTkEntry(self.evolution_frame,
                                                            textvariable=self.population_size_var)
        self.population_size_entry.grid(row=1, column=2, padx=10, pady=10)

        self.accuracy_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Accuracy:")
        self.accuracy_label.grid(row=2, column=1, padx=10, pady=10)
        self.accuracy_var = tkinter.StringVar(value='0')
        self.accuracy_entry = customtkinter.CTkEntry(self.evolution_frame, textvariable=self.accuracy_var)
        self.accuracy_entry.grid(row=2, column=2, padx=10, pady=10)

        # Epochs
        self.epochs_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Epochs")
        self.epochs_label.grid(row=3, column=1, columnspan=2, padx=10, pady=20)

        self.epochs_number_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Epoch number:")
        self.epochs_number_label.grid(row=4, column=1, padx=10, pady=10)
        self.epochs_number_var = tkinter.StringVar(value='0')
        self.epochs_entry = customtkinter.CTkEntry(self.evolution_frame, textvariable=self.epochs_number_var)
        self.epochs_entry.grid(row=4, column=2, padx=20, pady=10)

        # selection
        self.selection_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Selection")
        self.selection_label.grid(row=5, column=1, columnspan=3, padx=10, pady=20)

        self.selection_type_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Selection type:")
        self.selection_type_label.grid(row=6, column=1, padx=10, pady=10)
        self.selection_dropdown = customtkinter.CTkOptionMenu(self.evolution_frame,
                                                              values=list(SelectionType.values()),
                                                              command=self.change_selection_event)
        self.selection_dropdown.grid(row=6, column=2, pady=10, padx=20)

        self.tournament_size = customtkinter.CTkLabel(master=self.evolution_frame, text="Tournament size:")
        self.tournament_size.grid(row=7, column=1, padx=10, pady=10)
        self.tournament_size_var = tkinter.StringVar(value='0')
        self.tournament_size_entry = customtkinter.CTkEntry(self.evolution_frame, textvariable=self.tournament_size_var)
        self.tournament_size_entry.grid(row=7, column=2, padx=20, pady=10)

        # elite strategy
        self.elite_label = customtkinter.CTkLabel(master=self.evolution_frame, text="Elite strategy")
        self.elite_label.grid(row=8, column=1, columnspan=2, padx=10, pady=20)

        self.elite_group_size_var = tkinter.StringVar(value='0')
        self.elite_group_size_entry = customtkinter.CTkEntry(self.evolution_frame,
                                                             textvariable=self.elite_group_size_var)
        self.elite_group_size_entry.grid(row=9, column=1, padx=20, pady=10)

        self.elite_dropdown = customtkinter.CTkOptionMenu(self.evolution_frame,
                                                          values=list(EliteStrategyType.values()),
                                                          command=self.change_elite_strategy_event)
        self.elite_dropdown.grid(row=9, column=2, pady=10, padx=20)

        # crossing
        self.genetic_operators_frame = customtkinter.CTkFrame(self)
        self.genetic_operators_frame.grid(row=0, column=1, rowspan=3, sticky="nsew")
        self.Crossing_label = customtkinter.CTkLabel(master=self.genetic_operators_frame, text="Crossing")
        self.Crossing_label.grid(row=0, column=1, columnspan=2, padx=10, pady=10)

        self.Crossing_type_label = customtkinter.CTkLabel(master=self.genetic_operators_frame, text="Crossing type:")
        self.Crossing_type_label.grid(row=1, column=1, padx=10, pady=10)
        self.crossing_dropdown = customtkinter.CTkOptionMenu(self.genetic_operators_frame,
                                                             values=list(CrossingType.values()),
                                                             command=self.change_crossing_type_event)
        self.crossing_dropdown.grid(row=1, column=2, pady=10, padx=20, sticky="n")

        self.crossing_probability_var = tkinter.StringVar(value='0')
        self.crossing_probability_label = customtkinter.CTkLabel(master=self.genetic_operators_frame,
                                                                 text="Probability:")
        self.crossing_probability_label.grid(row=2, column=1, padx=10, pady=10)
        self.crossing_probability_entry = customtkinter.CTkEntry(self.genetic_operators_frame,
                                                                 textvariable=self.crossing_probability_var)
        self.crossing_probability_entry.grid(row=2, column=2, padx=20, pady=10, sticky="nsew")

        self.mutating_label = customtkinter.CTkLabel(master=self.genetic_operators_frame, text="Mutating")
        self.mutating_label.grid(row=3, column=1, columnspan=2, padx=10, pady=10)

        self.mutation_type_label = customtkinter.CTkLabel(master=self.genetic_operators_frame, text="Mutating type:")
        self.mutation_type_label.grid(row=4, column=1, padx=10, pady=10)
        self.mutation_type_dropdown = customtkinter.CTkOptionMenu(self.genetic_operators_frame,
                                                                  values=list(MutationType.values()),
                                                                  command=self.change_mutation_type_event)
        self.mutation_type_dropdown.grid(row=4, column=2, pady=10, padx=20, sticky="n")

        self.mutation_probability_var = tkinter.StringVar(value='0')
        self.mutation_probability_label = customtkinter.CTkLabel(master=self.genetic_operators_frame,
                                                                 text="Probability:")
        self.mutation_probability_label.grid(row=5, column=1, padx=10, pady=10)
        self.mutation_probability_entry = customtkinter.CTkEntry(self.genetic_operators_frame,
                                                                 textvariable=self.mutation_probability_var)
        self.mutation_probability_entry.grid(row=5, column=2, padx=20, pady=10, sticky="nsew")

        self.inversion_label = customtkinter.CTkLabel(master=self.genetic_operators_frame, text="Inversion")
        self.inversion_label.grid(row=6, column=1, columnspan=2, padx=10, pady=10)
        self.inversion_probability_var = tkinter.StringVar(value='0')
        self.inversion_probability_label = customtkinter.CTkLabel(master=self.genetic_operators_frame,
                                                                  text="Probability:")
        self.inversion_probability_label.grid(row=7, column=1, padx=10, pady=10)
        self.inversion_probability_entry = customtkinter.CTkEntry(self.genetic_operators_frame,
                                                                  textvariable=self.inversion_probability_var)
        self.inversion_probability_entry.grid(row=7, column=2, padx=20, pady=10, sticky="nsew")

        # results
        self.start_button = customtkinter.CTkButton(master=self.genetic_operators_frame, fg_color="transparent",
                                                    border_width=2, text_color=("gray10", "#DCE4EE"), text="Start",
                                                    command=self.start)
        self.start_button.grid(row=8, column=1, padx=20, pady=(25, 10))

        self.plot_button = customtkinter.CTkButton(master=self.genetic_operators_frame, fg_color="transparent",
                                                   border_width=2, text_color=("gray10", "#DCE4EE"),
                                                   text="Generate plots", command=self.generate_plots)
        self.plot_button.grid(row=8, column=2, padx=20, pady=(25, 10))

        self.result_label = customtkinter.CTkLabel(master=self.genetic_operators_frame, text="X:")
        self.result_label.grid(row=9, column=1, padx=20, pady=10)
        self.result_textbox = customtkinter.CTkTextbox(self.genetic_operators_frame, width=150, height=20)
        self.result_textbox.grid(row=9, column=2, padx=(0, 0), pady=(20, 0))

        self.fx_label = customtkinter.CTkLabel(master=self.genetic_operators_frame, text="F(x):")
        self.fx_label.grid(row=10, column=1, padx=20, pady=10)
        self.fx_textbox = customtkinter.CTkTextbox(self.genetic_operators_frame, width=150, height=20)
        self.fx_textbox.grid(row=10, column=2, pady=(20, 0))

        # defaults
        self.tournament_size_entry.configure(state="readonly")
        self.fx_textbox.configure(state="disabled")
        self.result_textbox.configure(state="disabled")

        self.configuration.set_selection_type(SelectionType['ROULETTE'])
        self.configuration.set_elite_group_type(EliteStrategyType['PERCENT'])
        self.configuration.set_crossing_type(CrossingType['SINGLE_POINT'])
        self.configuration.set_mutation_type(MutationType['SINGLE_POINT'])

    def start(self):
        try:
            self.configuration.set_population_size(self.population_size_var)
            self.configuration.set_epoch(self.epochs_number_var)
            self.configuration.set_accuracy(self.accuracy_var)
            self.configuration.set_tournament_size(self.tournament_size_var)
            self.configuration.set_elite_group_size(self.elite_group_size_var)
            self.configuration.set_mutation_probability(self.mutation_probability_var)
            self.configuration.set_crossing_probability(self.crossing_probability_var)
            self.configuration.set_inversion_probability(self.inversion_probability_var)

            x, fx = self.run_algorithm()

            self.fx_textbox.configure(state="normal")
            self.result_textbox.configure(state="normal")

            self.result_textbox.insert("0.0", str(x))
            self.fx_textbox.insert("0.0", str(fx))

            self.fx_textbox.configure(state="disabled")
            self.result_textbox.configure(state="disabled")
        except Exception as e:
            self.open_input_dialog_event(e)

    #     todo
    # run algorithm here
    def run_algorithm(self):
        return 0.2637483910, 0.123873463

    # todo
    def generate_plots(self):
        pass

    def open_input_dialog_event(self, error):
        customtkinter.CTkInputDialog(text=error, title="Error occurred")

    def change_selection_event(self, value: str):
        self.configuration.set_selection_type(value)
        if value == SelectionType["TOURNAMENT"]:
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
