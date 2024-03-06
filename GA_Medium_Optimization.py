import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

import pygad
import pygad.gann
import pygad.nn
import csv



filename = "Dummy_Optimization_Data.txt"

# Read the text file and store data in a list
data_list = []
with open(filename, newline='') as txtfile:
    txt_reader = csv.reader(txtfile, delimiter=',')
    for row in txt_reader:
        data_list.append(row)

# Convert the list of lists into a NumPy array (matrix)
data_matrix = np.array(data_list[1:])

# Define column names based on your data
column_names = ['Water Concentration (%)', 'Glucose Concentration (g/L)', 'CO2 Concentration (%)', 'Heat (°C)', 'pH', 'Growth Rate (OD600)']

# Create a Pandas DataFrame
data = pd.DataFrame(data=data_matrix, columns=column_names)

input_data = data.drop('Growth Rate (OD600)', axis=1)
target_data = data['Growth Rate (OD600)']

# Standardize input data using scikit-learn
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)
target_data = target_data.values

GANN_instance = pygad.gann.GANN(num_solutions= 6, # this is the number of models in the initial population
                                num_neurons_input= 5, # this is the number of features in the medium
                                num_neurons_hidden_layers=[16], # each index is each hidden layer's # of neurons
                                num_neurons_output=1, # output is the growth factor
                                hidden_activations=["relu"],
                                output_activation="None")

population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)

def fitness_func(ga_instance, solution, sol_idx):
    global GANN_instance, input_data_scaled, target_data

    predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx], data_inputs=input_data_scaled, problem_type="regression")

    solution_fitness = 1/mean_squared_error(target_data, predictions)

    return solution_fitness

def callback_generation(ga_instance):
    global GANN_instance

    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks, population_vectors=ga_instance.population)
    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")

initial_population = population_vectors.copy()

ga_instance = pygad.GA(num_generations= 500,  # number of times to run GA algorithim
                       num_parents_mating= 4, # number of models selected from population to exchange genes for next gen
                       initial_population=initial_population, # vector of initial population growth factors
                       fitness_func=fitness_func,
                       mutation_percent_genes= 5, # edit to have more or less crossover
                       init_range_low= -1, #idk what this does
                       init_range_high= 1, #idk what this does
                       parent_selection_type= "rws", # selects based on growth factor (higher % chance of selecting higher growth factor model)
                       crossover_type= "uniform",
                       mutation_type= "random",
                       keep_parents= 2,
                       on_generation=callback_generation)

ga_instance.run()

ga_instance.plot_fitness()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

# This predicts the growth factor of a given medium (in data inputs)

predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[solution_idx], data_inputs=input_data_scaled)
print(f"MSE of the trained network : {mean_squared_error(target_data, predictions)}")