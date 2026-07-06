from importlib.metadata import distributions

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import math
import json

def load_data():

    extinction = np.load("numpy-files/extinction.npy").round(2)  
    sample_names = np.load("numpy-files/sample_name.npy")
    wavelengths = np.load("numpy-files/wavelength.npy").round(2)  
    return extinction, sample_names, wavelengths

def compute_sample_mean(extinction, sample_names):

    unique_samples = np.unique(sample_names)
    row_str = unique_samples.astype(str).reshape(-1, 1)
    
    mean_values = np.array([np.mean(extinction[sample_names == sample], axis=0) for sample in unique_samples]).round(2)
    combined_mean = np.hstack([row_str, mean_values])    
    
    return unique_samples, mean_values, combined_mean

def compute_probability(extinction, max_output_size):

    wavelengths = list(max_output_size.keys())
    num_wavelengths = len(wavelengths)
    total = extinction.shape[0]  # Total number of samples

    distributions = defaultdict(lambda: defaultdict(int))

    for i in range(num_wavelengths):
        values = extinction[:, i].round(2)  

        unique, counts = np.unique(values, return_counts=True, sorted=False)
        for unique_val, cnt in zip(unique, counts):
            distributions[wavelengths[i]][unique_val] = cnt / total                     # Probability calculation

    return distributions

def compute_count_matrix(extinction, max_output_size):

    wavelengths = list(max_output_size.keys())
    num_wavelengths = len(wavelengths)
    frequency_matrix = np.empty(extinction.shape, dtype=int)

    for i in range(num_wavelengths):
        values = extinction[:, i].round(2)  
        unique, counts = np.unique(values, return_counts=True, sorted=False)
        count_dict = dict(zip(unique, counts))
        frequency_matrix[:, i] = [count_dict.get(val, 0) for val in values]  

    return frequency_matrix


def build_probability_matrix(data, probabilities):
    prob_matrix = np.empty(data.shape)
    wavelengths = list(probabilities.keys())

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            wl = wavelengths[j]
            value = data[i, j]
            prob_matrix[i, j] = probabilities[wl].get(value, 0)                         # Default 0 if value not found
    
    return prob_matrix

def build_entropy_matrix(data, probabilities):
    entropy_matrix = np.empty(data.shape)
    wavelengths = list(probabilities.keys())

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            wl = wavelengths[j]
            value = data[i, j]
            prob = probabilities[wl].get(value, 1e-10)
            prob = max(prob, 1e-10)  
            entropy_matrix[i, j] = -prob * math.log2(prob)                              # Entropy calculation
    
    return entropy_matrix

def calculate_entropy_per_sample(entropy_matrix,sample_names):
    entropy_per_sample = np.sum(entropy_matrix, axis=1).reshape(-1, 1)
    row_str = sample_names.astype(str).reshape(-1, 1)
    combined_result = np.hstack([row_str, entropy_per_sample])

    return combined_result

def calculate_max_entropy(max_output_per_wavelength):
    
    max_entropy = 0
    for size in max_output_per_wavelength.values():
        max_entropy += math.log2(size)
    return max_entropy

def get_min_max_per_wavelength(extinction, wavelengths):

    num_wavelengths = len(wavelengths)
    min_max_dict = {}

    for i in range(num_wavelengths):
        wl = wavelengths[i]
        values = extinction[:, i]
        min_val = np.min(values)
        max_val = np.max(values)
        min_max_dict[float(wl)] = {"min": float(min_val), "max": float(max_val)}

    return min_max_dict

def data_to_dataframe(extinction, sample_names, wavelengths): 
    return pd.DataFrame(extinction, index=sample_names, columns=wavelengths)

def combine_arrays(data,row,column):
    data_str = data.astype(float)
    row_str = row.astype(str).reshape(-1, 1)
    column_str = column.astype(float).reshape(1, -1)
    corner = np.array([['']])

    return np.block([[corner, column_str], [row_str, data_str]])

def plot_distributions(wavelengths, distributions):

    num_wavelengths = len(wavelengths)
    fig, axes = plt.subplots(num_wavelengths, 1, figsize=(10, 4 * num_wavelengths))
    if num_wavelengths == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        wl = wavelengths[i]
        dist = distributions[i]
        values, probs = zip(*sorted(dist.items()))
        ax.bar(values, probs)
        ax.set_title(f"Wavelength: {wl:.2f} nm")
        ax.set_xlabel("Extinction Value")
        ax.set_ylabel("Probability")

    plt.tight_layout()
    plt.savefig("probability_distributions.png")
    plt.show()

def plot_entropy_per_sample(entropy_per_sample):
    sample_names = entropy_per_sample[:, 0]
    entropy_values = entropy_per_sample[:, 1].astype(float)

    plt.figure(figsize=(12, 6))
    plt.bar(sample_names, entropy_values)
    plt.title("Entropy per Sample")
    plt.xlabel("Sample Name")
    plt.ylabel("Entropy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("entropy_per_sample.png")
    plt.show()

def calculate_max_output_size_per_wavelength(data, wavelengths):
    max_output_size = defaultdict(int)

    for i in range(data.shape[1]):
        unique_values = np.unique(data[:, i])
        max_output_size[wavelengths[i]] = len(unique_values)

    return max_output_size
        
def calculate_total_output_size(max_output_size):
    total_output_size = 1
    for size in max_output_size.values():
        total_output_size *= size
        
    return total_output_size

def is_greater_than_SHA256_limit(total_output_size):
    SHA256_limit = 2**256
    return total_output_size > SHA256_limit

def save_numpy_arrays_to_csv(data_array, sample_names, wavelengths, output_filename):
    data_array_df = pd.DataFrame(data_array, index=sample_names, columns=wavelengths)
    data_array_df.to_csv(output_filename, index=True)

def save_dict_to_json(data_dict, output_filename):
    with open(output_filename, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

def save_output_to_txt(data, output_filename):
    with open(output_filename, 'w') as f:
        for item in data:
            f.write(f"{item}\n")