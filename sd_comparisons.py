import math
from collections import defaultdict
import numpy as np
import json
import itertools
import matplotlib.pyplot as plt
import sys
from entropyops import *

def calculate_average_std_dev_measurements(extinction, sample_names):

    unique_samples = np.unique(sample_names)
    n_wavelengths = extinction.shape[1]

    std_dev_matrix = np.zeros((len(unique_samples), n_wavelengths))
    #std_per_sample = defaultdict(lambda :defaultdict(int))
    group_size = 10

    for i, sample in enumerate(unique_samples):
        
        rows = extinction[sample_names == sample]
        groups = math.ceil(rows.shape[0] / group_size)
        measurement_avgs = np.zeros((groups,n_wavelengths))

        for group in range(groups):
        
            std_per_wl_per_group = np.std(rows[group * group_size : (group + 1) * group_size], axis=0, ddof=1)
            measurement_avgs[group, :] = std_per_wl_per_group

        std_dev_matrix[i: ] = measurement_avgs.mean(axis = 0).round(3)   
        del measurement_avgs
    
    avg_std_dev_per_wv = std_dev_matrix.mean(axis=0).round(3)
    return avg_std_dev_per_wv, std_dev_matrix


def calculate_average_std_dev_printing(extinction, sample_names):

    unique_samples = np.unique(sample_names)
    n_wavelengths = extinction.shape[1]

    std_dev_matrix = np.zeros((len(unique_samples), n_wavelengths))
    #std_per_sample = defaultdict(lambda :defaultdict(int))
    group_size = 10

    for i, sample in enumerate(unique_samples):
        
        rows = extinction[sample_names == sample]
        groups = math.ceil(rows.shape[0] / group_size)
        printing_avgs = np.zeros((groups,n_wavelengths))

        for group in range(groups):
            printing_avgs[group : ] = rows[group * group_size : (group + 1) * group_size].mean(axis = 0)

        std_per_wl_per_sample = np.std(printing_avgs, axis=0, ddof=1)
        std_dev_matrix[i : ] = std_per_wl_per_sample
            
    avg_std_dev_per_wv = std_dev_matrix.mean(axis=0).round(3)
    return avg_std_dev_per_wv, std_dev_matrix


def get_optimum_sd(sd1,sd2,sd3):

    return np.maximum(np.maximum(sd1,sd2),sd3)

def plot_SD (SD1,SD2,SD3,WVs):
    plt.plot(WVs, SD1, color='r', label='Combined SD')
    plt.plot(WVs, SD2, color='g', label='Measurement SD')
    plt.plot(WVs, SD3, color='b', label='Printing SD')

    #plt.xticks(WVs)
    #plt.grid(True, which="both")
    plt.title("Combined SD, Measurement SD, Printing SD vs Wavelength")
    plt.xlabel("Wavelength components")
    plt.ylabel("Standard Deviation")
    plt.xticks(rotation=45)
    plt.legend()

    plt.show()

noise_factor = 1

config = sys.argv[1] if len(sys.argv) > 1 else exit(1)
noise_factor = int(sys.argv[2] if len(sys.argv) > 2 else 1)

extinction_path = config + "/numpy-files/extinction.npy"
sample_names_path = config + "/numpy-files/sample_name.npy"
wavelengths_path = config + "/numpy-files/wavelength.npy"

extinction, sample_names, wavelengths = load_data(extinction_path, sample_names_path, wavelengths_path)

sd_1 = calculate_average_std_dev(extinction,sample_names)
sd1_per_wv = sd_1[0]
sd1_matrix = sd_1[1]

sd_2_measurements = calculate_average_std_dev_measurements(extinction, sample_names)
sd2_per_wv = sd_2_measurements[0]
sd2_matrix = sd_2_measurements[1]

sd_3_measurements = calculate_average_std_dev_printing(extinction, sample_names)
sd3_per_wv = sd_3_measurements[0]
sd3_matrix = sd_3_measurements[1]

print(f"Standard Deviation 1 (Combined) = {sd1_per_wv}")
print(f"Standard Deviation 2 (Measurements) = {sd2_per_wv}")
print(f"Standard Deviation 3 (Printing) = {sd3_per_wv}")

print(f"Optimum sd: {get_optimum_sd(sd1_per_wv,sd2_per_wv,sd3_per_wv)}")
plot_series(
    wavelengths,
    (sd1_per_wv, "Combined SD", "m"),
    (sd2_per_wv, "Measurement SD", "g"),
    (sd3_per_wv, "Printing SD", "b"),
    title="Standard Deviation vs Wavelength",
    ylabel="Standard Deviation",
)