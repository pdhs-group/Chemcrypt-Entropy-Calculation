from entropyops import *
from experiment import ExperimentConfiguration
from quantization import BinnedExperiment
import numpy as np
import pandas as pd
import sys
import os
import shutil
from utils import *

noise_factor = 1

config = sys.argv[1] if len(sys.argv) > 1 else exit(1)
noise_factor = int(sys.argv[2] if len(sys.argv) > 2 else 1)

output_path = f"{config}/calculation_binning_output"

if os.path.isdir(output_path):
    shutil.rmtree(output_path)
    print(f"Clearing previous cache....")

os.mkdir(output_path)
print(f"\nOutput folder created in {config} ")

experiment = ExperimentConfiguration.from_directory(config)
binned = BinnedExperiment.create(experiment, noise_factor)
extinction = experiment.data
sample_names = experiment.sample_names
wavelengths = experiment.wavelengths
min_max_array = calculate_wavelength_min_max(extinction, wavelengths)
min_max_array_df = pd.DataFrame(min_max_array)
min_max_array_df.to_csv(f"{output_path}/min_max_array.csv",index=False,header=None)
print(f"\nMin max per wavelength - file created")

avg_std_dev_per_wv, std_matrix = experiment.standard_deviation_per_config()

avg_std_dev_per_wv_df = pd.DataFrame(avg_std_dev_per_wv.reshape(1,-1), columns= wavelengths)
avg_std_dev_per_wv_df.to_csv(f"{output_path}/avg_std_dev_per_wv.csv",index=False)
print(f"\nAverage Standard Dev per wavelength - file created")

#print(std_per_sample)
std_matrix_df = pd.DataFrame(std_matrix, index= np.unique(sample_names), columns= wavelengths)
std_matrix_df.to_csv(f"{output_path}/avg_std_dev_per_wv_per_sample.csv")
print(f"\nAverage Standard Dev per wavelength per sample - file created")

bin_edges_list = binned.bin_edges
num_bins_per_wv = binned.bins_per_wavelength
num_bins_per_wv_df = pd.DataFrame(num_bins_per_wv.reshape(1,-1), columns=wavelengths)
num_bins_per_wv_df.to_csv(f"{output_path}/total_bins_per_wv.csv",index=False)
print(f"\nTotal Bins per wavelength - file created")

save_dict_to_json(bin_edges_list, f"{output_path}/bin_edges_list_per_wv.json")
print(f"\nBins edges per wavelength - JSON File created")

bin_matrix = binned.values
bin_matrix_df = pd.DataFrame(bin_matrix, index= sample_names, columns= wavelengths)
bin_matrix_df.to_csv(f"{output_path}/Binned_matrix.csv")
print(f"\nBin Matrix created")


binned_distributions = binned.distributions
save_dict_to_json(binned_distributions,f"{output_path}/bin_probability_distributions.json")
print(f"\nPer Bin probability distribution - file created")

entropy_per_wl, avg_entropy, total_entropy = binned.entropy
save_dict_to_json(entropy_per_wl, f"{output_path}/entropy_per_wavelength.json")
print(f"\nEntropy per wavelength - file created")


cond_entropy, avg_cond_entropy = compute_conditional_binned_entropy(bin_matrix,sample_names)
save_dict_to_json(cond_entropy,f"{output_path}/conditional_entropy_per_sample.json")
print(f"\nConditional Entropy per Sample - file created")

max_entropy = compute_binned_max_entropy(num_bins_per_wv)

info_per_sample, avg_info = compute_binned_information_content_per_sample(bin_matrix, binned_distributions, wavelengths, sample_names)
save_output_to_txt(info_per_sample, f"{output_path}/info_per_sample.txt")
print("\nInformation per Sample - File created")

wv_entropy_performance = compute_wv_entropy_diff_from_max(binned_distributions, wavelengths)
save_dict_to_json(wv_entropy_performance,f"{output_path}/entropy_diff_from_max.json")
print(f"\nEntropy difference from Max - file created")

plot_json_dict_data(wv_entropy_performance, f"{output_path}/Entropy_diff_from_max")
print("Entropy difference Path plot created")

markov_entropy_per_wavelength, markov_total_entropy = estimate_markov_conditional_entropy(bin_matrix)
save_dict_to_json(
    dict(zip(wavelengths[1:].astype(float), markov_entropy_per_wavelength[1:])),
    f"{output_path}/markov_conditional_entropy.json",
)
save_dict_to_json(
    estimate_adjacent_mutual_information(bin_matrix, wavelengths),
    f"{output_path}/adjacent_mutual_information.json",
)
print("Markov conditional entropy and adjacent mutual information - files created")

efficiency_independent = system_entropy_efficiency(total_entropy, max_entropy)
efficiency_conditional = system_entropy_efficiency(avg_cond_entropy,total_entropy)

input_entropy = calculate_input_entropy(sample_names)

print("-" * 30)

print(f"Average Conditional Entropy per Sample : {avg_cond_entropy}")
print(f"Total Possible Entropy of system : {total_entropy}")
print(f"Max Entropy possible : {max_entropy}")
print(f"Average Entropy per wavelength : {avg_entropy}")
print(f"Average Information per Sample : {avg_info}")
print(f"System Entropy efficiency for Independent Wavelengths : {efficiency_independent}")
print(f"System Entropy efficiency for Continuous Dependent system : {efficiency_conditional}")
print(f"Input Entropy : {input_entropy}")
print(f"Markov conditional entropy : {markov_total_entropy}")

print(markov_entropy_per_wavelength)

plot_series(experiment.wavelengths, (list(x[1] for x in entropy_per_wl),f"Independent Entropy","b"),(markov_entropy_per_wavelength, f"1st order Markov Entropy", "r"), title=f"Satndard vs Conditional Entropy per wv", xlabel=f"Wavelengths", ylabel=f"Entropy value",save_path=f"{output_path}/Ind vs Cond entropy.png")

"""
print("-" * 30)
print(compute_binned_collision_rate_exclude_same_config(bin_matrix,sample_names))

print(list(item[1] for item in entropy_per_wl))
print("------------------------")
print(list(markov_entropy_per_wavelength))
"""