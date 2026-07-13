from entropyops_bins import *
import pandas as pd
import sys
import os
import shutil


config = sys.argv[1] if len(sys.argv) > 1 else exit(1)

extinction_path = config + "/numpy-files/extinction.npy"
sample_names_path = config + "/numpy-files/sample_name.npy"
wavelengths_path = config + "/numpy-files/wavelength.npy"

output_path = f"{config}/calculation_binning_output"

if os.path.isdir(output_path):
    shutil.rmtree(output_path)
    print(f"Clearing previous cache....")

os.mkdir(output_path)
print(f"\nOutput folder created in {config} ")

extinction, sample_names, wavelengths = load_data(extinction_path, sample_names_path, wavelengths_path)

min_max_array = calculate_wavelength_min_max(extinction, wavelengths)
min_max_array_df = pd.DataFrame(min_max_array)
min_max_array_df.to_csv(f"{output_path}/min_max_array.csv",index=False,header=None)
print(f"\nMin max per wavelength file created")

avg_std_dev_per_wv, std_matrix = calculate_average_std_dev(extinction, sample_names)

avg_std_dev_per_wv_df = pd.DataFrame(avg_std_dev_per_wv.reshape(1,-1), columns= wavelengths)
avg_std_dev_per_wv_df.to_csv(f"{output_path}/avg_std_dev_per_wv.csv",index=False)
print(f"\nAverage Standard Dev per wavelength file created")

#print(std_per_sample)
std_matrix_df = pd.DataFrame(std_matrix, index= np.unique(sample_names), columns= wavelengths)
std_matrix_df.to_csv(f"{output_path}/avg_std_dev_per_wv_per_sample.csv")
print(f"\nAverage Standard Dev per wavelength per sample file created")

bin_edges_list, num_bins_per_wv = calculate_bins_per_wavelength(min_max_array, avg_std_dev_per_wv)
num_bins_per_wv_df = pd.DataFrame(num_bins_per_wv.reshape(1,-1), columns=wavelengths)
num_bins_per_wv_df.to_csv(f"{output_path}/total_bins_per_wv.csv",index=False)
print(f"\nTotal Bins per wavelength file created")

save_dict_to_json(bin_edges_list, f"{output_path}/bin_edges_list_per_wv.json")
print(f"\nBins edges per wavelength JSON File created")

bin_matrix = assign_bins_per_wavelength(extinction, bin_edges_list)
bin_matrix_df = pd.DataFrame(bin_matrix, index= sample_names, columns= wavelengths)
bin_matrix_df.to_csv(f"{output_path}/Binned_matrix.csv")
print(f"\nBin Matrix created")

binned_distributions = compute_binned_probability(bin_matrix, wavelengths)
save_dict_to_json(binned_distributions,f"{output_path}/bin_probability_distributions.json")
print(f"\nPer Bin probability distribution file created")

entropy_per_wl, avg_entropy, total_entropy = compute_binned_entropy(binned_distributions, wavelengths)
save_dict_to_json(entropy_per_wl, f"{output_path}/entropy_per_wavelength.json")
print(f"\nEntropy per wavelength file created")

max_entropy = compute_binned_max_entropy(num_bins_per_wv)

info_per_sample, avg_info = compute_binned_information_content_per_sample(bin_matrix, binned_distributions, wavelengths, sample_names)
save_output_to_txt(info_per_sample, f"{output_path}/info_per_sample.txt")
print("\nInformation per Sample File created")

efficiency = system_entropy_efficiency(total_entropy, max_entropy)

input_entropy = calculate_input_entropy(sample_names)

print("-" * 30)

print(f"Average Entropy per wavelength : {avg_entropy}")
print(f"Total Entropy of system : {total_entropy}")
print(f"Max Entropy possible : {max_entropy}")
print(f"Average Information per Sample : {avg_info}")
print(f"System Entropy efficiency : {efficiency}")
print(f"Input Entropy : {input_entropy}")