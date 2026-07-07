from entropyops import *
import os
import sys

input_configuration = sys.argv[1] if len(sys.argv) > 1 else exit(1)
output_folder = sys.argv[2] if len(sys.argv) > 2 else "calculation_entropy_average"

input_configuration += "/"
input_configuration_path = input_configuration + "numpy-files"
output_folder = input_configuration  +  output_folder

os.mkdir(output_folder)
os.chmod(output_folder, 0o777)
print(f"\nCreated Output Folder: {output_folder}")

extinction, sample_names, wavelengths = load_data(input_configuration_path + "/extinction.npy",input_configuration_path + "/sample_name.npy", input_configuration_path + "/wavelength.npy")

unique_samples, mean_values, combined_mean = compute_sample_mean(extinction, sample_names)
save_numpy_arrays_to_csv(mean_values, unique_samples, wavelengths, f"{output_folder}/mean_values.csv")
print("\nExtinction data (Average of same sample runs) saved to CSV.")

max_output_size = calculate_max_output_size_per_wavelength(mean_values,wavelengths)
save_dict_to_json(max_output_size, f"{output_folder}/max_output_size.json")
print("\nUnique values per wavelength created")

frequency_lookup = wavelength_value_frequency_lookup_matrix(mean_values, max_output_size)
save_numpy_arrays_to_csv(frequency_lookup, unique_samples, wavelengths, f"{output_folder}/frequency_matrix.csv")
print("\nFrequency Lookup Matrix of unique values per wavelength created (Every value has been replaced by its occurence)")

distributions = compute_probability(mean_values,max_output_size)
save_dict_to_json(distributions, f"{output_folder}/probability_distributions.json")
print("\nProbability computed for each wavelength each component.")
    
prob_matrix = build_probability_matrix(mean_values, distributions)     
save_numpy_arrays_to_csv(prob_matrix, unique_samples, wavelengths, f"{output_folder}/probability_matrix.csv")   
print("\nProbability Matrix created")                                    

entropy_matrix = build_entropy_matrix(mean_values, distributions)
save_numpy_arrays_to_csv(entropy_matrix, unique_samples, wavelengths, f"{output_folder}/entropy_matrix.csv")
print("\nEntropy Matrix created")

information_per_sample, average_information = calculate_information_content_per_sample(mean_values, distributions, unique_samples)
save_output_to_txt(information_per_sample, f"{output_folder}/information_per_sample.txt")
print("\nInformation per Sample saved to TXT file.")

entropy_per_wavelength, average_entropy_per_wv, total_average_entropy = calculate_entropy_per_wavelength(distributions)
save_output_to_txt(entropy_per_wavelength, f"{output_folder}/entropy_per_wavelength.txt")
print("\nEntropy per Wavelength saved to TXT file.")

plot_entropy_per_sample(information_per_sample, f"{output_folder}/information_per_sample.png")
print("\nAverage information per sample plot saved")

print("\n********************************************************")

print("\nMax Entropy:", calculate_max_entropy(max_output_size))
print("\nSystem Entropy:", total_average_entropy)
print("\nAverage Entropy per Wavelength:", average_entropy_per_wv)
print("\nAverage Information per Sample:", average_information)
print("\nInput Entropy:", calculate_input_entropy(sample_names))
print("\nSystem Entropy Efficiency:", system_entropy_efficiency(total_average_entropy, calculate_max_entropy(max_output_size)))

print("\n****************************************************************")


