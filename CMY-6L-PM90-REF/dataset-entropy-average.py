from entropyops import *
import os
import sys

output_folder = sys.argv[1] if len(sys.argv) > 1 else "output_entropy_average"

os.mkdir(output_folder)
os.chmod(output_folder, 0o777)
print(f"Created Output Folder: {output_folder}")

extinction, sample_names, wavelengths = load_data()

unique_samples, mean_values, combined_mean = compute_sample_mean(extinction, sample_names)
save_numpy_arrays_to_csv(mean_values, unique_samples, wavelengths, f"{output_folder}/mean_values.csv")
#combined_array = combine_arrays(mean_values, unique_samples, wavelengths)
#print("\nCombined Mean Values Array:")
#print(combined_array)


max_output_size = calculate_max_output_size_per_wavelength(mean_values,wavelengths)
save_dict_to_json(max_output_size, f"{output_folder}/max_output_size.json")
print("\nUnique values per wavelength created")
#print(max_output_size := max_output_size)
#print(total_output_size := calculate_total_output_size(max_output_size))
#print("Max entropy per sample:", calculate_max_entropy_per_sample(max_output_size))

frequency = compute_count_matrix(mean_values, max_output_size)
save_numpy_arrays_to_csv(frequency, unique_samples, wavelengths, f"{output_folder}/frequency_matrix.csv")
print("\nFrequency Matrix of unique values per wavelength created")

distributions = compute_probability(mean_values,max_output_size)
save_dict_to_json(distributions, f"{output_folder}/probability_distributions.json")
print("Probability computed for each wavelength each component.")
#print("Probability computed for each wavelength each component.")
#for wv, dist in list(distributions.items()):
    #print(f"Wavelength {wv} nm:", dict(list(dist.items())))
    
prob_matrix = build_probability_matrix(mean_values, distributions)     
save_numpy_arrays_to_csv(prob_matrix, unique_samples, wavelengths, f"{output_folder}/probability_matrix.csv")   
print("\nProbability Matrix created")                                    
#combined_prob_matrix = combine_arrays(prob_matrix, unique_samples, wavelengths)
#print("\nCombined Probability Matrix:")
#print(combined_prob_matrix)

entropy_matrix = build_entropy_matrix(mean_values, distributions)
save_numpy_arrays_to_csv(entropy_matrix, unique_samples, wavelengths, f"{output_folder}/entropy_matrix.csv")
print("\nEntropy Matrix created")
#combined_entropy_matrix = combine_arrays(entropy_matrix, unique_samples, wavelengths)
#print("\nCombined Entropy Matrix:")
#print(combined_entropy_matrix)

entropy_per_sample = calculate_entropy_per_sample(entropy_matrix, unique_samples)
save_output_to_txt(entropy_per_sample, f"{output_folder}/entropy_per_sample.txt")
print("\nEntropy per Sample saved to TXT file.")
#print("\nEntropy per Sample:")
#print(entropy_per_sample)

print("\nMax Entropy:", calculate_max_entropy(max_output_size))

#plot_entropy_per_sample(entropy_per_sample)

#print("\nMin and Max values per wavelength:")
#min_max_dict = get_min_max_per_wavelength(extinction, wavelengths)
#for wl, min_max in min_max_dict.items():
#    print(f"Wavelength {wl} nm: Min = {min_max['min']}, Max = {min_max['max']}")