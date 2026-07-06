from entropyops import *
import os
import sys

output_folder = sys.argv[1] if len(sys.argv) > 1 else "output_entropy_all"

os.mkdir(output_folder)
os.chmod(output_folder, 0o777)
print(f"Created Output Folder: {output_folder}")

extinction, sample_names, wavelengths = load_data()
save_numpy_arrays_to_csv(extinction, sample_names, wavelengths, f"{output_folder}/extinction_data.csv")
print("\nExtinction data saved to CSV.")

max_output_size = calculate_max_output_size_per_wavelength(extinction,wavelengths)
save_dict_to_json(max_output_size, f"{output_folder}/max_output_size.json")

print("\nFrequency Matrix of unique values per wavelength created")
frequency = compute_count_matrix(extinction, max_output_size)
save_numpy_arrays_to_csv(frequency, sample_names, wavelengths, f"{output_folder}/frequency_matrix.csv")

distributions = compute_probability(extinction,max_output_size)
print("Probability computed for each wavelength each component.")
save_dict_to_json(distributions, f"{output_folder}/probability_distributions.json")
#for wv, dist in list(distributions.items()):
#    print(f"Wavelength {wv} nm:", dict(list(dist.items())))

#combined_array = combine_arrays(extinction, sample_names, wavelengths)
#print("\nCombined Array:")
#print(combined_array)

prob_matrix = build_probability_matrix(extinction, distributions) 
save_numpy_arrays_to_csv(prob_matrix, sample_names, wavelengths, f"{output_folder}/probability_matrix.csv")   
print("\nProbability Matrix created")                                    
#combined_prob_matrix = combine_arrays(prob_matrix, sample_names, wavelengths)
#print("\nCombined Probability Matrix:")
#print(combined_prob_matrix)

entropy_matrix = build_entropy_matrix(extinction, distributions)
save_numpy_arrays_to_csv(entropy_matrix, sample_names, wavelengths, f"{output_folder}/entropy_matrix.csv")
print("\nEntropy Matrix created")
#combined_entropy_matrix = combine_arrays(entropy_matrix, sample_names, wavelengths)
#print("\nCombined Entropy Matrix:")
#print(combined_entropy_matrix)

entropy_per_sample = calculate_entropy_per_sample(entropy_matrix, sample_names)
save_output_to_txt(entropy_per_sample, f"{output_folder}/entropy_per_sample.txt")
#print("\nEntropy per Sample saved to TXT file.")
#print("\nEntropy per Sample:")
#print(entropy_per_sample)

print("\nMax Entropy:", calculate_max_entropy(max_output_size))

#plot_entropy_per_sample(entropy_per_sample)

#print("\nMin and Max values per wavelength:")
#min_max_dict = get_min_max_per_wavelength(extinction, wavelengths)
#for wl, min_max in min_max_dict.items():
#    print(f"Wavelength {wl} nm: Min = {min_max['min']}, Max = {min_max['max']}")