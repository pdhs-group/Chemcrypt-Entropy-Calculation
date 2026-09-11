from entropyops import (
	calculate_input_entropy,
	plot_entropy_per_sample,
	system_entropy_efficiency,
)
from utils import save_dict_to_json, save_numpy_arrays_to_csv, save_output_to_txt
from chemcrypt_oop import RawEntropyAnalysis
import os
import sys

input_configuration = sys.argv[1] if len(sys.argv) > 1 else exit(1)
output_folder = sys.argv[2] if len(sys.argv) > 2 else "calculation_entropy_all"

input_configuration += "/"
input_configuration_path = input_configuration + "numpy-files"
output_folder = input_configuration  + output_folder

os.mkdir(output_folder)
os.chmod(output_folder, 0o777)
print(f"\nCreated Output Folder: {output_folder}")

analysis = RawEntropyAnalysis.from_directory(input_configuration)
experiment = analysis.experiment
extinction = experiment.extinction
sample_names = experiment.sample_names
wavelengths = experiment.wavelengths
save_numpy_arrays_to_csv(extinction, sample_names, wavelengths, f"{output_folder}/extinction_data.csv")
print("\nExtinction data (Every run counted individually) saved to CSV.")

max_output_size = analysis.max_output_size
save_dict_to_json(max_output_size, f"{output_folder}/max_output_size.json")
print("\nUnique values per wavelength created")

frequency_lookup = analysis.frequency_matrix()
save_numpy_arrays_to_csv(frequency_lookup, sample_names, wavelengths, f"{output_folder}/frequency_matrix.csv")
print("\nFrequency Matrix of unique values per wavelength created")

distributions = analysis.distributions
save_dict_to_json(distributions, f"{output_folder}/probability_distributions.json")
print("\nProbability computed for each wavelength each component.")

prob_matrix = analysis.probability_matrix()
save_numpy_arrays_to_csv(prob_matrix, sample_names, wavelengths, f"{output_folder}/probability_matrix.csv")   
print("\nProbability Matrix created")                                    

entropy_matrix = analysis.entropy_matrix()
save_numpy_arrays_to_csv(entropy_matrix, sample_names, wavelengths, f"{output_folder}/entropy_matrix.csv")
print("\nEntropy Matrix created")

information_per_sample, average_information = analysis.information_per_sample()
save_output_to_txt(information_per_sample, f"{output_folder}/information_per_sample.txt")
print("\nInformation per Sample saved to TXT file.")

entropy_per_wavelength, average_entropy_per_wv, total_average_entropy = analysis.entropy_per_wavelength()
save_output_to_txt(entropy_per_wavelength, f"{output_folder}/entropy_per_wavelength.txt")
print("\nEntropy per Wavelength saved to TXT file.")

plot_entropy_per_sample(information_per_sample, f"{output_folder}/information_per_sample.png")
print("\nAverage information per sample plot saved")

print("\n****************************************************************")

print("\nMax Entropy:", analysis.maximum_entropy)
print("\nSystem Entropy:", total_average_entropy)
print("\nAverage Entropy per Wavelength:", average_entropy_per_wv)
print("\nAverage Information per Sample:", average_information)
print("\nInput Entropy:", calculate_input_entropy(sample_names))
print("\nSystem Entropy Efficiency:", system_entropy_efficiency(total_average_entropy, analysis.maximum_entropy))

print("\n****************************************************************")
