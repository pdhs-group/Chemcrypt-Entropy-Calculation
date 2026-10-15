import argparse
from entropyops import *
from experiment import ExperimentConfiguration
from quantization import FixedResolutionQuantizer, StandardDeviationBinner
import numpy as np
import pandas as pd
import os
import shutil
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("window_size", nargs="?", type=int, default=1)
parser.add_argument(
    "--method",
    choices=("standard-deviation", "fixed-resolution"),
    default="standard-deviation",
)
parser.add_argument("--decimal-places", type=int, default=2)
args = parser.parse_args()
config = args.config

output_path = f"{config}/calculation_binning_output"

if os.path.isdir(output_path):
    shutil.rmtree(output_path)
    print(f"Clearing previous cache....")

os.mkdir(output_path)
print(f"\nOutput folder created in {config} ")

experiment = ExperimentConfiguration.from_directory(config)
if args.method == "standard-deviation":
    quantizer = StandardDeviationBinner(args.window_size)
else:
    quantizer = FixedResolutionQuantizer(args.decimal_places)
quantized = quantizer.transform(experiment)
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

quantized_matrix = quantized.values
if args.method == "standard-deviation":
    bin_edges_list = quantized.bin_edges
    num_bins_per_wv = quantized.bins_per_wavelength
    num_bins_per_wv_df = pd.DataFrame(num_bins_per_wv.reshape(1, -1), columns=wavelengths)
    num_bins_per_wv_df.to_csv(f"{output_path}/total_bins_per_wv.csv", index=False)
    save_dict_to_json(bin_edges_list, f"{output_path}/bin_edges_list_per_wv.json")
    matrix_filename = "Binned_matrix.csv"
else:
    symbol_counts = np.array(list(quantized.max_output_size.values()))
    pd.DataFrame(symbol_counts.reshape(1, -1), columns=wavelengths).to_csv(
        f"{output_path}/unique_symbols_per_wv.csv",
        index=False,
    )
    matrix_filename = "Quantized_matrix.csv"

quantized_matrix_df = pd.DataFrame(quantized_matrix, index=sample_names, columns=wavelengths)
quantized_matrix_df.to_csv(f"{output_path}/{matrix_filename}")
print(f"\nQuantized matrix created: {matrix_filename}")

save_numpy_arrays_to_csv(
    extinction,
    sample_names,
    wavelengths,
    f"{output_path}/extinction_data.csv",
)
save_dict_to_json(quantized.max_output_size, f"{output_path}/max_output_size.json")
save_dict_to_json(
    {
        "method": args.method,
        "window_size": args.window_size if args.method == "standard-deviation" else None,
        "decimal_places": args.decimal_places if args.method == "fixed-resolution" else None,
    },
    f"{output_path}/quantization.json",
)

frequency_lookup = quantized.frequency_matrix()
save_numpy_arrays_to_csv(
    frequency_lookup,
    sample_names,
    wavelengths,
    f"{output_path}/frequency_matrix.csv",
)

quantized_distributions = quantized.distributions
save_dict_to_json(quantized_distributions,f"{output_path}/bin_probability_distributions.json")
save_dict_to_json(quantized_distributions, f"{output_path}/probability_distributions.json")
print(f"\nPer Bin probability distribution - file created")

probability_matrix = quantized.probability_matrix()
save_numpy_arrays_to_csv(
    probability_matrix,
    sample_names,
    wavelengths,
    f"{output_path}/probability_matrix.csv",
)

entropy_matrix = quantized.entropy_matrix()
save_numpy_arrays_to_csv(
    entropy_matrix,
    sample_names,
    wavelengths,
    f"{output_path}/entropy_matrix.csv",
)

entropy_per_wl, avg_entropy, total_entropy = quantized.entropy
entropy_per_wl = [
    (float(wavelength), float(entropy))
    for wavelength, entropy in entropy_per_wl
]
save_dict_to_json(entropy_per_wl, f"{output_path}/entropy_per_wavelength.json")
save_output_to_txt(entropy_per_wl, f"{output_path}/entropy_per_wavelength.txt")
print(f"\nEntropy per wavelength - file created")


cond_entropy, avg_cond_entropy = compute_conditional_binned_entropy(quantized_matrix,sample_names)
save_dict_to_json(cond_entropy,f"{output_path}/conditional_entropy_per_sample.json")
print(f"\nConditional Entropy per Sample - file created")

max_entropy = quantized.maximum_entropy

info_per_sample, avg_info = compute_binned_information_content_per_sample(quantized_matrix, quantized_distributions, wavelengths, sample_names)
save_output_to_txt(info_per_sample, f"{output_path}/info_per_sample.txt")
save_output_to_txt(info_per_sample, f"{output_path}/information_per_sample.txt")
plot_entropy_per_sample(
    info_per_sample,
    f"{output_path}/information_per_sample.png",
)
print("\nInformation per Sample - File created")

wv_entropy_performance = compute_wv_entropy_diff_from_max(quantized_distributions, wavelengths)
wv_entropy_performance = {
    float(wavelength): float(difference)
    for wavelength, difference in wv_entropy_performance.items()
}
save_dict_to_json(wv_entropy_performance,f"{output_path}/entropy_diff_from_max.json")
print(f"\nEntropy difference from Max - file created")

plot_json_dict_data(wv_entropy_performance, f"{output_path}/Entropy_diff_from_max")
print("Entropy difference Path plot created")

markov_entropy_per_wavelength, markov_total_entropy = estimate_markov_conditional_entropy(quantized_matrix)
save_dict_to_json(
    dict(zip(wavelengths[1:].astype(float), markov_entropy_per_wavelength[1:])),
    f"{output_path}/markov_conditional_entropy.json",
)
save_dict_to_json(
    estimate_adjacent_mutual_information(quantized_matrix, wavelengths),
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
