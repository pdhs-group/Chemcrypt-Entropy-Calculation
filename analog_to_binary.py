from entropyops_bins import *
import sys
import shutil
import os
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import numpy as np
import math


def convert_to_binary(bin_matrix,sample_names):
    max_bins = int(np.max(bin_matrix))
    max_bit_width = math.ceil(math.log2(max_bins + 1))

    print(f"Max bin index found: {max_bins}")
    print(f"Fixed bit-width per bin: {max_bit_width} bits")

    binary_outputs = defaultdict(list)

    for i in range(bin_matrix.shape[0]):
        row = bin_matrix[i]

        binary_row = "".join(format(int(val), f'0{max_bit_width}b') for val in row)
        name = sample_names[i]

        #binary_outputs.append((name,binary_row))
        binary_outputs[name].append((binary_row))

    sample_count = len(list(binary_outputs.keys())) * len(list(binary_outputs.values())[0]) 
    bits_per_sample = max_bit_width * bin_matrix.shape[1]

    print(f"Total samples processed: {sample_count}")
    print(f"Total bits per sample: {bits_per_sample}")

    return binary_outputs

def get_hamming_weights(bin_data):

    hamming_weights = defaultdict(list)
    total_samples = 0
    
    for config,samples in bin_data.items():
        for bitstring in samples:
            weight = bitstring.count('1')
            hamming_weights[config].append(weight)
            total_samples += 1

    print(f"Total samples processed: {total_samples}")
    return hamming_weights

def get_hamming_weight_diff(hw_data):
    hamming_weight_diff = defaultdict(list)
    total_configs = 0

    for config,hw in hw_data.items():
        np_hws = np.array(hw)
        median_hw = np.median(np_hws)
        hamming_weight_diff[config] = (np_hws - median_hw).tolist()
        total_configs += 1

    print(f"Total configs processed: {total_configs}")
    return hamming_weight_diff  


def plot_hamming_weight_differences(hm_wt_diff, op_loc):

    for config,hw_diffs in hm_wt_diff.items():

        plt.plot(list(range(1,len(hw_diffs) + 1)), hw_diffs, color = 'b')
        plt.xticks(list(range(1,len(hw_diffs) + 1)))
        plt.grid(True,which="both")
        plt.title(f"Hamming weight differences - Median value , Sample : {str(config)}")
        plt.xlabel(f"Samples 1 - 30 : {str(config)}")
        plt.ylabel("Hamming weight diffs")
        plt.xticks(rotation=45)
        plt.savefig(f"{op_loc}/HW_Diffs-{config}")
        print(f"Hamming Weight Diffs : {config} saved ")
        plt.clf()                                                   #If this is commented then all graphs go on same plot - Very difficult to plot


def compute_inter_collision_binary(bin_data):
            
    inter_config_collisions = 0
    total_inter_config_pairs = 0
    
    for (c1, v1), (c2, v2) in itertools.combinations(bin_data.items(), 2):
        for bitstring1 in v1:
            for bitstring2 in v2:
                total_inter_config_pairs += 1
                if bitstring1 == bitstring2:
                    inter_config_collisions += 1
    
    # CORRECT rate calculation
    collision_rate = ((inter_config_collisions / total_inter_config_pairs) * 100) if total_inter_config_pairs > 0 else 0

    return total_inter_config_pairs, inter_config_collisions, collision_rate

def intra_sample_hamming_distance(bin_data, config_name):

    intra_config_hw_distance = defaultdict(int)
    total_intra_config_pairs = 0

    for i, bitstring1 in enumerate(bin_data[config_name]):
        for j, bitstring2 in enumerate(bin_data[config_name][i + 1 : ]):
            total_intra_config_pairs += 1

            xor_result = int(bitstring1,2) ^ int(bitstring2,2)
            intra_config_hw_distance[f"({i + 1},{i + 1 + j + 1})"] = bin(xor_result).count('1')

    print(f"\n Total Intra config pairs: {total_intra_config_pairs}\n")
    return intra_config_hw_distance

def plot_intra_confg_hamming_distance_bar(hd_data, config_name):
    plt.bar(list(hd_data.keys()), list(hd_data.values()), color = 'blue', edgecolor = 'black', width=0.5)

    plt.title(f"Hamming distances for every combination - {config_name}")
    plt.xlabel(f'Sample combinations (Tuples)')
    plt.ylabel('Hamming Distances')
    plt.xticks(fontsize=8, rotation=45, ha='right')

    plt.grid(axis='y', linestyle = '--', alpha = 0.5)

    plt.show()


config = sys.argv[1] if len(sys.argv) > 1 else exit(1)

S_factor = int(sys.argv[2] if len(sys.argv) > 2 else 1)

extinction_path = config + "/numpy-files/extinction.npy"
sample_names_path = config + "/numpy-files/sample_name.npy"
wavelengths_path = config + "/numpy-files/wavelength.npy"

output_path = f"{config}/Binary_Transformation"
output_path_hamming_weights = f"{output_path}/HW_plots_S-{S_factor}"

if os.path.isdir(output_path):
    shutil.rmtree(output_path)
    print(f"Clearing previous cache....")

os.mkdir(output_path)
print(f"\nOutput folder created in {config} ")

os.mkdir(output_path_hamming_weights)
print(f"\nHammimg plots Output folder created in {config} ")

extinction, sample_names, wavelengths = load_data(extinction_path, sample_names_path, wavelengths_path)

min_max_array = calculate_wavelength_min_max(extinction, wavelengths)
avg_std_dev_per_wv, std_matrix = calculate_average_std_dev(extinction, sample_names)

bin_edges_list = calculate_bins_per_wavelength(min_max_array, avg_std_dev_per_wv, S_factor)[0]
bin_matrix = assign_bins_per_wavelength(extinction, bin_edges_list)

binary_output_data = convert_to_binary(bin_matrix,sample_names)
binary_hamming_weights = get_hamming_weights(binary_output_data)
hamming_weight_differences = get_hamming_weight_diff(binary_hamming_weights)

save_dict_to_json(binary_output_data,f"{output_path}/binary_output_data")
print("\nSaved Binary Transformation to JSON")

save_dict_to_json(binary_hamming_weights,f"{output_path}/binary_hamming_data")
print("\nSaved Binary Hamming Weights to JSON")

plot_hamming_weight_differences(hamming_weight_differences, output_path_hamming_weights)

print(compute_inter_collision_binary(binary_output_data))

'''
intra_Sample_hd_data = intra_sample_hamming_distance(binary_output_data,'CCMMYY')
print(intra_Sample_hd_data)
plot_intra_confg_hamming_distance_bar(intra_Sample_hd_data, "CCMMYY")
'''