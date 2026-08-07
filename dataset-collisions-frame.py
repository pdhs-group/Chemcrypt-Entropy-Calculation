from entropyops_bins import *
import sys
import shutil
import os
from collections import defaultdict,Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import numpy as np
import math

def compute_mcv_sample(bin_matrix,sample_names):
    unique_configs = np.unique(sample_names)
    min_entropy = np.empty((len(unique_configs),bin_matrix.shape[1]))
    min_entropy_per_sample = defaultdict(int)

    # We need to handle groups of 10 samples per print
    group_size = 10
    all_group_entropies = []

    for config in unique_configs:
        mask = (sample_names == config)
        config_op = bin_matrix[mask]
        
        # Split config_op into groups of 10
        num_groups = math.ceil(config_op.shape[0] / group_size)
        
        for g in range(num_groups):
            group_op = config_op[g*group_size : (g+1)*group_size]
            
            group_entropy = 0.0
            for col_idx in range(group_op.shape[1]):
                col = group_op[:, col_idx]
                counts = Counter(col)
                n_samples = len(col)
                
                mcv_value, mcv_count = counts.most_common(1)[0]
                p_mcv = mcv_count / n_samples
                group_entropy += -math.log2(p_mcv)
            
            all_group_entropies.append(group_entropy)
            min_entropy_per_sample[f"{config}_g{g}"] = group_entropy

    total_min_entropy = np.mean(all_group_entropies)

    return min_entropy, min_entropy_per_sample, total_min_entropy

def compute_inter_collision(bin_matrix, sample_names):
    
    config_to_curves = defaultdict(list)

    for i, config in enumerate(sample_names):
        config_to_curves[config].append(tuple(bin_matrix[i]))
            
    inter_config_collisions = 0
    total_inter_config_pairs = 0
    
    for (c1, v1), (c2, v2) in itertools.combinations(config_to_curves.items(), 2):
        for curve1 in v1:
            for curve2 in v2:
                total_inter_config_pairs += 1
                if curve1 == curve2:
                    inter_config_collisions += 1
    
    # CORRECT rate calculation
    collision_rate = (inter_config_collisions / total_inter_config_pairs * 100) if total_inter_config_pairs > 0 else 0

    return total_inter_config_pairs, inter_config_collisions

config = sys.argv[1] if len(sys.argv) > 1 else exit(1)
noise_factor = list(range(1,20))
collisons_per_window_size = defaultdict(int)
intra_entropy_per_window_size = defaultdict(int)
total_system_entropy_per_window_size = defaultdict(int)
max_entropy_per_window_size = defaultdict(int)

extinction_path = config + "/numpy-files/extinction.npy"
sample_names_path = config + "/numpy-files/sample_name.npy"
wavelengths_path = config + "/numpy-files/wavelength.npy"

output_path = f"{config}/calculation_entropy_frame"

if os.path.isdir(output_path):
    shutil.rmtree(output_path)
    print(f"Clearing previous cache....")

os.mkdir(output_path)
print(f"\nOutput folder created in {config} ")

extinction, sample_names, wavelengths = load_data(extinction_path, sample_names_path, wavelengths_path)

min_max_array = calculate_wavelength_min_max(extinction, wavelengths)
avg_std_dev_per_wv, std_matrix = calculate_average_std_dev(extinction, sample_names)

print("\nCalculating MCV min entropy / Intra entropy")
for window_S in tqdm(noise_factor):
    bin_edges_list = calculate_bins_per_wavelength(min_max_array, avg_std_dev_per_wv, window_S)[0]
    bin_matrix = assign_bins_per_wavelength(extinction, bin_edges_list)
    intra_entropy_per_window_size[window_S] = compute_mcv_sample(bin_matrix,sample_names)[2]

print("\nCalculating collisions")
for window_S in tqdm(noise_factor):
    bin_edges_list = calculate_bins_per_wavelength(min_max_array, avg_std_dev_per_wv, window_S)[0]
    bin_matrix = assign_bins_per_wavelength(extinction, bin_edges_list)
    collisons_per_window_size[window_S] = compute_inter_collision(bin_matrix, sample_names)[1]

print("\nCalculating Max Entropy and System Entropy")
for window_S in tqdm(noise_factor):
    bin_edges_list, num_bins_per_wv = calculate_bins_per_wavelength(min_max_array, avg_std_dev_per_wv, window_S)
    bin_matrix = assign_bins_per_wavelength(extinction, bin_edges_list)
    binned_distributions = compute_binned_probability(bin_matrix, wavelengths)

    max_entropy_per_window_size[window_S] = compute_binned_max_entropy(num_bins_per_wv)
    total_system_entropy_per_window_size[window_S] = compute_binned_entropy(binned_distributions, wavelengths)[2]

    
save_dict_to_json(collisons_per_window_size,f"{output_path}/collisions_per_S")
print("\nSaved Inter collisions to JSON")

save_dict_to_json(intra_entropy_per_window_size,f"{output_path}/Intra_entropy_per_S")
print("Saved Intra Entropy to JSON")

save_dict_to_json(max_entropy_per_window_size,f"{output_path}/Max_entropy_per_S")
print("Saved Max Entropy per Window to JSON")

save_dict_to_json(total_system_entropy_per_window_size,f"{output_path}/System_entropy_per_S")
print("Saved Total System Entropy per Window to JSON")


plt.plot(list(intra_entropy_per_window_size.keys()), list(intra_entropy_per_window_size.values()), color = 'r')
plt.plot(list(collisons_per_window_size.keys()), list(collisons_per_window_size.values()), color = 'g' )
#plt.plot(list(collisons_per_window_size.keys()), list(total_max_entropy_distance_per_window_size.values()), color = 'b')
plt.xticks(noise_factor)
plt.grid(True,which="both")
plt.title("Intra Entropy and Inter Collisions vs Noise Factor (Window Size)")
plt.xlabel("Noise Factor (S)")
plt.ylabel("Intra Entropy + Inter collisions")
plt.xticks(rotation=45)
plt.savefig(f"{output_path}/Intra_entropy_and_Collisions_plot")
print("Saved Intra Entropy and collisions vs Sample window plot")
plt.clf()

plt.plot(list(intra_entropy_per_window_size.keys()), list(max_entropy_per_window_size.values()), color = 'r')
plt.plot(list(collisons_per_window_size.keys()), list(total_system_entropy_per_window_size.values()), color = 'g' )

plt.xticks(noise_factor)
plt.grid(True,which="both")
plt.title("Max Entropy and System entropy vs Noise Factor (Window Size)")
plt.xlabel("Noise Factor (S)")
plt.ylabel("Max Entropy + Inter Entropy")
plt.xticks(rotation=45)
plt.savefig(f"{output_path}/Max_entropy_and_Inter_entropy_plot")
print("Saved Max Entropy and Inter Entropy vs Sample window plot")

