from entropyops_bins import save_dict_to_json
from experiment import ExperimentConfiguration
from quantization import StandardDeviationBinner
from binary_conversion import BinaryConversion
import sys
import shutil
import os
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import itertools
import numpy as np
import math
from tqdm import tqdm

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

    return binary_outputs, sample_count, bits_per_sample

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
    group_size = 10

    for config, hw in hw_data.items():
        np_hws = np.array(hw)
        num_groups = math.ceil(len(np_hws) / group_size)
        diffs = []
        
        for g in range(num_groups):
            group_hws = np_hws[g*group_size : (g+1)*group_size]
            median_hw = np.median(group_hws)
            diffs.extend((group_hws - median_hw).tolist())
            
        hamming_weight_diff[config] = diffs
        total_configs += 1

    print(f"Total configs processed: {total_configs}")
    return hamming_weight_diff


def plot_median_hw_per_group_from_file(hw_file_path, group_size=10, out_dir=None, show=False):

    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    with open(hw_file_path, 'r') as f:
        hw_data = json.load(f)

    for config, values in hw_data.items():
        arr = np.array(values)
        if arr.size == 0:
            continue
        num_groups = math.ceil(arr.size / group_size)
        medians = []
        labels = []
        for g in range(num_groups):
            group_vals = arr[g * group_size: (g + 1) * group_size]
            med = float(np.median(group_vals)) if group_vals.size > 0 else float('nan')
            medians.append(med)
            labels.append(f"g{g+1}")

        plt.bar(range(1, len(medians) + 1), medians, color='C0', edgecolor='black')
        plt.xticks(range(1, len(medians) + 1), labels, rotation=45)
        plt.title(f"Median Hamming Weight per Group - {config}")
        plt.xlabel("Group")
        plt.ylabel("Median Hamming Weight")
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        out_file = os.path.join(out_dir, f"HW_Medians_{config}.png")
        plt.tight_layout()
        plt.savefig(out_file)
        print(f"Saved median-per-group plot for {config} to {out_file}")
        if show:
            plt.show()
        plt.clf()


def plot_hamming_weight_differences(hm_wt_diff, op_loc):

    for config,hw_diffs in hm_wt_diff.items():

        plt.bar(list(range(1,len(hw_diffs) + 1)), hw_diffs, color = 'b')
        plt.xticks(list(range(1,len(hw_diffs) + 1)))
        plt.grid(True,which="both")
        plt.title(f"Hamming weight differences - Group Median value , Sample : {str(config)}")
        plt.xlabel(f"Samples : {str(config)}")
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
    group_size = 10
    num_groups = math.ceil(len(bin_data[config_name]) / group_size)

    for g in range(num_groups):
        for i, bitstring1 in enumerate(bin_data[config_name][g * group_size: (g + 1)*group_size]):
                for j, bitstring2 in enumerate(bin_data[config_name][g * group_size + i + 1 : (g + 1)*group_size ]):
                    total_intra_config_pairs += 1
                    xor_result = int(bitstring1,2) ^ int(bitstring2,2)
                    intra_config_hw_distance[f"({g*group_size + i + 1},{g*group_size + i + 1 + j + 1})"] = bin(xor_result).count('1')

    #print(f"\n Total Intra config pairs: {total_intra_config_pairs}\n")
    return intra_config_hw_distance


def inter_sample_hamming_distance(bin_data, config_a, config_b, out_dir=None):

    if config_a not in bin_data or config_b not in bin_data:
        raise KeyError(f"One or both configs not found: {config_a}, {config_b}")

    distances = {}

    for i, bitstring1 in enumerate(bin_data[config_a]):
        for j, bitstring2 in enumerate(bin_data[config_b]):
            xor_result = int(bitstring1, 2) ^ int(bitstring2, 2)
            distances[f"({i+1},{j+1})"] = bin(xor_result).count('1')

    os.makedirs(f"{out_dir}/JSON", exist_ok=True)
    os.makedirs(f"{out_dir}/Map", exist_ok=True)

    out_json = f"{out_dir}/JSON/{config_a}_vs_{config_b}.json"
    try:
        save_dict_to_json(distances, out_json)
        print(f"JSON saved for {config_a} - {config_b}")
    except Exception:
        with open(out_json, 'w') as f:
            json.dump(distances, f, indent=2)

    return distances


    # build matrix for heatmap
    max_i = 0
    max_j = 0
    for k in distances.keys():
        i_str, j_str = k.strip('()').split(',')
        i = int(i_str)
        j = int(j_str)
        max_i = max(max_i, i)
        max_j = max(max_j, j)

    mat = np.zeros((max_i, max_j), dtype=int)
    for k, v in distances.items():
        i_str, j_str = k.strip('()').split(',')
        i = int(i_str) - 1
        j = int(j_str) - 1
        mat[i, j] = int(v)

    out_png = out_json = f"{out_dir}/Map/{config_a}_vs_{config_b}.png"
    plt.figure(figsize=(max(6, max_j/2), max(4, max_i/2)))
    im = plt.imshow(mat, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(im, label='Hamming distance')
    xa = config_b
    ya = config_a
    plt.xlabel(f"{xa} (index)")
    plt.ylabel(f"{ya} (index)")
    plt.title(f"Hamming distances: {config_a} vs {config_b}")
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"\nHeatMap saved for {config_a} - {config_b}")
    plt.clf()


def plot_intra_hamming_distance_differences (binary_op_data, configs, op_loc):

    sum_total = 0

    for config_name in configs:
        hd_data = binary_op_data.intra_hamming_distance_measurements(config_name)

        sum_total += sum(list(hd_data.values())) / 135

        plt.bar(list(hd_data.keys()), list(hd_data.values()), color = 'blue', edgecolor = 'black', width=0.5)

        plt.title(f"Hamming distances for every combination - {config_name}")
        plt.xlabel(f'Sample combinations (Tuples)')
        plt.ylabel('Hamming Distances')
        plt.xticks(fontsize=8, rotation=45, ha='right')

        plt.grid(axis='y', linestyle = '--', alpha = 0.5)
        plt.savefig(f"{op_loc}/HD_Diffs-{config_name}")

        print(f"Hamming Distance Diffs : {config_name} saved ")
        plt.clf()

    print(sum_total / len(configs))


def plot_grouped_intra_fractional_hamming_distance_histograms(binary_op_data, configs, op_loc):
    """Plot grouped intra-configuration distances: 3 x 10C2 per configuration."""
    os.makedirs(op_loc, exist_ok=True)
    bits_per_sample = binary_op_data.bits_per_sample
    if bits_per_sample < 1:
        raise ValueError("bits_per_sample must be positive")

    for config_name in configs:
        distances = binary_op_data.intra_hamming_distance_measurements(config_name)
        fractional_distances = np.asarray(
            list(distances.values()),
            dtype=float,
        ) / bits_per_sample

        if fractional_distances.size == 0:
            print(f"No intra-configuration pairs found for {config_name}")
            continue

        plt.figure()
        plt.hist(
            fractional_distances,
            bins=np.linspace(0.0, 1.0, 21),
            range=(0.0, 1.0),
            color="steelblue",
            edgecolor="black",
        )
        plt.title(f"Grouped fractional intra-configuration Hamming distance - {config_name}")
        plt.xlabel("Fractional Hamming distance")
        plt.ylabel("Pair count")
        plt.xlim(0.0, 1.0)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        output_file = os.path.join(op_loc, f"Grouped_Fractional_HD_Histogram-{config_name}.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Grouped fractional Hamming distance histogram saved: {output_file}")


def plot_all_intra_fractional_hamming_distance_histograms(binary_op_data, configs, op_loc):
    """Plot all intra-configuration distances: 30C2 per configuration."""
    os.makedirs(op_loc, exist_ok=True)
    bits_per_sample = binary_op_data.bits_per_sample
    if bits_per_sample < 1:
        raise ValueError("bits_per_sample must be positive")

    for config_name in configs:
        distances = binary_op_data.intra_hamming_distance_all_samples(config_name)
        fractional_distances = np.asarray(list(distances.values()), dtype=float) / bits_per_sample

        if fractional_distances.size == 0:
            print(f"No intra-configuration pairs found for {config_name}")
            continue

        plt.figure()
        plt.hist(
            fractional_distances,
            bins=np.linspace(0.0, 1.0, 21),
            range=(0.0, 1.0),
            color="steelblue",
            edgecolor="black",
        )
        plt.title(f"All fractional intra-configuration Hamming distance - {config_name}")
        plt.xlabel("Fractional Hamming distance")
        plt.ylabel("Pair count")
        plt.xlim(0.0, 1.0)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        output_file = os.path.join(op_loc, f"All_Fractional_HD_Histogram-{config_name}.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"All-sample fractional Hamming distance histogram saved: {output_file}")


def plot_inter_configuration_fractional_hamming_distance_histogram(
    binary_op_data,
    op_loc,
    random_seed=42,
):
    """Plot one histogram for one random sample from each configuration: 81C2."""
    os.makedirs(op_loc, exist_ok=True)
    bits_per_sample = binary_op_data.bits_per_sample
    if bits_per_sample < 1:
        raise ValueError("bits_per_sample must be positive")

    distances = binary_op_data.random_sample_inter_hamming_distances(random_seed)
    fractional_distances = np.asarray(list(distances.values()), dtype=float) / bits_per_sample
    if fractional_distances.size == 0:
        print("No inter-configuration pairs found")
        return

    plt.figure()
    plt.hist(
        fractional_distances,
        bins=np.linspace(0.0, 1.0, 21),
        range=(0.0, 1.0),
        color="darkorange",
        edgecolor="black",
    )
    plt.title("Fractional inter-configuration Hamming distance (81C2)")
    plt.xlabel("Fractional Hamming distance")
    plt.ylabel("Pair count")
    plt.xlim(0.0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    output_file = os.path.join(op_loc, "Inter_Configuration_Fractional_HD_Histogram.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Inter-configuration fractional Hamming distance histogram saved: {output_file}")

def calculate_pseudorandomness(hd_data, op_size):
    randomness = defaultdict(list)
    for config, data in hd_data.items():
        for value in data:
            randomness[config].append((f"1 : {(value / op_size)}", f"0 : {1 - (value / op_size)}"))

    return randomness

config = sys.argv[1] if len(sys.argv) > 1 else exit(1)

S_factor = int(sys.argv[2] if len(sys.argv) > 2 else 1)

output_path = f"{config}/Binary_Transformation"
output_path_hamming_weights = f"{output_path}/HW_plots_S-{S_factor}"
output_path_hamming_distances = f"{output_path}/HD_plots_S-{S_factor}"
output_path_inter_config_hamming = f"{output_path}/HD_Inter-Config-{S_factor}"


if os.path.isdir(output_path):
    shutil.rmtree(output_path)
    print(f"Clearing previous cache....")

os.mkdir(output_path)
print(f"\nOutput folder created in {config} ")

os.mkdir(output_path_hamming_weights)
print(f"\nHammimg plots Output folder created in {config} ")

os.mkdir(output_path_hamming_distances)
print(f"\nHammimg plots Output folder created in {config} ")

experiment = ExperimentConfiguration.from_directory(config)
unique_samples = experiment.variation_names
binned = StandardDeviationBinner(S_factor).transform(experiment)
binary = BinaryConversion(binned)
binary_output_data = binary.outputs


print(f"Total samples processed: {experiment.measurement_count}")
print(f"Total bits per sample: {binary.bits_per_sample}")

binary_hamming_weights = binary.hamming_weights()
hamming_weight_differences = binary.hamming_weight_differences()
output_pseudorandomness = binary.pseudorandomness()

save_dict_to_json(binary_output_data,f"{output_path}/binary_output_data")
print("\nSaved Binary Transformation to JSON")

save_dict_to_json(binary_hamming_weights,f"{output_path}/binary_hamming_weight_data")
print("\nSaved Binary Hamming Weights to JSON")

# Generate median-per-group bar plots from the saved hamming-weight data
hw_file = f"{output_path}/binary_hamming_weight_data"
plot_median_hw_per_group_from_file(hw_file, group_size=10, out_dir=output_path_hamming_weights)

save_dict_to_json(output_pseudorandomness,f"{output_path}/output_pseudorandomness")
print("\nSaved Binary Output Randomness Data to JSON")

plot_hamming_weight_differences(hamming_weight_differences, output_path_hamming_weights)


plot_intra_hamming_distance_differences(
    binary,
    experiment.variation_names,
    output_path_hamming_distances,
)

grouped_histogram_path = os.path.join(
    output_path_hamming_distances,
    "Grouped_10C2_Histograms",
)
plot_grouped_intra_fractional_hamming_distance_histograms(
    binary,
    experiment.variation_names,
    grouped_histogram_path,
)

all_samples_histogram_path = os.path.join(
    output_path_hamming_distances,
    "All_30C2_Histograms",
)
plot_all_intra_fractional_hamming_distance_histograms(
    binary,
    experiment.variation_names,
    all_samples_histogram_path,
)

inter_configuration_histogram_path = os.path.join(
    output_path_hamming_distances,
    "Inter_Configuration_81C2",
)
plot_inter_configuration_fractional_hamming_distance_histogram(
    binary,
    inter_configuration_histogram_path,
    random_seed=42,
)

min_HD_dist_combi = {}
minimum = 10000
sum_total = 0

for index, sample_i in enumerate(unique_samples):
    for sample_j in unique_samples[index + 1 : ]:
        dist_dict = binary.inter_hamming_distance(sample_i, sample_j)
        os.makedirs(f"{output_path_inter_config_hamming}/JSON", exist_ok=True)
        os.makedirs(f"{output_path_inter_config_hamming}/Map", exist_ok=True)
        save_dict_to_json(
            dist_dict,
            f"{output_path_inter_config_hamming}/JSON/{sample_i}_vs_{sample_j}.json",
        )
        sum_total += sum(list(dist_dict.values())) / 900
        #minimum = min(minimum, min(list(dist_dict.values())))
        #min_HD_dist_combi[(sample_i,sample_j)] = minimum

#print(f"\n{min_HD_dist_combi}")
print(sum_total / 3240)


#print(f"\nTotal Pairs: {compute_inter_collision_binary(binary_output_data)[0]}")
#print(f"\nTotal Collisions observed: {compute_inter_collision_binary(binary_output_data)[1]}")
#print(f"\nPossible Collision rate: {compute_inter_collision_binary(binary_output_data)[2]}")
