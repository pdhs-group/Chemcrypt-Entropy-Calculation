from utils import save_dict_to_json
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


def plot_fractional_hamming_weight_histograms(binary_op_data, configs, op_loc):
    """Plot within-print and cross-print fractional pairwise Hamming weights."""
    os.makedirs(op_loc, exist_ok=True)
    bits_per_sample = binary_op_data.bits_per_sample
    if bits_per_sample < 1:
        raise ValueError("bits_per_sample must be positive")

    for config_name in configs:
        measurement_distances = binary_op_data.intra_hamming_distance_measurements(
            config_name
        )
        printing_distances = {
            pair: distance
            for pair, distance in binary_op_data.intra_hamming_distance_all_samples(
                config_name
            ).items()
            if (int(pair.strip("()").split(",")[0]) - 1) // 10
            != (int(pair.strip("()").split(",")[1]) - 1) // 10
        }

        measurement_fractional_weights = (
            np.asarray(list(measurement_distances.values()), dtype=float)
            / bits_per_sample
        )
        printing_fractional_weights = (
            np.asarray(list(printing_distances.values()), dtype=float)
            / bits_per_sample
        )

        for name, values, pair_label in (
            (
                "10C2",
                measurement_fractional_weights,
                "within-print pairs",
            ),
            (
                "30C2_cross_print",
                printing_fractional_weights,
                "cross-print pairs",
            ),
        ):
            if values.size == 0:
                print(f"No {pair_label} found for {config_name}")
                continue

            plt.figure()
            plt.hist(
                values,
                bins=np.linspace(0.0, 1.0, 21),
                range=(0.0, 1.0),
                color="steelblue" if name == "10C2" else "seagreen",
                edgecolor="black",
            )
            plt.title(
                f"Fractional Hamming weight - {config_name} ({pair_label})"
            )
            plt.xlabel("Fractional Hamming weight")
            plt.ylabel("Pair count")
            plt.xlim(0.0, 1.0)
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            output_file = os.path.join(
                op_loc,
                f"Fractional_HW_Histogram-{config_name}-{name}.png",
            )
            plt.savefig(output_file, dpi=300)
            plt.close()
            print(f"Fractional Hamming weight histogram saved: {output_file}")


def plot_inter_configuration_fractional_hamming_distance_histogram(
    binary_op_data,
    op_loc,
    random_seed=42,
):
    """Save 81 histograms from six samples across each 81C2 config pair."""
    os.makedirs(op_loc, exist_ok=True)
    bits_per_sample = binary_op_data.bits_per_sample
    if bits_per_sample < 1:
        raise ValueError("bits_per_sample must be positive")

    distances_by_config = binary_op_data.random_sample_inter_hamming_distances_by_variation(
        random_seed
    )
    for config_name, distances in distances_by_config.items():
        fractional_distances = np.asarray(distances, dtype=float) / bits_per_sample
        if fractional_distances.size == 0:
            print(f"No inter-configuration pairs found for {config_name}")
            continue

        plt.figure()
        plt.hist(
            fractional_distances,
            bins=np.linspace(0.0, 1.0, 21),
            range=(0.0, 1.0),
            color="darkorange",
            edgecolor="black",
        )
        plt.title(
            f"Fractional inter-configuration Hamming distance - {config_name} "
            "(81C2 computations)"
        )
        plt.xlabel("Fractional Hamming distance")
        plt.ylabel("Pair count")
        plt.xlim(0.0, 1.0)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        output_file = os.path.join(
            op_loc,
            f"Inter_Configuration_Fractional_HD_Histogram-{config_name}.png",
        )
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Inter-configuration fractional Hamming distance histogram saved: {output_file}")


def plot_combined_fractional_hamming_distance_histograms(
    binary_op_data,
    configs,
    op_loc,
    random_seed=42,
    x_limits=(0.0, 1.0),
):
    """Plot measurement, printing, and inter-configuration distances together."""
    os.makedirs(op_loc, exist_ok=True)
    bits_per_sample = binary_op_data.bits_per_sample
    if bits_per_sample < 1:
        raise ValueError("bits_per_sample must be positive")
    x_min, x_max = x_limits
    if not 0.0 <= x_min < x_max <= 1.0:
        raise ValueError("x_limits must satisfy 0 <= x_min < x_max <= 1")

    inter_distances_by_config = (
        binary_op_data.random_sample_inter_hamming_distances_by_variation(random_seed)
    )

    for config_name in configs:
        intra_distances = binary_op_data.intra_hamming_distance_all_samples(config_name)
        measurement_distances = []
        printing_distances = []
        for pair, distance in intra_distances.items():
            first, second = (int(value) for value in pair.strip("()").split(","))
            target = (
                measurement_distances
                if (first - 1) // 10 == (second - 1) // 10
                else printing_distances
            )
            target.append(distance)

        measurement_fractional_distances = (
            np.asarray(measurement_distances, dtype=float) / bits_per_sample
        )
        printing_fractional_distances = (
            np.asarray(printing_distances, dtype=float) / bits_per_sample
        )
        inter_fractional_distances = (
            np.asarray(inter_distances_by_config.get(config_name, []), dtype=float)
            / bits_per_sample
        )

        if (
            measurement_fractional_distances.size == 0
            and printing_fractional_distances.size == 0
            and inter_fractional_distances.size == 0
        ):
            print(f"No combined distances found for {config_name}")
            continue

        plt.figure()
        bins = np.linspace(0.0, 1.0, 101)
        plt.hist(
            measurement_fractional_distances,
            bins=bins,
            range=(0.0, 1.0),
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
            label=f"Within measurement groups ({measurement_fractional_distances.size})",
        )
        plt.hist(
            printing_fractional_distances,
            bins=bins,
            range=(0.0, 1.0),
            color="seagreen",
            edgecolor="black",
            alpha=0.7,
            label=f"Within prints ({printing_fractional_distances.size})",
        )
        plt.hist(
            inter_fractional_distances,
            bins=bins,
            range=(0.0, 1.0),
            color="darkorange",
            edgecolor="black",
            alpha=0.7,
            label=f"Inter-configuration ({inter_fractional_distances.size})",
        )
        plt.title(f"Combined fHD - {config_name} ")
        plt.xlabel("Fractional Hamming distance")
        plt.ylabel("Pair count")
        plt.xlim(x_min, x_max)
        plt.xticks(np.arange(x_min,x_max + 0.1,0.1))
        y_min, y_max = plt.gca().get_ylim()                         
        plt.yticks(np.arange(y_min, y_max + 10, 10))
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        output_file = os.path.join(
            op_loc,
            f"Combined_fHD-{config_name}.png",
        )
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Combined fractional Hamming distance histogram saved: {output_file}")

def calculate_pseudorandomness(hd_data, op_size):
    randomness = defaultdict(list)
    for config, data in hd_data.items():
        for value in data:
            randomness[config].append((f"1 : {(value / op_size)}", f"0 : {1 - (value / op_size)}"))

    return randomness

config = sys.argv[1] if len(sys.argv) > 1 else exit(1)

S_factor = int(sys.argv[2] if len(sys.argv) > 2 else 1)
x_min = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
x_max = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
plot_individual_histograms = (
    sys.argv[5].lower() in {"1", "true", "yes"}
    if len(sys.argv) > 5
    else False
)

output_path = f"{config}/Binary_Transformation"
output_path_hamming_distances = f"{output_path}/HD_plots_S-{S_factor}"
output_path_hamming_weights = f"{output_path}/HW_plots_S-{S_factor}"


if os.path.isdir(output_path):
    shutil.rmtree(output_path)
    print(f"Clearing previous cache....")

os.mkdir(output_path)
print(f"\nOutput folder created in {config} ")

os.mkdir(output_path_hamming_distances)
print(f"\nHammimg plots Output folder created in {config} ")

experiment = ExperimentConfiguration.from_directory(config)
binned = StandardDeviationBinner(S_factor).transform(experiment)
binary = BinaryConversion(binned)
binary_output_data = binary.outputs


print(f"Total samples processed: {experiment.measurement_count}")
print(f"Total bits per sample: {binary.bits_per_sample}")

binary_hamming_weights = binary.hamming_weights()
output_pseudorandomness = binary.pseudorandomness()

save_dict_to_json(binary_output_data,f"{output_path}/binary_output_data")
print("\nSaved Binary Transformation to JSON")

save_dict_to_json(binary_hamming_weights,f"{output_path}/binary_hamming_weight_data")
print("\nSaved Binary Hamming Weights to JSON")

save_dict_to_json(output_pseudorandomness,f"{output_path}/output_pseudorandomness")
print("\nSaved Binary Output Randomness Data to JSON")

histogram_data_path = os.path.join(output_path_hamming_distances, "Histogram_Data")
os.makedirs(histogram_data_path, exist_ok=True)

measurement_distances = {
    str(config_name): binary.intra_hamming_distance_measurements(config_name)
    for config_name in experiment.variation_names
}
save_dict_to_json(
    measurement_distances,
    os.path.join(histogram_data_path, "intra_config_measurements.json"),
)

printing_distances = {
    str(config_name): {
        pair: distance
        for pair, distance in binary.intra_hamming_distance_all_samples(config_name).items()
        if (int(pair.strip("()").split(",")[0]) - 1) // 10
        != (int(pair.strip("()").split(",")[1]) - 1) // 10
    }
    for config_name in experiment.variation_names
}
save_dict_to_json(
    printing_distances,
    os.path.join(histogram_data_path, "intra_config_printing.json"),
)

save_dict_to_json(
    binary.random_sample_inter_hamming_distances_by_pair(random_seed=42),
    os.path.join(histogram_data_path, "inter_config.json"),
)
print(f"\nSaved histogram distance data to {histogram_data_path}")

if plot_individual_histograms:
    hamming_weight_histogram_path = os.path.join(
        output_path_hamming_weights,
        "Fractional_HW_Histograms",
    )
    plot_fractional_hamming_weight_histograms(
        binary,
        experiment.variation_names,
        hamming_weight_histogram_path,
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

combined_histogram_path = os.path.join(
    output_path_hamming_distances,
    "Combined_30C2_81C2",
)
plot_combined_fractional_hamming_distance_histograms(
    binary,
    experiment.variation_names,
    combined_histogram_path,
    random_seed=42,
    x_limits=(x_min, x_max),
)
