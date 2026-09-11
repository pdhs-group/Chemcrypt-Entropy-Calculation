"""Unified entropy utilities for raw-value and binned analyses."""

from collections import defaultdict, Counter
import math
import itertools
import matplotlib.pyplot as plt
import numpy as np

from utils import *


def compute_sample_mean(extinction, sample_names):
    unique_samples = np.unique(sample_names)
    row_str = unique_samples.astype(str).reshape(-1, 1)
    mean_values = np.array(
        [np.mean(extinction[sample_names == sample], axis=0) for sample in unique_samples]
    ).round(2)
    return unique_samples, mean_values, np.hstack([row_str, mean_values])


def compute_probability(extinction, max_output_size):
    wavelengths = list(max_output_size.keys())
    total = extinction.shape[0]
    distributions = defaultdict(lambda: defaultdict(int))
    for index, wavelength in enumerate(wavelengths):
        values = extinction[:, index].round(2)
        unique, counts = np.unique(values, return_counts=True)
        for value, count in zip(unique, counts):
            distributions[wavelength][value] = count / total
    return distributions


def wavelength_value_frequency_lookup_matrix(extinction, max_output_size):
    frequency_matrix = np.empty(extinction.shape, dtype=int)
    for index in range(extinction.shape[1]):
        values = extinction[:, index].round(2)
        unique, counts = np.unique(values, return_counts=True)
        count_dict = dict(zip(unique, counts))
        frequency_matrix[:, index] = [count_dict.get(value, 0) for value in values]
    return frequency_matrix


def build_probability_matrix(data, probabilities):
    wavelengths = list(probabilities.keys())
    probability_matrix = np.empty(data.shape)
    for row_index in range(data.shape[0]):
        for column_index in range(data.shape[1]):
            wavelength = wavelengths[column_index]
            probability_matrix[row_index, column_index] = probabilities[wavelength].get(
                data[row_index, column_index], 0
            )
    return probability_matrix


def build_entropy_matrix(data, probabilities):
    wavelengths = list(probabilities.keys())
    entropy_matrix = np.empty(data.shape)
    for row_index in range(data.shape[0]):
        for column_index in range(data.shape[1]):
            wavelength = wavelengths[column_index]
            probability = max(
                probabilities[wavelength].get(data[row_index, column_index], 1e-10),
                1e-10,
            )
            entropy_matrix[row_index, column_index] = -probability * math.log2(probability)
    return entropy_matrix


def calculate_information_content_per_sample(data, probabilities, sample_names):
    wavelengths = list(probabilities.keys())
    information = np.empty((data.shape[0], 1))
    for row_index in range(data.shape[0]):
        total_information = 0.0
        for column_index in range(data.shape[1]):
            wavelength = wavelengths[column_index]
            probability = max(
                probabilities[wavelength].get(data[row_index, column_index], 1e-10),
                1e-10,
            )
            total_information += -math.log2(probability)
        information[row_index, 0] = total_information
    combined = np.hstack([sample_names.astype(str).reshape(-1, 1), information])
    return combined, np.mean(information)


def calculate_max_entropy(max_output_per_wavelength):
    return sum(math.log2(size) for size in max_output_per_wavelength.values())


def calculate_entropy_per_wavelength(distributions):
    entropy_per_wavelength = [
        (wavelength, sum(-p * math.log2(p) for p in distribution.values() if p > 0))
        for wavelength, distribution in distributions.items()
    ]
    total_entropy = sum(value for _, value in entropy_per_wavelength)
    return entropy_per_wavelength, total_entropy / len(entropy_per_wavelength), total_entropy


def get_min_max_per_wavelength(extinction, wavelengths):
    return {
        float(wavelength): {
            "min": float(extinction[:, index].min()),
            "max": float(extinction[:, index].max()),
        }
        for index, wavelength in enumerate(wavelengths)
    }


def calculate_max_output_size_per_wavelength(data, wavelengths):
    return {
        wavelengths[index]: len(np.unique(data[:, index]))
        for index in range(data.shape[1])
    }


def calculate_total_output_size(max_output_size):
    total = 1
    for size in max_output_size.values():
        total *= size
    return total


def is_greater_than_SHA256_limit(total_output_size):
    return total_output_size > 2**256


def plot_distributions(wavelengths, distributions):
    figure, axes = plt.subplots(len(wavelengths), 1, figsize=(10, 4 * len(wavelengths)))
    axes = [axes] if len(wavelengths) == 1 else axes
    for index, axis in enumerate(axes):
        values, probabilities = zip(*sorted(distributions[index].items()))
        axis.bar(values, probabilities)
        axis.set_title(f"Wavelength: {wavelengths[index]:.2f} nm")
        axis.set_xlabel("Extinction Value")
        axis.set_ylabel("Probability")
    plt.tight_layout()
    plt.savefig("probability_distributions.png")
    plt.show()


def plot_entropy_per_sample(entropy_per_sample, output_filename):
    plt.figure(figsize=(12, 6))
    plt.bar(entropy_per_sample[:, 0], entropy_per_sample[:, 1].astype(float))
    plt.title("Entropy per Sample")
    plt.xlabel("Sample Name")
    plt.ylabel("Entropy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_filename)


def load_legacy_data(data_path, sample_name_path, wavelength_path):
    """Load data using the legacy rounded-and-scaled representation."""
    extinction, sample_names, wavelengths = load_data(
        data_path,
        sample_name_path,
        wavelength_path,
    )
    return (extinction.round(2) * 100), sample_names, wavelengths


def calculate_wavelength_min_max(data, wavelengths):
    wavelength_min = data.min(axis=0)
    wavelength_max = data.max(axis=0)
    return np.column_stack((wavelengths, wavelength_min, wavelength_max))


def calculate_average_std_dev(data, sample_names):
    unique_samples = np.unique(sample_names)
    std_dev_matrix = np.zeros((len(unique_samples), data.shape[1]))
    for index, sample in enumerate(unique_samples):
        std_dev_matrix[index, :] = np.std(
            data[sample_names == sample], axis=0, ddof=1
        )
    return std_dev_matrix.mean(axis=0).round(3), std_dev_matrix


def calculate_average_std_dev_measurements(data, sample_names):
    unique_samples = np.unique(sample_names)
    std_dev_matrix = np.zeros((len(unique_samples), data.shape[1]))
    group_size = 10
    for index, sample in enumerate(unique_samples):
        rows = data[sample_names == sample]
        groups = math.ceil(rows.shape[0] / group_size)
        measurement_stds = np.zeros((groups, data.shape[1]))
        for group in range(groups):
            measurement_stds[group] = np.std(
                rows[group * group_size:(group + 1) * group_size],
                axis=0,
                ddof=1,
            )
        std_dev_matrix[index] = measurement_stds.mean(axis=0).round(3)
    return std_dev_matrix.mean(axis=0).round(3), std_dev_matrix


def calculate_average_std_dev_printing(data, sample_names):
    unique_samples = np.unique(sample_names)
    std_dev_matrix = np.zeros((len(unique_samples), data.shape[1]))
    group_size = 10
    for index, sample in enumerate(unique_samples):
        rows = data[sample_names == sample]
        groups = math.ceil(rows.shape[0] / group_size)
        printing_means = np.zeros((groups, data.shape[1]))
        for group in range(groups):
            printing_means[group] = rows[
                group * group_size:(group + 1) * group_size
            ].mean(axis=0)
        std_dev_matrix[index] = np.std(printing_means, axis=0, ddof=1)
    return std_dev_matrix.mean(axis=0).round(3), std_dev_matrix


def get_optimum_sd(sd_one, sd_two, sd_three):
    return np.maximum(np.maximum(sd_one, sd_two), sd_three)


def plot_sd(sd_one, sd_two, sd_three, wavelengths):
    plt.plot(wavelengths, sd_one, color="r", label="Combined SD")
    plt.plot(wavelengths, sd_two, color="g", label="Measurement SD")
    plt.plot(wavelengths, sd_three, color="b", label="Printing SD")
    plt.title("Combined SD, Measurement SD, Printing SD vs Wavelength")
    plt.xlabel("Wavelength components")
    plt.ylabel("Standard Deviation")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


plot_SD = plot_sd


def calculate_bins_per_wavelength(min_max_array, std_dev_per_wv, scale):
    bin_edges_list = defaultdict(list)
    num_bins_per_wv = np.empty(min_max_array.shape[0], dtype=int)
    for index in range(min_max_array.shape[0]):
        wavelength, wavelength_min, wavelength_max = min_max_array[index]
        bin_width = float(scale * std_dev_per_wv[index]) or 1e-10
        num_bins = max(int(math.ceil((wavelength_max - wavelength_min) / bin_width)), 1)
        edges = np.arange(
            wavelength_min,
            wavelength_min + (num_bins + 1) * bin_width,
            bin_width,
        )
        edges[-1] = max(edges[-1], wavelength_max + 1e-10)
        bin_edges_list[wavelength] = edges.tolist()
        num_bins_per_wv[index] = num_bins
    return bin_edges_list, num_bins_per_wv


def assign_bins_per_wavelength(data, bin_edges_list):
    bin_matrix = np.empty_like(data, dtype=int)
    for index in range(data.shape[1]):
        edges = list(bin_edges_list.values())[index]
        bin_matrix[:, index] = np.clip(
            np.digitize(data[:, index], edges),
            1,
            len(edges),
        )
    return bin_matrix


def compute_binned_probability(quantized_matrix, wavelengths):
    distributions = defaultdict(lambda: defaultdict(float))
    for index, wavelength in enumerate(wavelengths):
        bins, counts = np.unique(quantized_matrix[:, index], return_counts=True)
        for bin_index, count in zip(bins, counts):
            distributions[wavelength][int(bin_index)] = count / quantized_matrix.shape[0]
    return distributions


def compute_conditional_binned_entropy(quantized_matrix, sample_names):
    conditional_entropy = {}
    for sample in np.unique(sample_names):
        curves = quantized_matrix[sample_names == sample]
        total_entropy = 0.0
        for column in curves.T:
            _, counts = np.unique(column, return_counts=True)
            probabilities = counts / len(column)
            total_entropy += sum(
                -probability * math.log2(probability)
                for probability in probabilities
                if probability > 0
            )
        conditional_entropy[sample] = total_entropy
    return conditional_entropy, np.mean(list(conditional_entropy.values()))


def compute_binned_entropy(binned_distributions, wavelengths):
    entropy_per_wavelength = []
    for wavelength in wavelengths:
        entropy = sum(
            -probability * math.log2(probability)
            for probability in binned_distributions[wavelength].values()
            if probability > 0
        )
        entropy_per_wavelength.append((wavelength, entropy))
    total_entropy = sum(value for _, value in entropy_per_wavelength)
    return (
        entropy_per_wavelength,
        total_entropy / len(entropy_per_wavelength),
        total_entropy,
    )


def compute_binned_max_entropy(num_bins_per_wv):
    return np.sum(np.log2(num_bins_per_wv))


def system_entropy_efficiency(total_entropy, max_entropy):
    if max_entropy == 0:
        return 0.0
    return (total_entropy / max_entropy) * 100.0


def calculate_input_entropy(sample_names):
    return math.log2(len(np.unique(sample_names))) if len(sample_names) else 0


def compute_binned_information_content_per_sample(
    quantized_matrix,
    binned_distributions,
    wavelengths,
    sample_names,
):
    information = np.zeros(quantized_matrix.shape[0])
    for row_index, row in enumerate(quantized_matrix):
        information[row_index] = sum(
            -math.log2(
                max(
                    binned_distributions[wavelength].get(int(row[column_index]), 1e-10),
                    1e-10,
                )
            )
            for column_index, wavelength in enumerate(wavelengths)
        )
    return np.column_stack([sample_names.astype(str), information]), float(information.mean())


def compute_binned_collision_rate_exclude_same_config(quantized_matrix, sample_names):
    config_to_curves = defaultdict(list)
    for index, name in enumerate(sample_names):
        config_to_curves[name].append(tuple(quantized_matrix[index]))
    collision_count = 0
    total_pairs = 0
    collisions_per_config = defaultdict(list)
    for first_name, second_name in itertools.combinations(config_to_curves, 2):
        first_curves = config_to_curves[first_name]
        second_curves = config_to_curves[second_name]
        threshold = min(len(first_curves), len(second_curves))
        for first_curve in first_curves[:threshold]:
            for second_curve in second_curves[:threshold]:
                total_pairs += 1
                if first_curve == second_curve:
                    collision_count += 1
                    collisions_per_config[first_name].append(second_name)
                    collisions_per_config[second_name].append(first_name)
    return {
        "total_inter_config_pairs": total_pairs,
        "collision_count": collision_count,
        "collision_rate": 100 * collision_count / total_pairs if total_pairs else 0,
        "collisions_per_config": dict(collisions_per_config),
    }


def compute_wv_entropy_diff_from_max(binned_distributions, wavelengths):
    differences = defaultdict(int)
    for wavelength in wavelengths:
        distribution = binned_distributions[wavelength]
        entropy = sum(
            -probability * math.log2(probability)
            for probability in distribution.values()
            if probability > 0
        )
        differences[wavelength] = math.log2(len(distribution)) - entropy
    return differences


def plot_json_dict_data(data, output_path):
    plt.bar(list(data.keys()), list(data.values()), color="blue", edgecolor="blue", width=1.5)
    plt.title("Differences of every component from max possible entropy")
    plt.xlabel("Wavelengths")
    plt.ylabel("Entropy differences")
    plt.xticks(fontsize=8, rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.75)
    plt.savefig(output_path)


def estimate_markov_conditional_entropy(quantized_matrix, order=1):
    sample_count, wavelength_count = quantized_matrix.shape
    entropy_per_position = np.zeros(wavelength_count)
    for position in range(wavelength_count):
        if position < order:
            counts = Counter(quantized_matrix[:, position])
            entropy_per_position[position] = sum(
                -(count / sample_count) * math.log2(count / sample_count)
                for count in counts.values()
            )
            continue
        context_counts = defaultdict(Counter)
        context_totals = Counter()
        for row in quantized_matrix:
            context = tuple(row[position - order:position])
            context_counts[context][row[position]] += 1
            context_totals[context] += 1
        entropy_per_position[position] = sum(
            (context_totals[context] / sample_count)
            * sum(
                -(count / context_totals[context])
                * math.log2(count / context_totals[context])
                for count in symbol_counts.values()
            )
            for context, symbol_counts in context_counts.items()
        )
    return entropy_per_position, entropy_per_position.sum()


def estimate_adjacent_mutual_information(quantized_matrix, wavelengths):
    if quantized_matrix.shape[1] != len(wavelengths):
        raise ValueError("quantized_matrix columns must match wavelengths")
    mutual_information = {}
    for index in range(1, quantized_matrix.shape[1]):
        joint_counts = Counter(zip(quantized_matrix[:, index - 1], quantized_matrix[:, index]))
        left_counts = Counter(quantized_matrix[:, index - 1])
        right_counts = Counter(quantized_matrix[:, index])
        total = float(quantized_matrix.shape[0])
        mutual_information[float(wavelengths[index])] = sum(
            (count / total)
            * math.log2(
                (count / total)
                / ((left_counts[left] / total) * (right_counts[right] / total))
            )
            for (left, right), count in joint_counts.items()
        )
    return mutual_information
