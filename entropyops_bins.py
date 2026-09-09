import math
from collections import defaultdict, Counter
import numpy as np
import json
import itertools
import matplotlib.pyplot as plt

def load_data(data_path,sample_name_path,wavelength_path):

    extinction = np.load(data_path) 
    sample_names = np.load(sample_name_path)
    wavelengths = np.load(wavelength_path).round(2) 
    return extinction, sample_names, wavelengths

def calculate_wavelength_min_max(data ,wavelengths):

    wavelength_min = data.min(axis=0)  
    wavelength_max = data.max(axis=0)  
    # Stack into a convenient 2‑D array: [wavelength, min, max]
    min_max_array = np.column_stack((wavelengths, wavelength_min, wavelength_max))
    return min_max_array

def calculate_average_std_dev(data, sample_names):

    unique_samples = np.unique(sample_names)
    n_wavelengths = data.shape[1]

    std_dev_matrix = np.zeros((len(unique_samples), n_wavelengths))
    #std_per_sample = defaultdict(lambda :defaultdict(int))

    for i, sample in enumerate(unique_samples):
        
        rows = data[sample_names == sample]
        
        std_per_wl = np.std(rows, axis=0, ddof=1)
        std_dev_matrix[i, :] = std_per_wl
        
        #std_per_sample[sample] = float(std_per_wl.mean())

    avg_std_dev_per_wv = std_dev_matrix.mean(axis=0).round(3)
    return avg_std_dev_per_wv, std_dev_matrix

def calculate_average_std_dev_measurements(data, sample_names):

    unique_samples = np.unique(sample_names)
    n_wavelengths = data.shape[1]

    std_dev_matrix = np.zeros((len(unique_samples), n_wavelengths))
    #std_per_sample = defaultdict(lambda :defaultdict(int))
    group_size = 10

    for i, sample in enumerate(unique_samples):
        
        rows = data[sample_names == sample]
        groups = math.ceil(rows.shape[0] / group_size)
        measurement_avgs = np.zeros((groups,n_wavelengths))

        for group in range(groups):
        
            std_per_wl_per_group = np.std(rows[group * group_size : (group + 1) * group_size], axis=0, ddof=1)
            measurement_avgs[group, :] = std_per_wl_per_group

        std_dev_matrix[i: ] = measurement_avgs.mean(axis = 0).round(3)   
        del measurement_avgs
    
    avg_std_dev_per_wv = std_dev_matrix.mean(axis=0).round(3)
    return avg_std_dev_per_wv, std_dev_matrix


def calculate_average_std_dev_printing(data, sample_names):

    unique_samples = np.unique(sample_names)
    n_wavelengths = data.shape[1]

    std_dev_matrix = np.zeros((len(unique_samples), n_wavelengths))
    #std_per_sample = defaultdict(lambda :defaultdict(int))
    group_size = 10

    for i, sample in enumerate(unique_samples):
        
        rows = data[sample_names == sample]
        groups = math.ceil(rows.shape[0] / group_size)
        printing_avgs = np.zeros((groups,n_wavelengths))

        for group in range(groups):
            printing_avgs[group : ] = rows[group * group_size : (group + 1) * group_size].mean(axis = 0)

        std_per_wl_per_sample = np.std(printing_avgs, axis=0, ddof=1)
        std_dev_matrix[i : ] = std_per_wl_per_sample
            
    avg_std_dev_per_wv = std_dev_matrix.mean(axis=0).round(3)
    return avg_std_dev_per_wv, std_dev_matrix


def get_optimum_sd(SD1,SD2,SD3):

    return np.maximum(np.maximum(SD1,SD2),SD3)

def plot_SD (SD1,SD2,SD3,WVs):
    plt.plot(WVs, SD1, color='r', label='Combined SD')
    plt.plot(WVs, SD2, color='g', label='Measurement SD')
    plt.plot(WVs, SD3, color='b', label='Printing SD')

    #plt.xticks(WVs)
    #plt.grid(True, which="both")
    plt.title("Combined SD, Measurement SD, Printing SD vs Wavelength")
    plt.xlabel("Wavelength components")
    plt.ylabel("Standard Deviation")
    plt.xticks(rotation=45)
    plt.legend()

    plt.show()


def calculate_bins_per_wavelength(min_max_array, std_dev_per_wv, S):

    n_wv = min_max_array.shape[0]
    bin_edges_list = defaultdict(list)
    #bin_edges_list = []
    num_bins_per_wv = np.empty(n_wv, dtype=int)

    for i in range(n_wv):
        wl, wl_min, wl_max = min_max_array[i]
        #bin_width = float(avg_std_dev_per_wv[i])
        bin_width = float(S * std_dev_per_wv[i])        
        if bin_width == 0:
            bin_width = 1e-10
        num_bins = int(math.ceil((wl_max - wl_min) / bin_width))
        
        num_bins = max(num_bins, 1)
        
        edges = np.arange(wl_min, wl_min + (num_bins + 1) * bin_width, bin_width)
        edges[-1] = max(edges[-1], wl_max + 1e-10)
        bin_edges_list[wl] = edges.tolist()
        #bin_edges_list.append(edges)
        num_bins_per_wv[i] = num_bins

    return bin_edges_list, num_bins_per_wv


def assign_bins_per_wavelength(data, bin_edges_list):

    n_wv = data.shape[1]
    bin_matrix = np.empty_like(data, dtype=int)
    for j in range(n_wv):
        edges = list(bin_edges_list.values())[j]
        
        col = data[:, j]
        #bin_idx = np.digitize(col, edges) - 1
        bin_idx = np.digitize(col, edges)

        num_bins = len(edges)
        bin_idx = np.clip(bin_idx, 1, num_bins)
        bin_matrix[:, j] = bin_idx
    
    return bin_matrix

def compute_binned_probability(QT_matrix, wavelengths):

    n_samples, n_wavelengths = QT_matrix.shape
    binned_distributions = defaultdict(lambda: defaultdict(float))

    for j in range(n_wavelengths):
        wl = wavelengths[j]
        col = QT_matrix[:, j]
        unique_bins, counts = np.unique(col, return_counts=True)
        for b, cnt in zip(unique_bins, counts):
            binned_distributions[wl][int(b)] = cnt / n_samples
    
    return binned_distributions

def compute_conditional_binned_entropy(QT_matrix, sample_names):
    unique_samples = np.unique(sample_names)
    cond_entropy = {}

    for config in unique_samples:
        indices = (sample_names == config)
        bin_curves = QT_matrix[indices]  # shape (30, 1065)
        total_entropy = 0.0

        for j in range(bin_curves.shape[1]):
            bins_at_wl = bin_curves[:, j]
            _, counts = np.unique(bins_at_wl, return_counts=True)

            total = len(bins_at_wl)
            probs = [c / total for c in counts]
            H_wl = sum(-p * math.log2(p) if p > 0 else 0 for p in probs)
            total_entropy += H_wl

        cond_entropy[config] = total_entropy

    return cond_entropy, np.mean(list(cond_entropy.values()))

def compute_binned_entropy(binned_distributions, wavelengths):

    entropy_per_wavelength = []
    for wl in wavelengths:
        dist = binned_distributions[wl]
        H = sum(-p * math.log2(p) for p in dist.values() if p > 0)
        entropy_per_wavelength.append((wl, H))
    total_entropy = sum(e for _, e in entropy_per_wavelength)
    average_entropy = total_entropy / len(entropy_per_wavelength)

    return entropy_per_wavelength, average_entropy, total_entropy


def compute_binned_max_entropy(num_bins_per_wv):

    return np.sum(np.log2(num_bins_per_wv))


def compute_binned_information_content_per_sample(QT_matrix,binned_distributions,wavelengths,sample_names):

    n_samples, n_wavelengths = QT_matrix.shape
    info_per_sample = np.zeros(n_samples)
    for i in range(n_samples):
        total_info = 0.0
        for j in range(n_wavelengths):
            wl = wavelengths[j]
            bin_idx = int(QT_matrix[i, j])
            p = binned_distributions[wl].get(bin_idx, 1e-10)
            total_info += -math.log2(max(p, 1e-10))
        info_per_sample[i] = total_info
    average_info = float(info_per_sample.mean())
    combined = np.column_stack([sample_names.astype(str), info_per_sample])
    return combined, average_info


def system_entropy_efficiency(total_entropy, max_entropy) -> float:

    if max_entropy == 0:
        return 0.0
    return (total_entropy / max_entropy) * 100.0

def calculate_input_entropy(sample_names):
    unique_samples, counts = np.unique(sample_names, return_counts=True)
    total_samples = len(unique_samples)
    
    return math.log2(total_samples) if total_samples > 0 else 0

def compute_binned_collision_rate_exclude_same_config(QT_matrix, sample_names):

    config_to_curves = defaultdict(list)
    for i, name in enumerate(sample_names):

        config_to_curves[name].append(tuple(QT_matrix[i]))

    unique_configs = list(config_to_curves.keys())
    total_inter_config_pairs = 0
    collision_count = 0
    collisions_per_config = defaultdict(list)

    for name1, name2 in itertools.combinations(unique_configs, 2):
        curves1 = config_to_curves[name1]
        curves2 = config_to_curves[name2]

        threshold = min(len(curves1), len(curves2))

        for c1 in curves1[:threshold]:
            for c2 in curves2[:threshold]:
                total_inter_config_pairs += 1
                if c1 == c2:
                    collision_count += 1
                    collisions_per_config[name1].append(name2)
                    collisions_per_config[name2].append(name1)

    collision_rate = (
        (collision_count / total_inter_config_pairs * 100)
        if total_inter_config_pairs > 0 else 0
    )

    return {
        "total_inter_config_pairs": total_inter_config_pairs,
        "collision_count": collision_count,
        "collision_rate": collision_rate,
        "collisions_per_config": dict(collisions_per_config)
    }


def compute_wv_entropy_diff_from_max(binned_distributions, wavelengths):

    data = defaultdict(int)

    for wl in wavelengths:
        dist = binned_distributions[wl]
        H = sum(-p * math.log2(p) for p in dist.values() if p > 0)

        H_max = math.log2(len(list(dist.keys())))

        data[wl] = H_max - H

    return data

def save_dict_to_json(data_dict, output_filename):
    with open(output_filename, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

def save_output_to_txt(data, output_filename):
    with open(output_filename, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

def plot_json_dict_data(data, output_path):

    plt.bar(list(data.keys()), list(data.values()), color = 'blue', edgecolor = 'blue', width=1.5)
    plt.title(f"Differences of every component from max possible entropy")
    plt.xlabel(f'Wavelengths')
    plt.ylabel('Entropy differences from max')
    plt.xticks(fontsize=8, rotation=45, ha='right')

    plt.grid(axis='y', linestyle = '--', alpha = 0.75)

    plt.savefig(output_path)

def estimate_markov_conditional_entropy(QT_matrix, m=1):

    N, L = QT_matrix.shape

    H_per_position = np.zeros(L)
    for k in range(L):

        if k < m:
            counts = Counter(QT_matrix[:, k])
            total = float(N)
            H = 0.0
            for c in counts.values():
                p = c / total
                H -= p * math.log2(p)
            H_per_position[k] = H
            continue

        context_counts = defaultdict(Counter)
        context_totals = Counter()
        for row in QT_matrix:
            context = tuple(row[k-m:k])  # previous m bins
            symbol = row[k]
            context_counts[context][symbol] += 1
            context_totals[context] += 1

        
        Hk = 0.0
        for context, sym_counts in context_counts.items():
            total_context = context_totals[context]
            p_context = total_context / N
            H_context = 0.0
            for c in sym_counts.values():
                p = c / total_context
                H_context -= p * math.log2(p)
            Hk += p_context * H_context
        H_per_position[k] = Hk

    return H_per_position, H_per_position.sum()


def estimate_adjacent_mutual_information(QT_matrix, wavelengths):
    """Estimate mutual information between adjacent wavelength bins."""
    if QT_matrix.shape[1] != len(wavelengths):
        raise ValueError("QT_matrix columns must match wavelengths")

    mutual_information = {}
    for j in range(1, QT_matrix.shape[1]):
        joint_counts = Counter(zip(QT_matrix[:, j - 1], QT_matrix[:, j]))
        left_counts = Counter(QT_matrix[:, j - 1])
        right_counts = Counter(QT_matrix[:, j])
        total = float(QT_matrix.shape[0])

        value = 0.0
        for (left, right), count in joint_counts.items():
            joint_probability = count / total
            left_probability = left_counts[left] / total
            right_probability = right_counts[right] / total
            value += joint_probability * math.log2(
                joint_probability / (left_probability * right_probability)
            )

        mutual_information[float(wavelengths[j])] = value

    return mutual_information