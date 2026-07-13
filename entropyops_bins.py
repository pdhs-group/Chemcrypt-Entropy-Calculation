import math
from collections import defaultdict
import numpy as np
import json

def load_data(data_path,sample_name_path,wavelength_path):

    extinction = np.load(data_path) 
    sample_names = np.load(sample_name_path)
    wavelengths = np.load(wavelength_path).round(2) 
    return extinction, sample_names, wavelengths

def calculate_wavelength_min_max(extinction ,wavelengths):

    wavelength_min = extinction.min(axis=0)  
    wavelength_max = extinction.max(axis=0)  
    # Stack into a convenient 2‑D array: [wavelength, min, max]
    min_max_array = np.column_stack((wavelengths, wavelength_min, wavelength_max))
    return min_max_array


def calculate_average_std_dev(extinction, sample_names):

    unique_samples = np.unique(sample_names)
    n_wavelengths = extinction.shape[1]

    std_dev_matrix = np.zeros((len(unique_samples), n_wavelengths))
    #std_per_sample = defaultdict(lambda :defaultdict(int))

    for i, sample in enumerate(unique_samples):
        
        rows = extinction[sample_names == sample]
        
        std_per_wl = np.std(rows, axis=0, ddof=1)
        std_dev_matrix[i, :] = std_per_wl
        
        #std_per_sample[sample] = float(std_per_wl.mean())

    
    avg_std_dev_per_wv = std_dev_matrix.mean(axis=0).round(3)
    return avg_std_dev_per_wv, std_dev_matrix

def calculate_bins_per_wavelength(min_max_array, avg_std_dev_per_wv):

    n_wv = min_max_array.shape[0]
    bin_edges_list = defaultdict(list)
    #bin_edges_list = []
    num_bins_per_wv = np.empty(n_wv, dtype=int)

    for i in range(n_wv):
        wl, wl_min, wl_max = min_max_array[i]
        #bin_width = float(avg_std_dev_per_wv[i])
        bin_width = float(4 * avg_std_dev_per_wv[i])        # Range Rule of Thumb
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


def assign_bins_per_wavelength(extinction, bin_edges_list):

    n_samples, n_wv = extinction.shape
    bin_matrix = np.empty_like(extinction, dtype=int)
    for j in range(n_wv):
        edges = list(bin_edges_list.values())[j]
        
        col = extinction[:, j]
        #bin_idx = np.digitize(col, edges) - 1
        bin_idx = np.digitize(col, edges)

        num_bins = len(edges)
        bin_idx = np.clip(bin_idx, 1, num_bins)
        bin_matrix[:, j] = bin_idx
    
    return bin_matrix

def compute_binned_probability(bin_matrix, wavelengths):

    n_samples, n_wavelengths = bin_matrix.shape
    binned_distributions = defaultdict(lambda: defaultdict(float))

    for j in range(n_wavelengths):
        wl = wavelengths[j]
        col = bin_matrix[:, j]
        unique_bins, counts = np.unique(col, return_counts=True)
        for b, cnt in zip(unique_bins, counts):
            binned_distributions[wl][int(b)] = cnt / n_samples
    
    return binned_distributions


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


def compute_binned_information_content_per_sample(bin_matrix,binned_distributions,wavelengths,sample_names):

    n_samples, n_wavelengths = bin_matrix.shape
    info_per_sample = np.zeros(n_samples)
    for i in range(n_samples):
        total_info = 0.0
        for j in range(n_wavelengths):
            wl = wavelengths[j]
            bin_idx = int(bin_matrix[i, j])
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

def save_dict_to_json(data_dict, output_filename):
    with open(output_filename, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

def save_output_to_txt(data, output_filename):
    with open(output_filename, 'w') as f:
        for item in data:
            f.write(f"{item}\n")