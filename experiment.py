from __future__ import annotations
from dataclasses import dataclass, field
from os import path
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from entropyops_bins import *
from utils import *

if TYPE_CHECKING:
    from quantization import QuantizedExperiment, Quantizer

@dataclass
class ExperimentConfiguration:
    #Raw measurements and metadata for one experimental configuration.
    
    name: str
    data: np.ndarray
    sample_names: np.ndarray
    wavelengths: np.ndarray
    #print_ids: np.ndarray
    measurements_per_print: int = 10
    avg_std_dev_wvs_all: NDArray[np.float64] = field(default_factory= lambda : np.empty((0, 0)))
    avg_std_dev_wvs_measurements: NDArray[np.float64] = field(default_factory= lambda : np.empty((0, 0)))
    avg_std_dev_wvs_printing: NDArray[np.float64] = field(default_factory= lambda : np.empty((0, 0)))


    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError("data must be a two-dimensional array")
        if self.sample_names.ndim != 1:
            raise ValueError("sample_names must be a one-dimensional array")
        if self.wavelengths.ndim != 1:
            raise ValueError("wavelengths must be a one-dimensional array")
        if self.data.shape[0] != len(self.sample_names):
            raise ValueError("Each data row must have one sample name")
        if self.data.shape[1] != len(self.wavelengths):
            raise ValueError("Each data column must have one wavelength")


    @classmethod
    #Call this method to create the experiment instance. The return value would be stored in an object in user interface
    def from_directory(cls, path_str: str, measurement_set = 10) -> ExperimentConfiguration:
        numpy_path = path.join(path_str, "numpy-files")
        data, sample_names, wavelengths = load_data(
            path.join(numpy_path, "extinction.npy"),
            path.join(numpy_path, "sample_name.npy"),
            path.join(numpy_path, "wavelength.npy"),
        )
        return cls(path.basename(path.normpath(path_str)),data, sample_names, wavelengths, measurement_set)

    @property
    def measurement_count(self) -> int:
        return self.data.shape[0]

    @property
    def wavelength_count(self) -> int:
        return self.data.shape[1]

    @property
    def variation_names(self) -> np.ndarray:
        return np.unique(self.sample_names)

    @property
    def variation_count(self) -> int:
        return len(self.variation_names)

    @property
    def group_total(self) -> int:
        return self.measurement_count // (self.variation_count * self.measurements_per_print)


    def mean_all_wvs(self) -> np.ndarray:

        return self.data.mean(axis = 0).round(3)


    def mean_per_config(self) -> np.ndarray:

        unique_samples = self.variation_names
        n_wv = self.wavelength_count

        mean_per_sample = np.empty((self.variation_count, n_wv))

        for i, sample in enumerate(unique_samples):

            rows = self.data[self.sample_names == sample]
            mean_per_wl = rows.mean(axis = 0)
            mean_per_sample[i, :] = mean_per_wl

        #mean_res = mean_per_sample.mean(axis = 0).round(3)

        return mean_per_sample

    def mean_per_measurement(self) -> np.ndarray:

        unique_samples = self.variation_names
        n_wavelengths = self.wavelength_count
        groups = self.group_total
        group_total = self.measurements_per_print
        mean_per_print_per_sample = np.empty((groups * self.variation_count, n_wavelengths))
        print_per_sample_name = []

        for i, sample in enumerate(unique_samples):
            
            rows = self.data[self.sample_names == sample]

            for group in range(groups):
                mean_per_print_per_sample[(i * groups) + group] = rows[group * group_total : (group + 1) * group_total].mean(axis = 0)
                print_per_sample_name.append(sample)
        
        return np.column_stack((print_per_sample_name, mean_per_print_per_sample))


    def standard_deviation_all(self) -> np.ndarray:
        return np.std(self.data, axis = 0, ddof = 1)


    def standard_deviation_per_config(self) -> tuple[np.ndarray, np.ndarray]:

        unique_samples = self.variation_names
        n_wavelengths = self.data.shape[1]

        std_dev_matrix = np.zeros((self.variation_count, n_wavelengths))
        #std_per_sample = defaultdict(lambda :defaultdict(int))

        for i, sample in enumerate(unique_samples):
            
            rows = self.data[self.sample_names == sample]
            
            std_per_wl = np.std(rows, axis=0, ddof=1)
            std_dev_matrix[i, :] = std_per_wl
            
            #std_per_sample[sample] = float(std_per_wl.mean())

        avg_std_dev_per_wv = std_dev_matrix.mean(axis=0).round(3)

        return avg_std_dev_per_wv, std_dev_matrix
        #return calculate_average_std_dev(self.data, self.sample_names)


    def standard_deviation_per_measurements(self) -> tuple[np.ndarray, np.ndarray]:
        unique_samples = self.variation_names
        n_wavelengths = self.wavelength_count

        std_dev_matrix = np.zeros((self.variation_count, n_wavelengths))
        #std_per_sample = defaultdict(lambda :defaultdict(int))
        group_size = self.measurements_per_print

        for i, sample in enumerate(unique_samples):
            
            rows = self.data[self.sample_names == sample]
            groups = self.group_total
            measurement_avgs = np.zeros((groups,n_wavelengths))

            for group in range(groups):
                measurement_avgs[group, :] = np.std(rows[group * group_size : (group + 1) * group_size], axis=0, ddof=1)

            std_dev_matrix[i: ] = measurement_avgs.mean(axis = 0).round(3)   
            del measurement_avgs
        
        avg_std_dev_per_wv = std_dev_matrix.mean(axis=0).round(3)
        return avg_std_dev_per_wv, std_dev_matrix


    def standard_deviation_per_printing(self) -> tuple[np.ndarray, np.ndarray]:

        unique_samples = self.variation_names
        n_wavelengths = self.wavelength_count

        std_dev_matrix = np.zeros((self.variation_count, n_wavelengths))
        #std_per_sample = defaultdict(lambda :defaultdict(int))
        group_size = self.measurements_per_print

        for i, sample in enumerate(unique_samples):
            
            rows = self.data[self.sample_names == sample]
            groups = self.group_total
            printing_avgs = np.zeros((groups,n_wavelengths))

            for group in range(groups):
                printing_avgs[group : ] = rows[group * group_size : (group + 1) * group_size].mean(axis = 0)

            std_per_wl_per_sample = np.std(printing_avgs, axis=0, ddof=1)
            std_dev_matrix[i : ] = std_per_wl_per_sample
                
        avg_std_dev_per_wv = std_dev_matrix.mean(axis=0).round(3)
        return avg_std_dev_per_wv, std_dev_matrix


    def calculate_component_min_max(self) -> np.ndarray:
        wavelength_min = self.data.min(axis=0)  
        wavelength_max = self.data.max(axis=0)  
        # Stack into a convenient 2‑D array: [wavelength, min, max]
        min_max_array = np.column_stack((self.wavelengths, wavelength_min, wavelength_max))
        return min_max_array        

    def quantize(self, quantizer: "Quantizer") -> "QuantizedExperiment":
        from quantization import QuantizedExperiment, Quantizer

        if not isinstance(quantizer, Quantizer):
            raise TypeError("quantizer must be a Quantizer instance")
        return quantizer.transform(self)

