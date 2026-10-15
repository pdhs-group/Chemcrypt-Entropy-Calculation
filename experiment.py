from __future__ import annotations
from dataclasses import dataclass, field
from os import path
import numpy as np
from numpy.typing import NDArray
from entropyops import *
from utils import *
from scipy import stats
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

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
        if self.measurements_per_print < 1:
            raise ValueError("Measurements per print should be non zero")
        
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

    def pearson_adjacent_wavelength_correlations(self, step: int) -> np.ndarray:

        if (self.measurement_count // step) < 2:
            raise ValueError(
                "At least two measurements are required for Pearson correlation"
            )

        correlations = np.full(self.wavelength_count - 1, np.nan, dtype=float)
        for wavelength_index in tqdm(range(self.wavelength_count - step),ascii=True, desc=f"Pearson Adjacent Wvs"):
            first = self.data[:, wavelength_index]
            second = self.data[:, wavelength_index + step]
            if np.ptp(first) == 0 or np.ptp(second) == 0:
                continue
            correlations[wavelength_index] = pearson_coefficient(first, second)
        return correlations

    def spearman_adjacent_wavelength_correlations(self, step: int) -> np.ndarray:
    
            if (self.measurement_count // step) < 2:
                raise ValueError(
                    "At least two measurements are required for Spearmann correlation"
                )
    
            correlations = np.full(self.wavelength_count - 1, np.nan, dtype=float)
            for wavelength_index in tqdm(range(self.wavelength_count - step),ascii=True, desc=f"Spearman Adjacent Wvs"):
                first = self.data[:, wavelength_index]
                second = self.data[:, wavelength_index + step]
                if np.ptp(first) == 0 or np.ptp(second) == 0:
                    continue
                correlations[wavelength_index] = stats.spearmanr(first, second).statistic
            return correlations

    """
    def pearson_same_print_correlations(self, group_size: int | None = None) -> dict[object, float]:
        
        group_size = self.measurements_per_print if group_size is None else group_size
        if group_size < 2:
            raise ValueError("group_size must be at least 2")
        if self.wavelength_count < 2:
            raise ValueError("At least two wavelengths are required for Pearson correlation")

        pair_indices = [
            (first, second)
            for first in range(group_size)
            for second in range(first + 1, group_size)
        ]
        correlations = {}
        for configuration in tqdm(self.variation_names):
            rows = self.data[self.sample_names == configuration]
            if len(rows) % group_size != 0:
                raise ValueError(f"Configuration {configuration!r} has {len(rows)} rows, "f"which is not divisible by group_size={group_size}")

            print_groups = rows.reshape(-1, group_size, self.wavelength_count)
            coefficients = np.empty(len(print_groups) * len(pair_indices), dtype=float)
            coefficient_index = 0
            for print_group in print_groups:
                for first, second in pair_indices:
                    coefficients[coefficient_index] = pearson_coefficient(
                        print_group[first], print_group[second]
                    )
                    coefficient_index += 1
            valid_coefficients = coefficients[~np.isnan(coefficients)]
            correlations[configuration] = (float(valid_coefficients.mean()) if valid_coefficients.size else float("nan"))

        return correlations
    """
    """
    def pearson_across_print_correlations(self,group_size: int | None = None) -> dict[object, float]:

        group_size = self.measurements_per_print if group_size is None else group_size
        if group_size < 1:
            raise ValueError("sample size must be at least 1")
        if self.wavelength_count < 2:
            raise ValueError("At least two wavelengths are required for Pearson correlation")

        correlations = {}
        for configuration in tqdm(self.variation_names):
            rows = self.data[self.sample_names == configuration]
            configuration_group_size = 1 if len(rows) < group_size else group_size
            print_groups = rows.reshape(-1, configuration_group_size, self.wavelength_count)
            group_count = len(print_groups)
            if group_count < 2:
                raise ValueError(
                    f"Configuration {configuration!r} must contain at least two print groups"
                )

            pair_count = configuration_group_size * configuration_group_size * (group_count * (group_count - 1) // 2)
            coefficients = np.empty(pair_count, dtype=float)
            coefficient_index = 0

            for first_group in range(group_count):
                for second_group in range(first_group + 1, group_count):
                    for first_row in range(configuration_group_size):
                        for second_row in range(configuration_group_size):
                            coefficients[coefficient_index] = pearson_coefficient(print_groups[first_group, first_row],print_groups[second_group, second_row])
                            coefficient_index += 1

            valid_coefficients = coefficients[~np.isnan(coefficients)]
            correlations[configuration] = (float(valid_coefficients.mean()) if valid_coefficients.size else float("nan"))

        return correlations
    """
    """
    def pearson_across_configs(self) -> dict[object, float]:
        correlations = {}

        if self.wavelength_count < 2:
            raise ValueError("At least two wavelengths are required for Pearson correlation")
        if self.variation_count < 2:
            raise ValueError("At least two configurations are required")

        configuration_means = self.mean_per_config()
        coefficient_sums = np.zeros(self.variation_count, dtype=float)
        coefficient_counts = np.zeros(self.variation_count, dtype=int)

        for first_index in tqdm(range(self.variation_count)):
            for second_index in range(first_index + 1, self.variation_count):
                coefficient = pearson_coefficient(
                    configuration_means[first_index],
                    configuration_means[second_index],
                )
                if np.isnan(coefficient):
                    continue
                coefficient_sums[first_index] += coefficient
                coefficient_sums[second_index] += coefficient
                coefficient_counts[first_index] += 1
                coefficient_counts[second_index] += 1

        for index, configuration in enumerate(self.variation_names):
            if coefficient_counts[index] > 0:
                correlations[configuration] = float(
                    coefficient_sums[index] / coefficient_counts[index]
                )
            else:
                correlations[configuration] = float("nan")

        return correlations
    """
    def kolmogorov_smirnov_normality_wvs(self,alpha: float = 0.05,random_seed: int | None = None) -> dict[object, dict[str, float | bool]]:
        """Test normality at each wavelength using all measurements."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")

        generator = np.random.default_rng(random_seed)
        results = {}
        for wavelength_index, wavelength in tqdm(enumerate(self.wavelengths),ascii=True,desc=f"KS Normality Wavelengths"):

            values = self.data[:, wavelength_index]
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite measurements at wavelength {wavelength!r}")
            
            test = stats.goodness_of_fit(stats.norm,values,statistic="ks",n_mc_samples=self.measurement_count,rng=generator)
            results[wavelength] = {"statistic": float(test.statistic),"pvalue": float(test.pvalue),"reject_null": bool(test.pvalue < alpha)}
        
        return results

    def kolmogorov_smirnov_normality_configs(self,alpha: float = 0.05,random_seed: int | None = None) -> dict[object, dict[str, float | bool]]:
        """Test normality at each wavelength using all measurements."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")

        generator = np.random.default_rng(random_seed)
        results = {}

        for unique_config in tqdm(self.variation_names,ascii=True,desc=f"KS Normality configurations"):
            rows = self.data[unique_config == (self.sample_names)]
            if not np.all(np.isfinite(rows)):
                raise ValueError(f"Non finite measurement samples") 
            
            mean_per_config = rows.mean(axis = 0)
            test = stats.goodness_of_fit(stats.norm ,mean_per_config,statistic="ks",n_mc_samples=self.wavelength_count,rng=generator)
            results[unique_config] = {"statistic": float(test.statistic),"pvalue": float(test.pvalue),"reject_null": bool(test.pvalue < alpha)}

        return results

    def shapiro_wilk_normality_wvs(self,alpha: float = 0.05) -> dict[object, dict[str, float | bool]]:
        """Test raw measurements for normality at each wavelength."""

        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.measurement_count < 3:
            raise ValueError("Shapiro-Wilk requires at least 3 measurements")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("Measurements must all be finite")

        test = stats.shapiro(self.data, axis=0)
        results = {}
        for index, wavelength in enumerate(self.wavelengths):
            results[wavelength] = {
                "statistic": float(test.statistic[index]),
                "pvalue": float(test.pvalue[index]),
                "reject_null": bool(test.pvalue[index] < alpha),
            }
        return results

    def shapiro_wilk_normality_config(self,alpha: float = 0.05)-> dict[object, dict[str, float | bool]]:
        
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.measurement_count < 3:
            raise ValueError("Shapiro-Wilk requires at least 3 measurements")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("Measurements must all be finite")

        results = {}
        for unique_config in tqdm(self.variation_names, ascii=True, desc=f"Shapiro Wilkes Config"):
            rows = self.data[unique_config == (self.sample_names)]
            if not np.all(np.isfinite(rows)):
                raise ValueError(f"Non finite measurement samples") 

            mean_per_config = rows.mean(axis = 0)
            test = stats.shapiro(mean_per_config)

            results[unique_config] = {
                "statistic": float(test.statistic),
                "pvalue": float(test.pvalue),
                "reject_null": bool(test.pvalue < alpha),
            }
        
        return results

    def anderson_darling_normality_wvs(self,alpha: float = 0.05) -> dict[object, dict[str, float | bool]]:

        """Test raw measurements for normality at each wavelength."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("Measurements must all be finite")

        results = {}
        for index, wavelength in tqdm(enumerate(self.wavelengths), ascii=True, desc=f"Anderson Darling Test Wvs"):
            test = stats.anderson(self.data[:, index],dist="norm",method="interpolate")
            results[wavelength] = {
                "statistic": float(test.statistic),
                "pvalue": float(test.pvalue),
                "reject_null": bool(test.pvalue < alpha),
            }
        return results

    def anderson_darling_normality_configs(self,alpha: float = 0.05) -> dict[object, dict[str, float | bool]]:

        """Test each configuration's mean spectrum for normality across wavelengths."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("Measurements must all be finite")

        results = {}
        for unique_config in tqdm(self.variation_names, ascii=True, desc=f"Anderson Darling Test Configs"):
            rows = self.data[self.sample_names == unique_config]
            mean_spectrum = rows.mean(axis=0)
            test = stats.anderson(mean_spectrum,dist="norm",method="interpolate")
            results[unique_config] = {
                "statistic": float(test.statistic),
                "pvalue": float(test.pvalue),
                "reject_null": bool(test.pvalue < alpha),
            }
        return results

    def mean_per_config(self) -> np.ndarray:

        unique_samples = self.variation_names
        n_wv = self.wavelength_count

        mean_per_sample = np.empty((self.variation_count, n_wv))

        for i, sample in tqdm(enumerate(unique_samples),ascii=True,desc=f"Mean per config"):

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

        for i, sample in tqdm(enumerate(unique_samples),ascii=True, desc=f"Mean per measurement"):
            
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

        for i, sample in tqdm(enumerate(unique_samples),ascii=True,desc=f"SD per config"):
            
            rows = self.data[self.sample_names == sample]
            
            std_per_wl = np.std(rows, axis=0, ddof=1)
            std_dev_matrix[i, :] = std_per_wl
            
        avg_std_dev_per_wv = std_dev_matrix.mean(axis=0).round(3)

        return avg_std_dev_per_wv
   
    def standard_deviation_per_measurements(self) -> tuple[np.ndarray, np.ndarray]:
        unique_samples = self.variation_names
        n_wavelengths = self.wavelength_count

        std_dev_matrix = np.zeros((self.variation_count, n_wavelengths))
        #std_per_sample = defaultdict(lambda :defaultdict(int))
        group_size = self.measurements_per_print

        for i, sample in tqdm(enumerate(unique_samples),ascii=True,desc=f"SD per measurement"):
            
            rows = self.data[self.sample_names == sample]
            groups = self.group_total
            measurement_avgs = np.zeros((groups,n_wavelengths))

            for group in range(groups):
                measurement_avgs[group, :] = np.std(rows[group * group_size : (group + 1) * group_size], axis=0, ddof=1)

            std_dev_matrix[i: ] = measurement_avgs.mean(axis = 0).round(3)   
            del measurement_avgs
        
        avg_std_dev_per_wv = std_dev_matrix.mean(axis=0).round(3)
        return avg_std_dev_per_wv


    def standard_deviation_per_printing(self) -> tuple[np.ndarray, np.ndarray]:

        unique_samples = self.variation_names
        n_wavelengths = self.wavelength_count

        std_dev_matrix = np.zeros((self.variation_count, n_wavelengths))
        #std_per_sample = defaultdict(lambda :defaultdict(int))
        group_size = self.measurements_per_print

        for i, sample in tqdm(enumerate(unique_samples),ascii=True,desc=f"SD per print group"):
            
            rows = self.data[self.sample_names == sample]
            groups = self.group_total
            printing_avgs = np.zeros((groups,n_wavelengths))

            for group in range(groups):
                printing_avgs[group : ] = rows[group * group_size : (group + 1) * group_size].mean(axis = 0)

            std_per_wl_per_sample = np.std(printing_avgs, axis=0, ddof=1)
            std_dev_matrix[i : ] = std_per_wl_per_sample
                
        avg_std_dev_per_wv = std_dev_matrix.mean(axis=0).round(3)
        return avg_std_dev_per_wv


    def calculate_component_min_max(self) -> np.ndarray:
        wavelength_min = self.data.min(axis=0)  
        wavelength_max = self.data.max(axis=0)  
        # Stack into a convenient 2‑D array: [wavelength, min, max]
        min_max_array = np.column_stack((self.wavelengths, wavelength_min, wavelength_max))
        return min_max_array        

    def mutual_information_adjacent_wavelengths(self,n_neighbors: int = 3,random_seed: int | None = None) -> np.ndarray:
        """Estimate continuous MI in bits for each adjacent wavelength pair."""
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")

        measurements = self.data

        if len(measurements) <= n_neighbors:
            raise ValueError("More measurements than n_neighbors are required")
        if not np.all(np.isfinite(measurements)):
            raise ValueError("Measurements must all be finite")

        mutual_information = np.zeros(self.wavelength_count - 1, dtype=float)
        for wavelength_index in tqdm(range(self.wavelength_count - 1),ascii=True,desc=f"Mutual Information Wvs - Scikit"):
            first = measurements[:, wavelength_index]
            second = measurements[:, wavelength_index + 1]
            mutual_information[wavelength_index] = mutual_info_regression(
                first.reshape(-1, 1),
                second,
                n_neighbors=n_neighbors,
                random_state=random_seed,
            )[0] / np.log(2)

        return mutual_information


    def principal_component_analysis(self,n_components: int | float = None) -> tuple[np.ndarray, PCA]:
        """Reduce raw wavelength data and return scores with the fitted PCA."""
        if not np.all(np.isfinite(self.data)):
            raise ValueError("PCA requires all measurements to be finite")

        X_scaled = StandardScaler().fit_transform(self.data)
        #pca = PCA(n_components=0.999)
        pca = PCA()

        pca.fit(X_scaled)

        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        comp_100 = np.argmax(cumulative_variance >= 0.9999) + 1
        comp_99  = np.argmax(cumulative_variance >= 0.9900) + 1
        comp_98  = np.argmax(cumulative_variance >= 0.9800) + 1

        print(f"Components for 95% Variance:  {comp_98}")
        print(f"Components for 99% Variance:  {comp_99}")
        print(f"Components for 100% Variance: {comp_100}")

        return pca.components_
        return pca.fit_transform(self.data), pca.fit(X_scaled)
    
        #reduced_data = pca.fit_transform(self.data)
        #return reduced_data, pca
