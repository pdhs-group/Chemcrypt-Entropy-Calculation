from experiment import ExperimentConfiguration
from quantization import (
    BinnedExperiment,
    FixedResolutionQuantizer,
    QuantizedExperiment,
    Quantizer,
    StandardDeviationBinner,
)
from binary_conversion import BinaryConversion
from entropyops_zzz import (
    build_entropy_matrix,
    build_probability_matrix,
    calculate_entropy_per_wavelength,
    calculate_information_content_per_sample,
    calculate_max_entropy,
    calculate_max_output_size_per_wavelength,
    compute_probability,
    load_data as load_raw_data,
    wavelength_value_frequency_lookup_matrix,
)
from entropyops_zzz import (
    calculate_max_output_size_per_wavelength,
    compute_probability,
)


class RawEntropyAnalysis:
    """Entropy analysis for the legacy rounded, scaled raw-value pipeline."""

    def __init__(self, experiment: ExperimentConfiguration) -> None:
        self.experiment = experiment
        self.max_output_size = calculate_max_output_size_per_wavelength(
            experiment.extinction,
            experiment.wavelengths,
        )
        self.distributions = compute_probability(
            experiment.extinction,
            self.max_output_size,
        )

    @classmethod
    def from_directory(cls, path: str) -> "RawEntropyAnalysis":
        numpy_path = path + "/numpy-files"
        extinction, sample_names, wavelengths = load_raw_data(
            numpy_path + "/extinction.npy",
            numpy_path + "/sample_name.npy",
            numpy_path + "/wavelength.npy",
        )
        return cls(ExperimentConfiguration(path.rstrip("/").split("/")[-1], extinction, sample_names, wavelengths))

    def frequency_matrix(self):
        return wavelength_value_frequency_lookup_matrix(self.experiment.extinction, self.max_output_size)

    def probability_matrix(self):
        return build_probability_matrix(self.experiment.extinction, self.distributions)

    def entropy_matrix(self):
        return build_entropy_matrix(self.experiment.extinction, self.distributions)

    def information_per_sample(self):
        return calculate_information_content_per_sample(
            self.experiment.extinction,
            self.distributions,
            self.experiment.sample_names,
        )

    def entropy_per_wavelength(self):
        return calculate_entropy_per_wavelength(self.distributions)

    @property
    def maximum_entropy(self) -> float:
        return calculate_max_entropy(self.max_output_size)
