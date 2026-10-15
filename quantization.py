from __future__ import annotations
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math
from typing import TYPE_CHECKING, Any
import numpy as np
from scipy import stats
from entropyops import *

if TYPE_CHECKING:
    from experiment import ExperimentConfiguration


@dataclass
class QuantizedExperiment:
    """Discrete representation produced by a quantization algorithm."""

    experiment: ExperimentConfiguration
    values: np.ndarray
    algorithm: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.values.shape != self.experiment.data.shape:
            raise ValueError("Quantized values must match the extinction data shape")
        if not np.issubdtype(self.values.dtype, np.integer):
            raise TypeError("Quantized values must contain integers")

    @property
    def Quantized_matrix(self) -> np.ndarray:
        return self.values

    @property
    def window_size(self) -> int:
        return self.metadata["window_size"]

    @property
    def bin_edges(self):
        return self.metadata["bin_edges"]

    @property
    def bins_per_wavelength(self):
        return self.metadata["bins_per_wavelength"]

    @property
    def distributions(self):
        return compute_binned_probability(self.values,self.experiment.wavelengths.astype(float))

    @property
    def max_output_size(self) -> dict[object, int]:
        return calculate_max_output_size_per_wavelength(self.values,self.experiment.wavelengths.astype(float))

    def frequency_matrix(self) -> np.ndarray:
        return wavelength_value_frequency_lookup_matrix(self.values,self.max_output_size)

    def probability_matrix(self) -> np.ndarray:
        return build_probability_matrix(self.values, self.distributions)

    def entropy_matrix(self) -> np.ndarray:
        return build_entropy_matrix(self.values, self.distributions)

    def information_per_sample(self) -> tuple[np.ndarray, float]:
        return calculate_information_content_per_sample(self.values,self.distributions,self.experiment.sample_names)

    @property
    def entropy(self):
        return compute_binned_entropy(self.distributions, self.experiment.wavelengths)

    def adjacent_mutual_information(self) -> dict[float, float]:
        """Return mutual information in bits for each adjacent wavelength pair."""
        return estimate_adjacent_mutual_information(self.values,self.experiment.wavelengths,)

    def normal_histogram_goodness_of_fit(
        self,
        alpha: float = 0.05,
        min_expected_count: float = 5.0,
    ) -> dict[object, dict[str, float | int | bool | None]]:
        """Compare each quantized wavelength histogram with fitted-normal bin counts."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if min_expected_count <= 0:
            raise ValueError("min_expected_count must be positive")

        raw_data = self.experiment.data
        if not np.all(np.isfinite(raw_data)):
            raise ValueError("Measurements must all be finite")

        results = {}
        for index, wavelength in enumerate(self.experiment.wavelengths):
            values = raw_data[:, index]
            observed_symbols = self.values[:, index]
            sample_count = len(values)
            mean = float(np.mean(values))
            standard_deviation = float(np.std(values, ddof=0))

            if standard_deviation == 0:
                results[wavelength] = {
                    "statistic": float("nan"),
                    "pvalue": float("nan"),
                    "degrees_of_freedom": 0,
                    "groups_after_merging": 0,
                    "reject_null": None,
                }
                continue

            if self.algorithm == "fixed_resolution":
                scale = 10 ** self.metadata["decimal_places"]
                minimum_symbol = int(observed_symbols.min())
                maximum_symbol = int(observed_symbols.max())
                symbols = np.arange(minimum_symbol, maximum_symbol + 1)
                observed, _ = np.histogram(
                    observed_symbols,
                    bins=np.arange(minimum_symbol - 0.5, maximum_symbol + 1.5),
                )
                boundaries = np.arange(
                    minimum_symbol - 0.5,
                    maximum_symbol + 1.5,
                ) / scale
                probabilities = np.diff(
                    np.concatenate(([0.0], stats.norm.cdf(boundaries, mean, standard_deviation), [1.0]))
                )
                observed = np.concatenate(([0], observed, [0]))
                expected = sample_count * probabilities
            elif self.algorithm == "standard_deviation_binning":
                edges = np.asarray(list(self.bin_edges.values())[index], dtype=float)
                symbol_count = len(edges)
                observed = np.bincount(
                    observed_symbols.astype(int) - 1,
                    minlength=symbol_count,
                )[:symbol_count]
                cdf_values = stats.norm.cdf(edges, mean, standard_deviation)
                probabilities = np.concatenate(
                    ([cdf_values[1]], np.diff(cdf_values[1:]), [1.0 - cdf_values[-1]])
                )
                expected = sample_count * probabilities
            else:
                raise ValueError(f"Unsupported quantization algorithm: {self.algorithm}")

            grouped_observed = []
            grouped_expected = []
            observed_total = 0.0
            expected_total = 0.0
            for observed_count, expected_count in zip(observed, expected):
                observed_total += observed_count
                expected_total += expected_count
                if expected_total >= min_expected_count:
                    grouped_observed.append(observed_total)
                    grouped_expected.append(expected_total)
                    observed_total = 0.0
                    expected_total = 0.0

            if expected_total > 0 and grouped_expected:
                grouped_observed[-1] += observed_total
                grouped_expected[-1] += expected_total
            elif expected_total > 0:
                grouped_observed.append(observed_total)
                grouped_expected.append(expected_total)

            degrees_of_freedom = len(grouped_expected) - 3
            if degrees_of_freedom > 0:
                statistic = float(
                    np.sum(
                        (np.asarray(grouped_observed) - grouped_expected) ** 2
                        / grouped_expected
                    )
                )
                pvalue = float(stats.chi2.sf(statistic, degrees_of_freedom))
                reject_null = bool(pvalue < alpha)
            else:
                statistic = float("nan")
                pvalue = float("nan")
                reject_null = None

            results[wavelength] = {
                "statistic": statistic,
                "pvalue": pvalue,
                "degrees_of_freedom": degrees_of_freedom,
                "groups_after_merging": len(grouped_expected),
                "reject_null": reject_null,
            }
        return results

    @property
    def maximum_entropy(self):
        if "bins_per_wavelength" in self.metadata:
            return float(compute_binned_max_entropy(self.bins_per_wavelength))
        return float(calculate_max_entropy(self.max_output_size))

    def intra_min_entropy(self, group_size=10):
        if group_size < 1:
            raise ValueError("group_size must be at least 1")
        results = {}
        for variation in self.experiment.variation_names:
            rows = self.values[self.experiment.sample_names == variation]
            for group_index, start in enumerate(range(0, len(rows), group_size)):
                group = rows[start:start + group_size]
                entropy = sum(
                    -math.log2(Counter(column).most_common(1)[0][1] / len(column))
                    for column in group.T
                )
                results[f"{variation}_g{group_index}"] = entropy
        return results

    def inter_collisions(self):
        grouped = defaultdict(list)
        for name, row in zip(self.experiment.sample_names, self.values):
            grouped[name].append(tuple(row))
        total_pairs = collisions = 0
        for (_, first), (_, second) in combinations(grouped.items(), 2):
            total_pairs += len(first) * len(second)
            collisions += sum(a == b for a in first for b in second)
        return total_pairs, collisions


class Quantizer(ABC):
    """Strategy interface for converting measurements into integer symbols."""

    @abstractmethod
    def transform(self, experiment: ExperimentConfiguration) -> QuantizedExperiment:
        raise NotImplementedError


class FixedResolutionQuantizer(Quantizer):
    """Quantize values to a fixed decimal resolution, such as 0.01."""

    def __init__(self, decimal_places: int = 2) -> None:
        if decimal_places < 0:
            raise ValueError("decimal_places must be non-negative")
        self.decimal_places = decimal_places

    def transform(self, experiment: ExperimentConfiguration) -> QuantizedExperiment:
        scale = 10 ** self.decimal_places
        values = np.rint(experiment.data * scale).astype(np.int64)
        return QuantizedExperiment(
            experiment=experiment,
            values=values,
            algorithm="fixed_resolution",
            metadata={"decimal_places": self.decimal_places},
        )


class StandardDeviationBinner(Quantizer):
    """Quantize using wavelength-specific standard-deviation bin widths."""

    def __init__(self, window_size: int = 1) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size = window_size

    def transform(self, experiment: ExperimentConfiguration) -> QuantizedExperiment:
        min_max = calculate_wavelength_min_max(experiment.data, experiment.wavelengths)
        average_std = experiment.standard_deviation_all()
        bin_edges, bins_per_wavelength = calculate_bins_per_wavelength(min_max,average_std,self.window_size)
        values = assign_bins_per_wavelength(experiment.data, bin_edges)
        return QuantizedExperiment(
            experiment=experiment,
            values=values,
            algorithm="standard_deviation_binning",
            metadata={
                "window_size": self.window_size,
                "bin_edges": bin_edges,
                "bins_per_wavelength": bins_per_wavelength,
            },
        )

