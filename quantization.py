from __future__ import annotations
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math
from typing import TYPE_CHECKING, Any
import numpy as np
from entropyops_bins import *

if TYPE_CHECKING:
    from experiment import ExperimentConfiguration


@dataclass
class QuantizedExperiment:
    #Discrete representation produced by any quantization algorithm

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
        values = np.rint(experiment.extinction * scale).astype(np.int64)
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
        average_std, _ = experiment.standard_deviation_per_config()
        bin_edges, bins_per_wavelength = calculate_bins_per_wavelength(
            min_max,
            average_std,
            self.window_size,
        )
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


class BinnedExperiment(QuantizedExperiment):
    """Backward-compatible binned representation with binning conveniences."""

    def __init__(self, experiment, window_size, bin_edges, bin_matrix, bins_per_wavelength):
        super().__init__(
            experiment=experiment,
            values=bin_matrix,
            algorithm="standard_deviation_binning",
            metadata={
                "window_size": window_size,
                "bin_edges": bin_edges,
                "bins_per_wavelength": bins_per_wavelength,
            },
        )
        self.window_size = window_size
        self.bin_edges = bin_edges
        self.bins_per_wavelength = bins_per_wavelength

    @classmethod
    def create(cls, experiment, window_size):
        quantized = StandardDeviationBinner(window_size).transform(experiment)
        return cls(
            experiment,
            window_size,
            quantized.metadata["bin_edges"],
            quantized.values,
            quantized.metadata["bins_per_wavelength"],
        )

    @property
    def distributions(self):
        return compute_binned_probability(self.values, self.experiment.wavelengths)

    @property
    def entropy(self):
        return compute_binned_entropy(self.distributions, self.experiment.wavelengths)

    @property
    def maximum_entropy(self):
        return float(compute_binned_max_entropy(self.bins_per_wavelength))

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
