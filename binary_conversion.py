from __future__ import annotations
from collections import defaultdict
from itertools import combinations
import math
from typing import Any
import numpy as np
from quantization import QuantizedExperiment

class BinaryConversion:
    """Binary representation and measurements for any quantized experiment."""

    def __init__(self, quantized: QuantizedExperiment) -> None:
        self.quantized = quantized
        self.bit_width = max(1, math.ceil(math.log2(np.max(quantized.values) + 1)))
        self.outputs = self._convert()

    @property
    def bits_per_sample(self) -> int:
        return self.bit_width * self.quantized.values.shape[1]

    def _convert(self) -> dict[Any, list[str]]:
        outputs = defaultdict(list)
        for row, name in zip(self.quantized.values, self.quantized.experiment.sample_names):
            outputs[name].append("".join(format(int(value), f"0{self.bit_width}b") for value in row))
        return dict(outputs)

    def hamming_weights(self) -> dict[Any, list[int]]:
        return {
            variation: [bitstring.count("1") for bitstring in bitstrings]
            for variation, bitstrings in self.outputs.items()
        }

    def hamming_weight_differences(self, group_size: int = 10) -> dict[Any, list[float]]:
        if group_size < 1:
            raise ValueError("group_size must be at least 1")
        differences = {}
        for variation, weights in self.hamming_weights().items():
            values = np.asarray(weights, dtype=float)
            result = []
            for start in range(0, len(values), group_size):
                group = values[start:start + group_size]
                result.extend((group - np.median(group)).tolist())
            differences[variation] = result
        return differences

    def pseudorandomness(self) -> dict[Any, list[tuple[str, str]]]:
        return {
            variation: [
                (f"1 : {weight / self.bits_per_sample}",
                 f"0 : {1 - (weight / self.bits_per_sample)}")
                for weight in weights
            ]
            for variation, weights in self.hamming_weights().items()
        }

    def intra_hamming_distance_measurements(self, variation: Any, group_size: int = 10) -> dict[str, int]:
        if group_size < 1:
            raise ValueError("group_size must be at least 1")
        distances = {}
        samples = self.outputs[variation]
        for group_start in range(0, len(samples), group_size):
            group = samples[group_start:group_start + group_size]
            for first_index, first in enumerate(group):
                for second_index in range(first_index + 1, len(group)):
                    second = group[second_index]
                    first_number = group_start + first_index + 1
                    second_number = group_start + second_index + 1
                    distances[f"({first_number},{second_number})"] = sum(
                        left != right for left, right in zip(first, second)
                    )
        return distances

    def intra_hamming_distance_all_samples(self, variation: Any) -> dict[str, int]:
        """Return pairwise distances across every measurement in a variation."""
        samples = self.outputs[variation]
        distances = {}
        for first_index, first in enumerate(samples):
            for second_index in range(first_index + 1, len(samples)):
                second = samples[second_index]
                distances[f"({first_index + 1},{second_index + 1})"] = sum(
                    left != right for left, right in zip(first, second)
                )
        return distances

    def random_inter_hamming_distance(
        self,
        reference_variation: Any,
        random_seed: int | None = None,
    ) -> tuple[int, dict[Any, list[int]]]:
        """Compare one random reference sample with all samples in other variations."""
        reference_samples = self.outputs[reference_variation]
        if not reference_samples:
            raise ValueError(f"No samples found for variation {reference_variation}")

        generator = np.random.default_rng(random_seed)
        reference_index = int(generator.integers(len(reference_samples)))
        reference = reference_samples[reference_index]
        distances = {}

        for variation, samples in self.outputs.items():
            if variation == reference_variation:
                continue
            distances[variation] = [
                sum(left != right for left, right in zip(reference, sample))
                for sample in samples
            ]

        return reference_index, distances

    def random_sample_inter_hamming_distances(
        self,
        random_seed: int | None = None,
    ) -> dict[str, int]:
        """Compare one random sample from every variation across variation pairs."""
        generator = np.random.default_rng(random_seed)
        selected_samples = {
            variation: samples[int(generator.integers(len(samples)))]
            for variation, samples in self.outputs.items()
            if samples
        }
        distances = {}
        variations = list(selected_samples)

        for first_index, first_variation in enumerate(variations):
            first = selected_samples[first_variation]
            for second_variation in variations[first_index + 1:]:
                second = selected_samples[second_variation]
                distances[f"{first_variation}_vs_{second_variation}"] = sum(
                    left != right for left, right in zip(first, second)
                )

        return distances

    def random_sample_inter_hamming_distances_by_variation(
        self,
        random_seed: int | None = None,
        samples_per_group: int = 2,
        group_size: int = 10,
    ) -> dict[Any, list[int]]:
        """Return six balanced inter-configuration distances per variation pair."""
        if samples_per_group < 1:
            raise ValueError("samples_per_group must be at least 1")
        if group_size < 1:
            raise ValueError("group_size must be at least 1")

        generator = np.random.default_rng(random_seed)
        selected_samples = {
            variation: [
                samples[index]
                for group_start in range(0, len(samples), group_size)
                for index in generator.choice(
                    np.arange(group_start, min(group_start + group_size, len(samples))),
                    size=samples_per_group,
                    replace=False,
                )
            ]
            for variation, samples in self.outputs.items()
            if samples and len(samples) % group_size == 0
        }
        distances_by_variation = {variation: [] for variation in selected_samples}
        variations = list(selected_samples)

        for first_index, first_variation in enumerate(variations):
            first = selected_samples[first_variation]
            for second_variation in variations[first_index + 1:]:
                second = selected_samples[second_variation]
                distances = [
                    sum(left != right for left, right in zip(first_sample, second_sample))
                    for first_sample, second_sample in zip(first, second)
                ]
                distances_by_variation[first_variation].extend(distances)
                distances_by_variation[second_variation].extend(distances)

        return distances_by_variation

    def random_sample_inter_hamming_distances_by_pair(
        self,
        random_seed: int | None = None,
        samples_per_group: int = 2,
        group_size: int = 10,
    ) -> dict[str, list[int]]:
        """Return six balanced distances keyed by each configuration pair."""
        if samples_per_group < 1:
            raise ValueError("samples_per_group must be at least 1")
        if group_size < 1:
            raise ValueError("group_size must be at least 1")

        generator = np.random.default_rng(random_seed)
        selected_samples = {
            variation: [
                samples[index]
                for group_start in range(0, len(samples), group_size)
                for index in generator.choice(
                    np.arange(group_start, min(group_start + group_size, len(samples))),
                    size=samples_per_group,
                    replace=False,
                )
            ]
            for variation, samples in self.outputs.items()
            if samples and len(samples) % group_size == 0
        }
        variations = list(selected_samples)
        distances_by_pair = {}

        for first_index, first_variation in enumerate(variations):
            first = selected_samples[first_variation]
            for second_variation in variations[first_index + 1:]:
                second = selected_samples[second_variation]
                distances_by_pair[
                    f"{first_variation}_vs_{second_variation}"
                ] = [
                    sum(left != right for left, right in zip(first_sample, second_sample))
                    for first_sample, second_sample in zip(first, second)
                ]

        return distances_by_pair

    def inter_hamming_distance(self, first_variation: Any, second_variation: Any) -> dict[str, int]:
        distances = {}
        for first_index, first in enumerate(self.outputs[first_variation]):
            for second_index, second in enumerate(self.outputs[second_variation]):
                distances[f"({first_index + 1},{second_index + 1})"] = sum(
                    left != right for left, right in zip(first, second)
                )
        return distances

    def inter_collisions(self) -> tuple[int, int, float]:
        total_pairs = collisions = 0
        for (_, first), (_, second) in combinations(self.outputs.items(), 2):
            total_pairs += len(first) * len(second)
            collisions += sum(a == b for a in first for b in second)
        rate = 100 * collisions / total_pairs if total_pairs else 0.0
        return total_pairs, collisions, rate
