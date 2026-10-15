# Chemcrypt-Entropy-Calculation
This repository includes the library to calcuate entropy and statistical analysis of Chemcrypt data. I have included the numpy files in this repository.

1. The 3 folders belong to 3 different experimental configurations.

2. Create a python environments to run the python scripts.

    python -m venv env       
    pip install -r requirements.txt
    pip list    (To verify installed packages)

This will create your python environment and install dependencies.

3. Run entropy analysis with standard-deviation binning (default):

    python dataset-entropy-binning.py CMY-4L-P81-EXT

To use fixed-resolution quantization instead:

    python dataset-entropy-binning.py CMY-4L-P81-EXT --method fixed-resolution --decimal-places 2

4. Get output Statistics and confirmation of calculation files being created.

5. Go to the Folder of that configuration to find a sub folder which has the results of the calculations.

## Object-oriented API

The reusable domain classes are defined in their respective modules. Quantization
is selected independently from entropy and binary analysis:

```python
from experiment import ExperimentConfiguration
from quantization import FixedResolutionQuantizer, StandardDeviationBinner
from binary_conversion import BinaryConversion

experiment = ExperimentConfiguration.from_directory("CMY-4L-P81-EXT")

# Existing wavelength-specific statistical binning
quantizer = StandardDeviationBinner(window_size=1)
quantized = quantizer.transform(experiment)

# Fixed 0.01 resolution quantization
# quantizer = FixedResolutionQuantizer(decimal_places=2)
# quantized = quantizer.transform(experiment)

binary = BinaryConversion(quantized)
print(binary.bits_per_sample)
print(binary.inter_collisions())
```

Both quantizers produce the same `QuantizedExperiment` interface, so additional
algorithms can be added by implementing `Quantizer.transform()`.
