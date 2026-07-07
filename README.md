# Chemcrypt-Entropy-Calculation
This repository includes the library to calcuate entropy and statistical analysis of Chemcrypt data. I have included the numpy files in this repository.

1. The 3 folders belong to 3 different experimental configurations.

2. Create a python environments to run the python scripts.

    python -m venv env       
    pip install -r requirements.txt
    pip list    (To verify installed packages)

This will create your python environment and install dependencies.

3. Pass Command line Arguments to the python3 execution command to choose which experimental configuration to run.
    python3 dataset-entropy-average.py CMY-4L-P81-EXT 

This command runs the average program with 2 possible arguments {One of the three copnfigurations (See Folder Names)} {Choose name of Output file}
If 2nd argument is kept blank then name is chosen by default.

4. Get output Statistics and confirmation of calculation files being created.

5. Go to the Folder of that configuration to find a sub folder which has the results of the calculations.
