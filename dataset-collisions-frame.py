from utils import save_dict_to_json
from experiment import ExperimentConfiguration
from quantization import StandardDeviationBinner
import sys
import shutil
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

config = sys.argv[1] if len(sys.argv) > 1 else exit(1)
noise_factor = list(range(1,20))
collisons_per_window_size = defaultdict(int)
intra_entropy_per_window_size = defaultdict(int)
total_system_entropy_per_window_size = defaultdict(int)
max_entropy_per_window_size = defaultdict(int)

output_path = f"{config}/calculation_entropy_frame"

if os.path.isdir(output_path):
    shutil.rmtree(output_path)
    print(f"Clearing previous cache....")

os.mkdir(output_path)
print(f"\nOutput folder created in {config} ")

experiment = ExperimentConfiguration.from_directory(config)

print("\nCalculating MCV min entropy / Intra entropy")
for window_S in tqdm(noise_factor):
    binned = StandardDeviationBinner(window_S).transform(experiment)
    intra_entropy_per_window_size[window_S] = np.mean(
        list(binned.intra_min_entropy().values())
    )

print("\nCalculating collisions")
for window_S in tqdm(noise_factor):
    binned = StandardDeviationBinner(window_S).transform(experiment)
    collisons_per_window_size[window_S] = binned.inter_collisions()[1]

print("\nCalculating Max Entropy and System Entropy")
for window_S in tqdm(noise_factor):
    binned = StandardDeviationBinner(window_S).transform(experiment)
    max_entropy_per_window_size[window_S] = binned.maximum_entropy
    total_system_entropy_per_window_size[window_S] = binned.entropy[2]

    
save_dict_to_json(collisons_per_window_size,f"{output_path}/collisions_per_S")
print("\nSaved Inter collisions to JSON")

save_dict_to_json(intra_entropy_per_window_size,f"{output_path}/Intra_entropy_per_S")
print("Saved Intra Entropy to JSON")

save_dict_to_json(max_entropy_per_window_size,f"{output_path}/Max_entropy_per_S")
print("Saved Max Entropy per Window to JSON")

save_dict_to_json(total_system_entropy_per_window_size,f"{output_path}/System_entropy_per_S")
print("Saved Total System Entropy per Window to JSON")


plt.plot(list(intra_entropy_per_window_size.keys()), list(intra_entropy_per_window_size.values()), color = 'r')
plt.plot(list(collisons_per_window_size.keys()), list(collisons_per_window_size.values()), color = 'g' )
#plt.plot(list(collisons_per_window_size.keys()), list(total_max_entropy_distance_per_window_size.values()), color = 'b')
plt.xticks(noise_factor)
plt.grid(True,which="both")
plt.title("Intra Entropy and Inter Collisions vs Noise Factor (Window Size)")
plt.xlabel("Noise Factor (S)")
plt.ylabel("Intra Entropy + Inter collisions")
plt.xticks(rotation=45)
plt.savefig(f"{output_path}/Intra_entropy_and_Collisions_plot")
print("Saved Intra Entropy and collisions vs Sample window plot")
plt.clf()

plt.plot(list(intra_entropy_per_window_size.keys()), list(max_entropy_per_window_size.values()), color = 'r')
plt.plot(list(collisons_per_window_size.keys()), list(total_system_entropy_per_window_size.values()), color = 'g' )

plt.xticks(noise_factor)
plt.grid(True,which="both")
plt.title("Max Entropy and System entropy vs Noise Factor (Window Size)")
plt.xlabel("Noise Factor (S)")
plt.ylabel("Max Entropy + Inter Entropy")
plt.xticks(rotation=45)
plt.savefig(f"{output_path}/Max_entropy_and_Inter_entropy_plot")
print("Saved Max Entropy and Inter Entropy vs Sample window plot")

