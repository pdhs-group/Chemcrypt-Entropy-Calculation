import pandas as pd
import sys
import shutil
import os
import json
import matplotlib.pyplot as plt

def init(op_folder: str, subfolder: str):
    
    output_path = f"{op_folder}/{subfolder}"

    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
        print(f"Clearing previous cache....")

    os.mkdir(output_path)
    print(f"\nOutput folder created in {op_folder} ")
    return output_path

def save_to_csv(data_array, subfolder, filename, idx = None, col = None):

    data_df = pd.DataFrame(data_array, index= idx, columns= col)
    data_df.to_csv(f"{subfolder}/{filename}.csv", index= idx is not None, header=col is not None)
    print(f"{filename}.csv - File created")

def save_dict_to_json(data_dict, output_filename):
    with open(output_filename, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

def save_output_to_txt(data, output_filename):
    with open(output_filename, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

def plot_series(X_axis,*series,labels=None,colors=None,title="Values vs Wavelength",xlabel = "Data",ylabel="Value",save_path=None):

    if not series:
        raise ValueError("At least one series is required")
    if labels is not None and len(labels) != len(series):
        raise ValueError("labels must match the number of series")
    if colors is not None and len(colors) != len(series):
        raise ValueError("colors must match the number of series")

    plt.figure()

    for index, series_data in enumerate(series):
        if len(series_data) == 2:
            values, series_label = series_data
            color = None if colors is None else colors[index]
        elif len(series_data) == 3:
            values, series_label, color = series_data
        else:
            raise ValueError("Each series must contain values and a label")

        if labels is not None:
            series_label = labels[index]
        plt.plot(X_axis, values, label=series_label, color=color)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return True
    else:
        plt.show()
        return False


def plot_wavelength_component(task):
	index, sample_names, values, wavelength, save_subdir = task
	plot_series(
		sample_names,
		(values, f"{wavelength} mm component", "b"),
		title=f"{wavelength} mm component distribution",
        xlabel="Sample_Names",
		ylabel="Values",
		save_path=f"{save_subdir}/{wavelength} distribution.png",
	)
	return index, wavelength
