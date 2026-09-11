from entropyops import *
from experiment import ExperimentConfiguration
from utils import *
from concurrent.futures import ProcessPoolExecutor
import os


def main():
	config = sys.argv[1] if len(sys.argv) > 1 else exit(1)
	get_raw_wv_data = sys.argv[2] if len(sys.argv) > 2 else "n"

	res_loc = init(config,f"Raw_Data_Analysis")

	experiment = ExperimentConfiguration.from_directory(config)

	if get_raw_wv_data.upper() == "Y":
		wv_comps = init(config,f"WV_Raw_data")
		tasks = [
			(i, experiment.sample_names, experiment.data[:, i], experiment.wavelengths[i], wv_comps)
			for i in range(experiment.wavelength_count)
		]
		max_workers = min(os.cpu_count() or 1, experiment.wavelength_count)
		with ProcessPoolExecutor(max_workers=max_workers) as executor:
			for index, wavelength in executor.map(plot_wavelength_component, tasks, chunksize=8):
				print(f"Wavelength {wavelength}mm plot saved...")
		return

	min_max_array = experiment.calculate_component_min_max()
	save_to_csv(min_max_array,res_loc,"Min_Max_per_wv",col=["wv","min","max"])

	mean_per_wv = experiment.mean_all_wvs()
	save_to_csv(mean_per_wv, res_loc, "Mean_per_wv", idx= experiment.wavelengths.T)

	mean_per_sample= experiment.mean_per_config()
	save_to_csv(mean_per_sample ,res_loc, "Mean_per_input_sample", idx= experiment.variation_names, col=experiment.wavelengths)

	mean_per_print_per_sample = experiment.mean_per_measurement()
	save_to_csv(mean_per_print_per_sample, res_loc, "Mean_per_print_per_sample")

	std_all_data = experiment.standard_deviation_all()
	save_to_csv(std_all_data, res_loc, "Standard_Dev_combined", idx=experiment.wavelengths.T)

	std_per_config = experiment.standard_deviation_per_config()
	save_to_csv(std_per_config[0], res_loc, "Average_SD_per_config", idx= experiment.wavelengths.T)
	save_to_csv(std_per_config[1], res_loc, "Standard_Dev_per_config", idx=experiment.variation_names, col=experiment.wavelengths)

	std_per_measurements = experiment.standard_deviation_per_measurements()
	save_to_csv(std_per_measurements[0], res_loc, "Average_SD_per_measurements", idx= experiment.wavelengths.T)
	save_to_csv(std_per_measurements[1], res_loc, "Standard_Dev_per_measurements", idx=experiment.variation_names, col=experiment.wavelengths)

	std_per_printing = experiment.standard_deviation_per_printing()
	save_to_csv(std_per_printing[0], res_loc, "Average_SD_per_printing", idx= experiment.wavelengths.T)
	save_to_csv(std_per_printing[1], res_loc, "Standard_Dev_per_printing", idx=experiment.variation_names, col=experiment.wavelengths)

	plot_status = plot_series(
		experiment.wavelengths,
		(std_all_data, "Combined SD", "m"),
		(std_per_config[0], "Config SD", "r"),
		(std_per_measurements[0], "Measurement SD", "g"),
		(std_per_printing[0], "Printing SD", "b"),
		title="Standard Deviation vs Wavelength",
		ylabel="Standard Deviation",
		save_path=f"{res_loc}/Standard_Deviations vs Wavelength.png",
	)

	print(f"Standard Deviation vs Wavelength Plot saved") if plot_status else print(f"Standard Deviation vs Wavelength Plot not saved...")


if __name__ == "__main__":
	main()

