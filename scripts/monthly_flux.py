import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import argparse
import os
from netcdf_io import load_data, write_dataset

if __name__ == '__main__':
    
	parser = argparse.ArgumentParser(description="sample argument parser")
	parser.add_argument("--inputs", nargs="+", required=True, help="Daily broadband flux files")
	parser.add_argument("--reference-data", required=True, help="Preprocessed file used for grid coordinates")
	parser.add_argument("--output", required=True, help="Monthly output NPZ path")
	parser.add_argument("--no-plot", action="store_true", help="Skip Cartopy diagnostic plot")
	parser.add_argument("-r","--resolution", type=int, default=2)
	parser.add_argument("-l", "--lambda-center", type=float, default=-106)
	args = parser.parse_args()
	
	input_files = args.inputs
	reference_data = args.reference_data
	output_file = args.output
	no_plot = args.no_plot
	res = args.resolution
	lambda_center = args.lambda_center

	flux_G16 = []
	flux_G18 = []

	#with open("broadband_fluxes.txt", 'r') as file:
	#	lines = file.readlines()
	#	for line in lines:
	#		line = line[:-1]
	#		flux_G16_tmp = np.load(line)['arr_0']#.flatten() 	
	#		flux_G18_tmp = np.load(line)['arr_1']#.flatten()
	#		#flux_G16_tmp[(flux_G16_tmp<50) | (flux_G16_tmp>1e03) | (lza_interp_grid_G16.flatten()>70) | (lza_interp_grid_G18.flatten()>70)] = np.nan
	#		#flux_G18_tmp[(flux_G18_tmp<50) | (flux_G18_tmp>1e03) | (lza_interp_grid_G16.flatten()>70) | (lza_interp_grid_G18.flatten()>70)] = np.nan
	#		flux_G16.append(flux_G16_tmp)
	#		flux_G18.append(flux_G18_tmp)

	#flux_G16_stacked = np.stack(flux_G16, axis=0)
	#flux_G18_stacked = np.stack(flux_G18, axis=0)

	sum_G16 = None
	sum_G18 = None
	
	count_G16 = None
	count_G18 = None
	
	# Welford variables for std(G16 - G18)
	mean_diff = None
	M2_diff = None
	count_diff = None
	
	for input_file in input_files:
		dataset = load_data(input_file)
		flux_G16_tmp = dataset["flux_G16"]
		flux_G18_tmp = dataset["flux_G18"]
			
		if sum_G16 is None:
			shape = flux_G16_tmp.shape

			# monthly means
			sum_G16 = np.zeros(shape, dtype=np.float64)
			sum_G18 = np.zeros(shape, dtype=np.float64)

			count_G16 = np.zeros(shape, dtype=np.uint16)
			count_G18 = np.zeros(shape, dtype=np.uint16)

			# Welford stddev of (G16-G18)
			mean_diff = np.zeros(shape, dtype=np.float64)
			M2_diff = np.zeros(shape, dtype=np.float64)
			count_diff = np.zeros(shape, dtype=np.uint16)

		# Monthly mean G16
		valid_G16 = ~np.isnan(flux_G16_tmp)
		sum_G16[valid_G16] += flux_G16_tmp[valid_G16]
		count_G16[valid_G16] += 1

		# Monthly mean G18
		valid_G18 = ~np.isnan(flux_G18_tmp)
		sum_G18[valid_G18] += flux_G18_tmp[valid_G18]
		count_G18[valid_G18] += 1

		# Welford std(G16-G18)
		diff_tmp = flux_G16_tmp - flux_G18_tmp
		valid_diff = ~np.isnan(diff_tmp)
		count_diff[valid_diff] += 1

		delta = diff_tmp[valid_diff] - mean_diff[valid_diff]
		mean_diff[valid_diff] += delta / count_diff[valid_diff]

		delta2 = diff_tmp[valid_diff] - mean_diff[valid_diff]
		M2_diff[valid_diff] += delta * delta2
	
	#
	# Final means
	#
	flux_G16_monthly = np.full(shape, np.nan)
	flux_G18_monthly = np.full(shape, np.nan)
	
	valid = count_G16 > 0
	flux_G16_monthly[valid] = sum_G16[valid] / count_G16[valid]
	
	valid = count_G18 > 0
	flux_G18_monthly[valid] = sum_G18[valid] / count_G18[valid]
	
	#
	# Final stddev of (G16-G18)
	#
	diff = np.full(shape, np.nan)
	
	valid = count_diff > 0
	diff[valid] = np.sqrt(M2_diff[valid] / count_diff[valid])

	#flux_G16_monthly = np.nanmean(flux_G16_stacked, axis=0)
	#flux_G18_monthly = np.nanmean(flux_G18_stacked, axis=0)

	#diff = flux_G16_monthly - flux_G18_monthly
	#print(np.nanstd(diff.flatten()))
	#print(np.nanstd(diff.flatten()/flux_G16_monthly.flatten()))
	#print(np.nanstd(diff.flatten()/flux_G18_monthly.flatten()))
	#diff = np.nanstd(flux_G16_stacked-flux_G18_stacked, axis=0)

	reference = load_data(reference_data)
	lat_interp_grid = reference["lat_interp_grid"]
	lon_interp_grid = reference["lon_interp_grid"]

	if not no_plot:
		import matplotlib.pyplot as plt
		import cartopy.crs as ccrs

		# Draw labels on original map
		min_flux = 150
		max_flux = 350
		max_diff = 25
		lon_min = -153
		lon_max = -59
		fig, axs = plt.subplots(1, 3, figsize=(22, 8), subplot_kw={'projection': ccrs.Sinusoidal(central_longitude=lambda_center)})
		pc0 = axs[0].pcolormesh(lon_interp_grid, lat_interp_grid, flux_G16_monthly, cmap='jet', transform=ccrs.PlateCarree(), vmin=min_flux, vmax=max_flux)
		axs[0].set_global()
		axs[0].set_extent([lon_min, lon_max, -90, 90], crs=ccrs.PlateCarree())
		axs[0].coastlines()
		axs[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--', xlocs=range(-180,180,30), ylocs=range(-90,90,30))
		fig.colorbar(pc0, ax=axs[0], orientation="vertical")
		pc1 = axs[1].pcolormesh(lon_interp_grid, lat_interp_grid, flux_G18_monthly, cmap='jet', transform=ccrs.PlateCarree(), vmin=min_flux, vmax=max_flux)
		axs[1].set_global()
		axs[1].set_extent([lon_min, lon_max, -90, 90], crs=ccrs.PlateCarree())
		axs[1].coastlines()
		axs[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--', xlocs=range(-180,180,30), ylocs=range(-90,90,30))
		fig.colorbar(pc1, ax=axs[1], orientation="vertical")
		pc2 = axs[2].pcolormesh(lon_interp_grid, lat_interp_grid, diff, cmap='jet', transform=ccrs.PlateCarree(), vmin=0, vmax=20)
		axs[2].set_global()
		axs[2].set_extent([lon_min, lon_max, -90, 90], crs=ccrs.PlateCarree())
		axs[2].coastlines()
		axs[2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--', xlocs=range(-180,180,30), ylocs=range(-90,90,30))
		fig.colorbar(pc2, ax=axs[2], orientation="vertical")
		fig.tight_layout()
		os.makedirs("figures", exist_ok=True)
		plt.savefig("figures/broadband_flux_monthly_res"+str(res)+"km.png")

	os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
	write_dataset(
		output_file,
		{
			"flux_G16_monthly": flux_G16_monthly,
			"flux_G18_monthly": flux_G18_monthly,
			"std_diff": diff,
			"lat": lat_interp_grid,
			"lon": lon_interp_grid,
		},
		{
			"flux_G16_monthly": ("y", "x"),
			"flux_G18_monthly": ("y", "x"),
			"std_diff": ("y", "x"),
			"lat": ("y", "x"),
			"lon": ("y", "x"),
		},
		attrs={"product": "ECO monthly broadband flux statistics"},
	)
