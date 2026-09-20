import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm
import time
import math
import argparse
import joblib
from netcdf_io import load_data, write_dataset

def build_arrays(input_file):
	# -----------------------------
	# Retrieve data
	# -----------------------------
	dataset = load_data(input_file)
	#width = npzfile['arr_0']
	#height = npzfile['arr_1']
	width = dataset['width']
	height = dataset['height']
	lat_interp_grid = dataset['lat_interp_grid']
	lon_interp_grid = dataset['lon_interp_grid']

	lza_interp_grid_G16 = np.cos(np.radians(dataset['lza_G16_interp'])).flatten()
	lza_interp_grid_G18 = np.cos(np.radians(dataset['lza_G18_interp'])).flatten()

	BT_C08_interp_G16 = dataset['BT_G16_interp'][:, :, 0].flatten()
	BT_C11_interp_G16 = dataset['BT_G16_interp'][:, :, 3].flatten()
	BT_C12_interp_G16 = dataset['BT_G16_interp'][:, :, 4].flatten()
	BT_C14_interp_G16 = dataset['BT_G16_interp'][:, :, 6].flatten()
	BT_C15_interp_G16 = dataset['BT_G16_interp'][:, :, 7].flatten()
	BT_C16_interp_G16 = dataset['BT_G16_interp'][:, :, 8].flatten()

	BT_C08_interp_G18 = dataset['BT_G18_interp'][:, :, 0].flatten()
	BT_C11_interp_G18 = dataset['BT_G18_interp'][:, :, 3].flatten()
	BT_C12_interp_G18 = dataset['BT_G18_interp'][:, :, 4].flatten()
	BT_C14_interp_G18 = dataset['BT_G18_interp'][:, :, 6].flatten()
	BT_C15_interp_G18 = dataset['BT_G18_interp'][:, :, 7].flatten()
	BT_C16_interp_G18 = dataset['BT_G18_interp'][:, :, 8].flatten()

	# Define averages
	BT_C08_av = (BT_C08_interp_G16 + BT_C08_interp_G18)/2
	BT_C11_av = (BT_C11_interp_G16 + BT_C11_interp_G18)/2
	BT_C12_av = (BT_C12_interp_G16 + BT_C12_interp_G18)/2
	BT_C14_av = (BT_C14_interp_G16 + BT_C14_interp_G18)/2
	BT_C15_av = (BT_C15_interp_G16 + BT_C15_interp_G18)/2
	BT_C16_av = (BT_C16_interp_G16 + BT_C16_interp_G18)/2
	BT_diff1 = BT_C14_av - BT_C11_av
	BT_diff2 = BT_C14_av - BT_C15_av
	BT_diff3 = BT_C14_av - BT_C08_av
	BT_diff4 = BT_C14_av - BT_C16_av
	
	BT = np.stack([
	    BT_C08_av, BT_C11_av, BT_C12_av, BT_C14_av, BT_C15_av, BT_C16_av,
	    BT_diff1, BT_diff2, BT_diff3, BT_diff4
	], axis=-1).reshape(-1, 10)
	
	# Mask invalids
	BT_mask = np.stack([
	    BT_C08_interp_G16, BT_C11_interp_G16, BT_C12_interp_G16, BT_C14_interp_G16, BT_C15_interp_G16, BT_C16_interp_G16,
	    BT_C08_interp_G18, BT_C11_interp_G18, BT_C12_interp_G18, BT_C14_interp_G18, BT_C15_interp_G18, BT_C16_interp_G18
	], axis=-1)
	
	mask = (
	    (BT_mask[:, 0] > 0) & (BT_mask[:, 1] > 0) & (BT_mask[:, 2] > 0) & (BT_mask[:, 3] > 0) &
	    (BT_mask[:, 4] > 0) & (BT_mask[:, 5] > 0) & (BT_mask[:, 6] > 0) & (BT_mask[:, 7] > 0) &
	    (BT_mask[:, 8] > 0) & (BT_mask[:, 9] > 0) & (BT_mask[:, 10] > 0) & (BT_mask[:, 11] > 0) &
	    (BT_mask[:, 0] < 1e03) & (BT_mask[:, 1] < 1e03) & (BT_mask[:, 2] < 1e03) & (BT_mask[:, 3] < 1e03) &
	    (BT_mask[:, 4] < 1e03) & (BT_mask[:, 5] < 1e03) & (BT_mask[:, 6] < 1e03) & (BT_mask[:, 7] < 1e03) &
	    (BT_mask[:, 8] < 1e03) & (BT_mask[:, 9] < 1e03) & (BT_mask[:, 10] < 1e03) & (BT_mask[:, 11] < 1e03) &
	    ~np.isnan(BT_mask[:, 0]) & ~np.isnan(BT_mask[:, 1]) & ~np.isnan(BT_mask[:, 2]) & ~np.isnan(BT_mask[:, 3]) &
	    ~np.isnan(BT_mask[:, 4]) & ~np.isnan(BT_mask[:, 5]) & ~np.isnan(BT_mask[:, 6]) & ~np.isnan(BT_mask[:, 7]) &
	    ~np.isnan(BT_mask[:, 8]) & ~np.isnan(BT_mask[:, 9]) & ~np.isnan(BT_mask[:, 10]) & ~np.isnan(BT_mask[:, 11])
	)
	
	BT_nozeros = BT[mask]

	return BT, BT_nozeros, mask

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="GOES16/18 Scene ID with optional PCA+GMM")
	parser.add_argument("-f", "--input_file", type=str, default="data/preprocessed_files/abi_pix1000_step5.nc")
	parser.add_argument("--model", type=str, default="data/models/gmm_pipeline_merged_res2km_10comp.joblib")
	parser.add_argument("-l", "--lambda_center", type=float, default=-106)
	args = parser.parse_args()
	
	input_file = args.input_file
	model = args.model
	lambda_center = args.lambda_center
	
	day = input_file.split("/")[-1].split(".")[0].split("_")[1]
	res = input_file.split("/")[-1].split(".")[0].split("_")[2]
		
	dataset = load_data(input_file)
	lat_interp_grid = dataset['lat_interp_grid']
	lon_interp_grid = dataset['lon_interp_grid']

	# Retrieve data
	BT, BT_nozeros, mask = build_arrays(input_file)

	# Retrieve model
	pipeline = joblib.load(model)
	scaler = pipeline.named_steps["scaler"]
	pca = pipeline.named_steps["pca"]
	gmm = pipeline.named_steps["gmm"]
	n_components = gmm.n_components

	labels_nozeros = pipeline.predict(BT_nozeros)
	
	# Map labels back to full grid
	labels = np.full_like(BT[:, 0], fill_value=0, dtype=int)
	labels[mask] = labels_nozeros

	# -----------------------------
	# Physically meaningful label ordering
	# -----------------------------
	# Sort by original-space mean of C14_G16 (index 2 in BT stacking)
	# gmm.means_ are in Z-space; map to original feature space:
	#  (1) if PCA used: inverse PCA, then inverse scale
	means_orig = scaler.inverse_transform(pca.inverse_transform(gmm.means_))
	
	sort_idx = np.argsort(means_orig[:, 3])  # 2 == C14_G16
	label_map = {orig: new for new, orig in enumerate(sort_idx)}
	sorted_labels_nozeros = np.array([label_map[l] for l in labels_nozeros])
	
	sorted_labels = np.full_like(BT[:, 0], fill_value=np.nan)
	sorted_labels[mask] = sorted_labels_nozeros
	
	scene_map = sorted_labels.reshape(lat_interp_grid.shape[0], lat_interp_grid.shape[1])
	
	# -----------------------------
	# Plot
	# -----------------------------
	lon_min = -153
	lon_max = -59
	cmap = mpl.cm.turbo
	bounds = [i - 0.5 for i in range(n_components+1)]
	norm = BoundaryNorm(bounds, cmap.N)
	
	fig, axs = plt.subplots(1, 1, figsize=(10, 8),
	                        subplot_kw={'projection': ccrs.Sinusoidal(central_longitude=lambda_center)})
	pc0 = axs.pcolormesh(lon_interp_grid, lat_interp_grid, scene_map, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
	axs.set_global()
	axs.coastlines()
	axs.set_xlabel("G16 lza (°)")
	axs.set_ylabel("G18 lza (°)")
	cbar = fig.colorbar(pc0, ax=axs, orientation="vertical")
	cbar.set_label("Scene ID")
	ticks = [i for i in range(n_components)]
	cbar.set_ticks(ticks)
	fig.tight_layout()
	
	suffix = f"{day}_{res}_{int(n_components)}comp"
	os.makedirs("figures/scene_id", exist_ok=True)
	plt.savefig(f"figures/scene_id/scene_id_{suffix}.png", dpi=150)
	plt.show()
	
	# -----------------------------
	# Save outputs (labels + settings)
	# -----------------------------
	out_path = f"data/scene_id/scene_id_{suffix}.nc"
	write_dataset(
		out_path,
		{
			"labels": labels,
			"labels_nozeros": labels_nozeros,
			"sorted_labels": sorted_labels,
			"lat": lat_interp_grid,
			"lon": lon_interp_grid,
		},
		{
			"labels": ("pixel",),
			"labels_nozeros": ("valid_pixel",),
			"sorted_labels": ("pixel",),
			"lat": ("y", "x"),
			"lon": ("y", "x"),
		},
		attrs={"product": "ECO scene identification", "n_components": int(n_components)},
	)
