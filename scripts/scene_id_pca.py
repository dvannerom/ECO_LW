import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
from netcdf_io import write_dataset
from scene_features import build_scene_features
from product_paths import model_path, selected_n_components

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="GOES16/18 Scene ID with optional PCA+GMM")
	parser.add_argument("-f", "--input_file", type=str, default="data/preprocessed_files/abi_pix1000_step5.nc")
	parser.add_argument("--model", type=str, default=None)
	parser.add_argument("-l", "--lambda_center", type=float, default=-106)
	parser.add_argument("--no-plot", action="store_true", help="Skip plotting during compute; generate plots offline with plot_scene_id.py")
	args = parser.parse_args()
	
	input_file = args.input_file
	model = args.model or model_path(2, selected_n_components())
	lambda_center = args.lambda_center
	
	day = input_file.split("/")[-1].split(".")[0].split("_")[1]
	res = input_file.split("/")[-1].split(".")[0].split("_")[2]
		
	# Retrieve data
	BT_nozeros, valid_flat, mask, lat_interp_grid, lon_interp_grid = build_scene_features(input_file)

	# Retrieve model
	pipeline = joblib.load(model)
	scaler = pipeline.named_steps["scaler"]
	pca = pipeline.named_steps["pca"]
	gmm = pipeline.named_steps["gmm"]
	n_components = gmm.n_components

	labels_nozeros = pipeline.predict(BT_nozeros)
	
	# Map labels back to full grid while keeping memory use bounded.
	n_pixels = mask.size
	labels = np.zeros(n_pixels, dtype=np.int16)
	labels[valid_flat] = labels_nozeros

	# -----------------------------
	# Physically meaningful label ordering
	# -----------------------------
	# Sort by original-space mean of C14_G16 (index 2 in BT stacking)
	# gmm.means_ are in Z-space; map to original feature space:
	#  (1) if PCA used: inverse PCA, then inverse scale
	means_orig = scaler.inverse_transform(pca.inverse_transform(gmm.means_))
	
	sort_idx = np.argsort(means_orig[:, 3])  # 2 == C14_G16
	label_map = {orig: new for new, orig in enumerate(sort_idx)}
	sorted_labels_nozeros = np.array([label_map[l] for l in labels_nozeros], dtype=np.int16)
	
	sorted_labels = np.full(n_pixels, fill_value=np.nan, dtype=np.float32)
	sorted_labels[valid_flat] = sorted_labels_nozeros
	
	scene_map = sorted_labels.reshape(lat_interp_grid.shape[0], lat_interp_grid.shape[1])
	suffix = f"{day}_{res}_{int(n_components)}comp"
	
	# -----------------------------
	# Plot (optional; can be done offline)
	# -----------------------------
	if not args.no_plot:
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
