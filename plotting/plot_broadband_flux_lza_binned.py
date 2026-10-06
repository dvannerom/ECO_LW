#!/usr/bin/env python3
"""Bin the G16-G18 broadband flux difference by viewing-zenith-angle pair.

Produces two heatmaps (mean and standard deviation of the flux difference,
each pixel binned by its GOES-16 and GOES-18 viewing zenith angles) in one
run. Accepts one broadband_flux_*.nc file for a daily heatmap, or several for
a multi-day/monthly aggregate; both are computed with the same incremental,
one-file-at-a-time accumulation so no per-day flux fields are stacked in
memory.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from netcdf_io import load_data

N_BINS = 9
BIN_WIDTH_DEG = 10.0
BIN_LABELS = [f"{lo}-{lo + 10}" for lo in range(0, 90, 10)]


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
	"""Create a heatmap from a 2D array and row/column bin labels."""
	if ax is None:
		ax = plt.gca()
	if cbar_kw is None:
		cbar_kw = {}

	im = ax.imshow(data, **kwargs)

	cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
	cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

	ax.set_xticks(range(data.shape[1]), labels=col_labels, rotation=30, ha="right", rotation_mode="anchor")
	ax.set_yticks(range(data.shape[0]), labels=row_labels)
	ax.set_xlabel("G16 lza (°)")
	ax.set_ylabel("G18 lza (°)")

	ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
	ax.spines[:].set_visible(False)
	ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
	ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
	ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
	ax.tick_params(which="minor", bottom=False, left=False)

	return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
	"""Write the value of each heatmap cell as text, colored by contrast with the cell's fill."""
	if not isinstance(data, (list, np.ndarray)):
		data = im.get_array()

	threshold = im.norm(threshold) if threshold is not None else im.norm(np.nanmax(data)) / 2.

	kw = dict(horizontalalignment="center", verticalalignment="center")
	kw.update(textkw)

	if isinstance(valfmt, str):
		valfmt = mpl.ticker.StrMethodFormatter(valfmt)

	texts = []
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			if not np.isfinite(data[i, j]):
				continue
			kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
			texts.append(im.axes.text(j, i, valfmt(data[i, j], None), **kw))

	return texts


def bin_flux_difference_by_lza(input_files, reference_data):
	"""Accumulate mean/stddev of (flux_G16 - flux_G18) per (lza_G18, lza_G16) bin.

	Uses Chan et al.'s parallel variance combination to merge each file's
	per-bin batch statistics (from a single vectorized `bincount` pass) into
	running per-bin mean/M2 accumulators, one file at a time.
	"""
	reference = load_data(reference_data, variable_names=("lza_G16_interp_corr", "lza_G18_interp_corr"))
	lza_G16 = reference["lza_G16_interp_corr"].ravel()
	lza_G18 = reference["lza_G18_interp_corr"].ravel()
	finite_lza = np.isfinite(lza_G16) & np.isfinite(lza_G18)
	bin_G16 = np.clip(np.floor(lza_G16 / BIN_WIDTH_DEG), 0, N_BINS - 1)
	bin_G18 = np.clip(np.floor(lza_G18 / BIN_WIDTH_DEG), 0, N_BINS - 1)

	n_bins = N_BINS * N_BINS
	count = np.zeros(n_bins, dtype=np.int64)
	mean = np.zeros(n_bins, dtype=np.float64)
	M2 = np.zeros(n_bins, dtype=np.float64)

	for input_file in input_files:
		dataset = load_data(input_file, variable_names=("flux_G16", "flux_G18"))
		diff = (dataset["flux_G16"] - dataset["flux_G18"]).ravel()
		valid = finite_lza & np.isfinite(diff)
		del dataset

		combined_bin = (bin_G18[valid] * N_BINS + bin_G16[valid]).astype(np.int64)
		values = diff[valid]
		del diff, valid

		batch_count = np.bincount(combined_bin, minlength=n_bins)
		batch_sum = np.bincount(combined_bin, weights=values, minlength=n_bins)
		batch_sumsq = np.bincount(combined_bin, weights=values ** 2, minlength=n_bins)
		del combined_bin, values

		with np.errstate(invalid="ignore", divide="ignore"):
			batch_mean = np.where(batch_count > 0, batch_sum / np.maximum(batch_count, 1), 0.0)
			batch_M2 = batch_sumsq - batch_count * batch_mean ** 2

			new_count = count + batch_count
			delta = batch_mean - mean
			mean = np.where(
				new_count > 0,
				(count * mean + batch_count * batch_mean) / np.maximum(new_count, 1),
				mean,
			)
			M2 = M2 + batch_M2 + delta ** 2 * count * batch_count / np.maximum(new_count, 1)
			count = new_count

	mean_grid = np.where(count > 0, mean, np.nan).reshape(N_BINS, N_BINS)
	stddev_grid = np.sqrt(np.where(count > 1, M2 / np.maximum(count, 1), np.nan)).reshape(N_BINS, N_BINS)
	count_grid = count.reshape(N_BINS, N_BINS)

	# Row 0 = highest G18 viewing zenith angle bin, matching the plotted orientation.
	return np.flipud(mean_grid), np.flipud(stddev_grid), np.flipud(count_grid)


def plot_lza_binned_heatmaps(mean_grid, stddev_grid, output_dir, tag):
	"""Plot both the mean (diverging, bwr) and stddev (sequential, rainbow) heatmaps."""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	row_labels = BIN_LABELS[::-1]

	abs_max = np.nanmax(np.abs(mean_grid)) if np.any(np.isfinite(mean_grid)) else 1.0
	fig, ax = plt.subplots()
	im, _ = heatmap(
		mean_grid, row_labels, BIN_LABELS, ax=ax, cmap="bwr", vmin=-abs_max, vmax=abs_max,
		cbarlabel="Mean of G16-G18 (W/m²)",
	)
	# Diverging map: sign-aware labels, contrast threshold at zero (the colormap's center).
	annotate_heatmap(im, data=mean_grid, valfmt="{x:+.1f}", threshold=0.0)
	fig.tight_layout()
	mean_path = output_dir / f"broadband_flux_lza_binned_mean_{tag}.png"
	fig.savefig(mean_path)
	plt.close(fig)

	fig, ax = plt.subplots()
	im, _ = heatmap(
		stddev_grid, row_labels, BIN_LABELS, ax=ax, cmap="rainbow", vmin=0,
		cbarlabel="Standard deviation of G16-G18 (W/m²)",
	)
	# Sequential map: unsigned labels, default half-of-max contrast threshold.
	annotate_heatmap(im, data=stddev_grid, valfmt="{x:.1f}")
	fig.tight_layout()
	stddev_path = output_dir / f"broadband_flux_lza_binned_stddev_{tag}.png"
	fig.savefig(stddev_path)
	plt.close(fig)

	return mean_path, stddev_path


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Bin the G16-G18 broadband flux difference by viewing zenith angle "
		            "and plot its mean and standard deviation as heatmaps."
	)
	parser.add_argument(
		"--inputs", nargs="+", required=True,
		help="One or more broadband_flux_*.nc files (one file for a daily heatmap, "
		     "several for a multi-day/monthly aggregate)",
	)
	parser.add_argument(
		"--reference-data", required=True,
		help="Preprocessed file providing the lza_G16_interp_corr/lza_G18_interp_corr grids",
	)
	parser.add_argument(
		"--tag", required=True,
		help="Label used in the output filenames, e.g. a day number or 'monthly'",
	)
	parser.add_argument("--output-dir", type=str, default="figures/diagnostics/broadband_flux")
	args = parser.parse_args()

	mean_grid, stddev_grid, _ = bin_flux_difference_by_lza(args.inputs, args.reference_data)
	mean_path, stddev_path = plot_lza_binned_heatmaps(mean_grid, stddev_grid, args.output_dir, args.tag)
	print(f"Saved {mean_path}")
	print(f"Saved {stddev_path}")
