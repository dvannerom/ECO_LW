"""Degrade a preprocessed ABI product to a coarser resolution by block-averaging.

Reads a `data/preprocessed_files/abi_{day}_res{res}km_step{step}.nc` product
(as produced by preprocess_data_ABI.py) and writes a new preprocessed product
at res * block_size km, so it can be dropped straight back into the rest of
the workflow (train_GMM.py, scene_id_pca.py, fit_ADM.py, ...) simply by
setting `resolution_km` to the new value in config.yaml.

Each spatial variable is read block_size rows at a time so the full-resolution
array (up to ~3.8 GB per variable) is never held in memory at once; only the
much smaller degraded-resolution output is assembled.

Radiance is block-averaged directly (the grid is equal-area, so a plain mean
is the correct area-weighted footprint average). Brightness temperature is
not block-averaged: the Planck inversion is nonlinear, so BT is instead
recomputed from the already-averaged radiance, matching preprocess_data_ABI.py.

lat/lon are recomputed exactly from the sinusoidal projection (saved as
global attrs by preprocess_data_ABI.py) rather than block-averaged, when those
attrs are available; older preprocessed files without them fall back to
block-averaging lat/lon.
"""

import argparse
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pyproj
from netCDF4 import Dataset

from geospatial import normalize_longitude
from preprocess_data_ABI import PreprocessedWriter
from radiometry import radiance_to_brightness_temperature

FILENAME_PATTERN = re.compile(r"abi_(\d+)_res(\d+)km_step(\d+)")
PROJECTION_ATTRS = ("lambda_center", "xmin", "ymin", "res_m", "authalic_radius")


def block_average_chunk(chunk, block_size):
	"""Average a (rows, cols[, channel]) chunk into (1, out_cols[, channel])."""
	chunk = np.asarray(chunk)
	rows, cols = chunk.shape[:2]
	pad_rows = math.ceil(rows / block_size) * block_size
	pad_cols = math.ceil(cols / block_size) * block_size
	if pad_rows != rows or pad_cols != cols:
		padded = np.full((pad_rows, pad_cols) + chunk.shape[2:], np.nan, dtype=chunk.dtype)
		padded[:rows, :cols] = chunk
		chunk = padded
	reshaped = chunk.reshape(
		pad_rows // block_size, block_size, pad_cols // block_size, block_size, *chunk.shape[2:]
	)
	reshaped = reshaped.swapaxes(1, 2)
	return np.nanmean(reshaped, axis=(2, 3))


def block_average_variable(source_variable, block_size):
	"""Block-average a full variable by streaming block_size rows at a time."""
	height = source_variable.shape[0]
	rows = []
	for y0 in range(0, height, block_size):
		chunk = np.asarray(source_variable[y0:y0 + block_size])
		rows.append(block_average_chunk(chunk, block_size))
		del chunk
	return np.concatenate(rows, axis=0)


def recompute_lonlat_grid(source, block_size, out_height, out_width):
	"""Recompute lat/lon exactly on the coarser sinusoidal grid from saved projection attrs."""
	lambda_center = float(source.getncattr("lambda_center"))
	xmin = float(source.getncattr("xmin"))
	ymin = float(source.getncattr("ymin"))
	res_m = float(source.getncattr("res_m"))
	Rq = float(source.getncattr("authalic_radius"))

	new_res_m = res_m * block_size
	proj = pyproj.Proj(proj="sinu", R=Rq, lon_0=lambda_center)
	x = xmin + (np.arange(out_width) + 0.5) * new_res_m
	y = ymin + (np.arange(out_height) + 0.5) * new_res_m
	x_grid, y_grid = np.meshgrid(x, y)
	lon_grid, lat_grid = proj(x_grid, y_grid, inverse=True)
	return lat_grid, normalize_longitude(lon_grid), new_res_m, Rq, xmin, ymin, lambda_center


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Degrade a preprocessed ABI product to a coarser resolution."
	)
	parser.add_argument("-f", "--input-file", type=str, required=True)
	parser.add_argument("-b", "--block-size", type=int, required=True)
	args = parser.parse_args()

	input_path = Path(args.input_file)
	match = FILENAME_PATTERN.match(input_path.stem)
	if match is None:
		raise ValueError(f"Cannot parse day/resolution/step from filename: {input_path.name}")
	day, res_km, step = match.groups()
	new_res_km = int(res_km) * args.block_size

	source = Dataset(input_path, "r")
	try:
		out_height = math.ceil(source.dimensions["y"].size / args.block_size)
		out_width = math.ceil(source.dimensions["x"].size / args.block_size)

		output_path = f"data/preprocessed_files/abi_{day}_res{new_res_km}km_step{step}.nc"
		writer = PreprocessedWriter(output_path, out_height, out_width)
		try:
			writer.write_scalar("width", out_width)
			writer.write_scalar("height", out_height)
			writer.write_scalar("step", int(source.variables["step"][()]))

			has_projection_attrs = all(attr in source.ncattrs() for attr in PROJECTION_ATTRS)
			skip_names = {"width", "height", "step"}
			if has_projection_attrs:
				lat_grid, lon_grid, new_res_m, Rq, xmin, ymin, lambda_center = recompute_lonlat_grid(
					source, args.block_size, out_height, out_width
				)
				writer.dataset.setncattr("lambda_center", lambda_center)
				writer.dataset.setncattr("xmin", xmin)
				writer.dataset.setncattr("ymin", ymin)
				writer.dataset.setncattr("res_m", new_res_m)
				writer.dataset.setncattr("authalic_radius", Rq)
				writer.write_2d("lat_interp_grid", lat_grid)
				writer.write_2d("lon_interp_grid", lon_grid)
				del lat_grid, lon_grid
				skip_names |= {"lat_interp_grid", "lon_interp_grid"}
			else:
				warnings.warn(
					"Source file has no saved projection attrs; block-averaging "
					"lat_interp_grid/lon_interp_grid as an approximation instead of "
					"recomputing them exactly."
				)

			planck_by_satellite = {
				name.split("_")[1]: np.asarray(source.variables[name][:])
				for name in ("planck_G16", "planck_G18")
				if name in source.variables
			}

			for name, variable in source.variables.items():
				if name in skip_names:
					continue
				if name.startswith("planck_"):
					writer.write_planck(name, variable[:])
					continue
				if name.startswith("BT_"):
					continue  # derived below from the matching rad_* variable

				averaged = block_average_variable(variable, args.block_size)
				if averaged.ndim == 2:
					writer.write_2d(name, averaged)
				elif averaged.ndim == 3:
					for channel in range(averaged.shape[2]):
						writer.write_channel(name, channel, averaged[:, :, channel])
				else:
					raise ValueError(f"Unsupported variable shape for {name}: {averaged.shape}")

				if name.startswith("rad_"):
					satellite = name.split("_")[1]
					planck = planck_by_satellite[satellite]
					bt_name = name.replace("rad_", "BT_", 1)
					for channel in range(averaged.shape[2]):
						brightness_temperature = radiance_to_brightness_temperature(
							averaged[:, :, channel], planck[channel]
						)
						writer.write_channel(bt_name, channel, brightness_temperature)
					del brightness_temperature
				del averaged
		finally:
			writer.close()
	finally:
		source.close()
