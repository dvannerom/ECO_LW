import numpy as np
import h5py
import xarray as xr
import math
import argparse
import os
from pathlib import Path
from scipy.interpolate import griddata, RegularGridInterpolator
from pyresample import geometry, kd_tree
import pyproj
import time
from netCDF4 import Dataset
from geospatial import atanh_safe, normalize_longitude
from goes_navigation import (
    latlon_to_xy_goes,
    parallax_correction,
    xy_goes_to_latlon_grid,
)
from radiometry import radiance_to_brightness_temperature


wrap_lon = normalize_longitude
latlon_to_xyGOES = latlon_to_xy_goes
xyGOES_to_latlonGRID = xy_goes_to_latlon_grid

def build_files(day):
	list_G16_C08 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409/G16/").rglob("*C08*s2024*") if f.is_file()])
	list_G16_C09 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409_missing_channels/G16/").rglob("*C09*s2024*") if f.is_file()])
	list_G16_C10 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409_missing_channels/G16/").rglob("*C10*s2024*") if f.is_file()])
	list_G16_C11 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409/G16/").rglob("*C11*s2024*") if f.is_file()])
	list_G16_C12 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409/G16/").rglob("*C12*s2024*") if f.is_file()])
	list_G16_C13 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409_missing_channels/G16/").rglob("*C13*s2024*") if f.is_file()])
	list_G16_C14 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409/G16/").rglob("*C14*s2024*") if f.is_file()])
	list_G16_C15 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409/G16/").rglob("*C15*s2024*") if f.is_file()])
	list_G16_C16 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409/G16/").rglob("*C16*s2024*") if f.is_file()])
	list_G16_CTH = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/cloud-top-height/202409/G16/").rglob("*ACHA2KMF*G16*") if f.is_file()])
	list_G18_C08 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409/G18/").rglob("*C08*s2024*") if f.is_file()])
	list_G18_C09 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409_missing_channels/G18/").rglob("*C09*s2024*") if f.is_file()])
	list_G18_C10 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409_missing_channels/G18/").rglob("*C10*s2024*") if f.is_file()])
	list_G18_C11 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409/G18/").rglob("*C11*s2024*") if f.is_file()])
	list_G18_C12 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409/G18/").rglob("*C12*s2024*") if f.is_file()])
	list_G18_C13 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409_missing_channels/G18/").rglob("*C13*s2024*") if f.is_file()])
	list_G18_C14 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409/G18/").rglob("*C14*s2024*") if f.is_file()])
	list_G18_C15 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409/G18/").rglob("*C15*s2024*") if f.is_file()])
	list_G18_C16 = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/202409/G18/").rglob("*C16*s2024*") if f.is_file()])
	list_G18_CTH = sorted([str(f.resolve()) for f in Path("/ksb-orb/science/goes/ABI/cloud-top-height/202409/G18/").rglob("*ACHA2KMF*G18*") if f.is_file()])

	file_G16_C08 = [f for f in list_G16_C08 if day in f]
	file_G16_C09 = [f for f in list_G16_C09 if day in f]
	file_G16_C10 = [f for f in list_G16_C10 if day in f]
	file_G16_C11 = [f for f in list_G16_C11 if day in f]
	file_G16_C12 = [f for f in list_G16_C12 if day in f]
	file_G16_C13 = [f for f in list_G16_C13 if day in f]
	file_G16_C14 = [f for f in list_G16_C14 if day in f]
	file_G16_C15 = [f for f in list_G16_C15 if day in f]
	file_G16_C16 = [f for f in list_G16_C16 if day in f]
	file_G16_CTH = [f for f in list_G16_CTH if day in f]
	file_G18_C08 = [f for f in list_G18_C08 if day in f]
	file_G18_C09 = [f for f in list_G18_C09 if day in f]
	file_G18_C10 = [f for f in list_G18_C10 if day in f]
	file_G18_C11 = [f for f in list_G18_C11 if day in f]
	file_G18_C12 = [f for f in list_G18_C12 if day in f]
	file_G18_C13 = [f for f in list_G18_C13 if day in f]
	file_G18_C14 = [f for f in list_G18_C14 if day in f]
	file_G18_C15 = [f for f in list_G18_C15 if day in f]
	file_G18_C16 = [f for f in list_G18_C16 if day in f]
	file_G18_CTH = [f for f in list_G18_CTH if day in f]
	
	return file_G16_C08, file_G16_C09, file_G16_C10, file_G16_C11, file_G16_C12, file_G16_C13, file_G16_C14, file_G16_C15, file_G16_C16, file_G16_CTH, file_G18_C08, file_G18_C09, file_G18_C10, file_G18_C11, file_G18_C12, file_G18_C13, file_G18_C14, file_G18_C15, file_G18_C16, file_G18_CTH

def preprocess_data(files, step):#, res, lon_min, lon_max, lat_max):
	# Retrieve data and coordinates
	ds_C08 = xr.open_dataset(os.path.join(files[0][0]))
	rad_C08 = ds_C08.Rad[::step,::step].data
	planck_C08 = [ds_C08.planck_fk1.data,ds_C08.planck_fk2.data,ds_C08.planck_bc1.data,ds_C08.planck_bc2.data]
	BT_C08 = radiance_to_brightness_temperature(rad_C08, [ds_C08.planck_fk1.data, ds_C08.planck_fk2.data, ds_C08.planck_bc1.data, ds_C08.planck_bc2.data])
	ds_C09 = xr.open_dataset(os.path.join(files[1][0]))
	rad_C09 = ds_C09.Rad[::step,::step].data
	planck_C09 = [ds_C09.planck_fk1.data,ds_C09.planck_fk2.data,ds_C09.planck_bc1.data,ds_C09.planck_bc2.data]
	BT_C09 = radiance_to_brightness_temperature(rad_C09, [ds_C09.planck_fk1.data, ds_C09.planck_fk2.data, ds_C09.planck_bc1.data, ds_C09.planck_bc2.data])
	ds_C10 = xr.open_dataset(os.path.join(files[2][0]))
	rad_C10 = ds_C10.Rad[::step,::step].data
	planck_C10 = [ds_C10.planck_fk1.data,ds_C10.planck_fk2.data,ds_C10.planck_bc1.data,ds_C10.planck_bc2.data]
	BT_C10 = radiance_to_brightness_temperature(rad_C10, [ds_C10.planck_fk1.data, ds_C10.planck_fk2.data, ds_C10.planck_bc1.data, ds_C10.planck_bc2.data])
	ds_C11 = xr.open_dataset(os.path.join(files[3][0]))
	rad_C11 = ds_C11.Rad[::step,::step].data
	planck_C11 = [ds_C11.planck_fk1.data,ds_C11.planck_fk2.data,ds_C11.planck_bc1.data,ds_C11.planck_bc2.data]
	BT_C11 = radiance_to_brightness_temperature(rad_C11, [ds_C11.planck_fk1.data, ds_C11.planck_fk2.data, ds_C11.planck_bc1.data, ds_C11.planck_bc2.data])
	ds_C12 = xr.open_dataset(os.path.join(files[4][0]))
	rad_C12 = ds_C12.Rad[::step,::step].data
	planck_C12 = [ds_C12.planck_fk1.data,ds_C12.planck_fk2.data,ds_C12.planck_bc1.data,ds_C12.planck_bc2.data]
	BT_C12 = radiance_to_brightness_temperature(rad_C12, [ds_C12.planck_fk1.data, ds_C12.planck_fk2.data, ds_C12.planck_bc1.data, ds_C12.planck_bc2.data])
	ds_C13 = xr.open_dataset(os.path.join(files[5][0]))
	rad_C13 = ds_C13.Rad[::step,::step].data
	planck_C13 = [ds_C13.planck_fk1.data,ds_C13.planck_fk2.data,ds_C13.planck_bc1.data,ds_C13.planck_bc2.data]
	BT_C13 = radiance_to_brightness_temperature(rad_C13, [ds_C13.planck_fk1.data, ds_C13.planck_fk2.data, ds_C13.planck_bc1.data, ds_C13.planck_bc2.data])
	ds_C14 = xr.open_dataset(os.path.join(files[6][0]))
	rad_C14 = ds_C14.Rad[::step,::step].data
	planck_C14 = [ds_C14.planck_fk1.data,ds_C14.planck_fk2.data,ds_C14.planck_bc1.data,ds_C14.planck_bc2.data]
	BT_C14 = radiance_to_brightness_temperature(rad_C14, [ds_C14.planck_fk1.data, ds_C14.planck_fk2.data, ds_C14.planck_bc1.data, ds_C14.planck_bc2.data])
	ds_C15 = xr.open_dataset(os.path.join(files[7][0]))
	rad_C15 = ds_C15.Rad[::step,::step].data
	planck_C15 = [ds_C15.planck_fk1.data,ds_C15.planck_fk2.data,ds_C15.planck_bc1.data,ds_C15.planck_bc2.data]
	BT_C15 = radiance_to_brightness_temperature(rad_C15, [ds_C15.planck_fk1.data, ds_C15.planck_fk2.data, ds_C15.planck_bc1.data, ds_C15.planck_bc2.data])
	ds_C16 = xr.open_dataset(os.path.join(files[8][0]))
	rad_C16 = ds_C16.Rad[::step,::step].data
	planck_C16 = [ds_C16.planck_fk1.data,ds_C16.planck_fk2.data,ds_C16.planck_bc1.data,ds_C16.planck_bc2.data]
	BT_C16 = radiance_to_brightness_temperature(rad_C16, [ds_C16.planck_fk1.data, ds_C16.planck_fk2.data, ds_C16.planck_bc1.data, ds_C16.planck_bc2.data])
	ds_CTH = xr.open_dataset(os.path.join(files[9][0]))
	CTH = ds_CTH.HT[::step,::step].data

	xGOES = ds_C11.x[::step].data
	yGOES = ds_C11.y[::step].data
	xGOES_grid, yGOES_grid = np.meshgrid(xGOES,yGOES)#,indexing='ij')

	goes_imager_projection = ds_C11.goes_imager_projection

	rad = np.stack([rad_C08,rad_C09,rad_C10,rad_C11,rad_C12,rad_C13,rad_C14,rad_C15,rad_C16], axis=-1)
	BT = np.stack([BT_C08,BT_C09,BT_C10,BT_C11,BT_C12,BT_C13,BT_C14,BT_C15,BT_C16], axis=-1)
	planck = np.stack([planck_C08,planck_C09,planck_C10,planck_C11,planck_C12,planck_C13,planck_C14,planck_C15,planck_C16])

	# Convert GOES x-y coordinates to lat-lon
	lat, lon, lat_corr, lon_corr, lza = xyGOES_to_latlonGRID(xGOES_grid, yGOES_grid, goes_imager_projection, CTH)

	return lat, wrap_lon(lon), lat_corr, wrap_lon(lon_corr), lza, rad, BT, CTH, planck, goes_imager_projection


def load_navigation(files, step):
	"""Load geometry without loading the nine radiance channels."""
	with xr.open_dataset(files[9][0], engine="netcdf4") as cth_dataset:
		cth = cth_dataset.HT[::step, ::step].values

	with xr.open_dataset(files[3][0], engine="netcdf4") as reference_dataset:
		x_goes = reference_dataset.x[::step].values
		y_goes = reference_dataset.y[::step].values
		projection = reference_dataset.goes_imager_projection

	x_grid, y_grid = np.meshgrid(x_goes, y_goes)
	lat, lon, lat_corr, lon_corr, lza = xyGOES_to_latlonGRID(
		x_grid, y_grid, projection, cth
	)

	return {
		"lat": lat,
		"lon": wrap_lon(lon),
		"lat_corr": lat_corr,
		"lon_corr": wrap_lon(lon_corr),
		"lza": lza,
		"cth": cth,
		"projection": projection,
	}


def resample_field(source_grid, values, area_def, radius_of_influence):
	"""Resample one field so pyresample never receives a 3-D channel stack."""
	return kd_tree.resample_nearest(
		source_grid,
		values,
		area_def,
		radius_of_influence=radius_of_influence,
		fill_value=np.nan,
	)


class PreprocessedWriter:
	"""Write the preprocessing product while arrays are still short-lived."""

	def __init__(self, path, height, width):
		Path(path).parent.mkdir(parents=True, exist_ok=True)
		self.dataset = Dataset(path, "w", format="NETCDF4")
		self.dataset.setncattr("product", "ECO ABI preprocessed data")
		self.dataset.createDimension("y", height)
		self.dataset.createDimension("x", width)
		self.dataset.createDimension("channel", 9)
		self.dataset.createDimension("planck_parameter", 4)
		self.variables = {}

	def _create(self, name, values, dimensions):
		array = np.asarray(values)
		chunksizes = None
		if dimensions == ("y", "x", "channel"):
			chunksizes = (
				min(self.dataset.dimensions["y"].size, 256),
				min(self.dataset.dimensions["x"].size, 256),
				1,
			)
		variable = self.dataset.createVariable(
			name,
			"f4" if array.dtype.kind == "f" else array.dtype,
			dimensions,
			zlib=True,
			complevel=1,
			shuffle=True,
			chunksizes=chunksizes,
		)
		self.variables[name] = variable
		return variable

	def write_scalar(self, name, value):
		self._create(name, np.asarray(value), ())[:] = value

	def write_2d(self, name, values):
		self._create(name, values, ("y", "x"))[:] = values

	def write_planck(self, name, values):
		self._create(name, values, ("channel", "planck_parameter"))[:] = values

	def write_channel(self, name, channel, values):
		if name not in self.variables:
			self._create(name, values, ("y", "x", "channel"))
		self.variables[name][:, :, channel] = values

	def close(self):
		self.dataset.close()


def process_satellite(files, navigation, source_grid, corrected_grid, area_def,
					  radius_of_influence, writer, satellite_name, step):
	"""Resample one satellite and one channel at a time."""
	static_fields = (
		("lza", "lza", source_grid, ""),
		("CTH", "cth", source_grid, ""),
		("lza", "lza", corrected_grid, "_corr"),
		("CTH", "cth", corrected_grid, "_corr"),
	)
	for output_name, navigation_name, grid, suffix in static_fields:
		resampled = resample_field(
			grid,
			navigation[navigation_name],
			area_def,
			radius_of_influence,
		)
		writer.write_2d(
			f"{output_name}_{satellite_name}_interp{suffix}", resampled
		)
		del resampled

	planck = np.empty((9, 4), dtype=np.float32)
	for channel, radiance_files in enumerate(files[:9]):
		with xr.open_dataset(radiance_files[0], engine="netcdf4") as dataset:
			radiance = dataset.Rad[::step, ::step].values
			planck[channel] = [
				dataset.planck_fk1.values,
				dataset.planck_fk2.values,
				dataset.planck_bc1.values,
				dataset.planck_bc2.values,
			]
			brightness_temperature = radiance_to_brightness_temperature(
				radiance,
				[dataset.planck_fk1.values, dataset.planck_fk2.values,
				 dataset.planck_bc1.values, dataset.planck_bc2.values],
			)

		for suffix, grid in (("", source_grid), ("_corr", corrected_grid)):
			resampled_radiance = resample_field(
				grid, radiance, area_def, radius_of_influence
			)
			resampled_temperature = resample_field(
				grid, brightness_temperature, area_def, radius_of_influence
			)
			writer.write_channel(
				f"rad_{satellite_name}_interp{suffix}",
				channel,
				resampled_radiance,
			)
			writer.write_channel(
				f"BT_{satellite_name}_interp{suffix}",
				channel,
				resampled_temperature,
			)
			del resampled_radiance, resampled_temperature

		del radiance, brightness_temperature

	writer.write_planck(f"planck_{satellite_name}", planck)
	del planck


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Preprocess ABI radiances")
	parser.add_argument("-d", "--day", type=str, default="245")
	parser.add_argument("-s", "--step", type=int, default=1)
	parser.add_argument("-r", "--res_km", type=float, default=2)
	parser.add_argument("-l", "--lambda_center", type=float, default=-106)
	args = parser.parse_args()

	day = args.day
	step = args.step
	res_km = args.res_km
	lambda_center = args.lambda_center
	files = build_files(day)
	navigation_g16 = load_navigation(files[:10], step)
	navigation_g18 = load_navigation(files[10:], step)

	goes = navigation_g16["projection"]
	a = float(goes.semi_major_axis)
	b = float(goes.semi_minor_axis)
	e = np.sqrt(1.0 - (b * b) / (a * a))
	Rq = a * np.sqrt(0.5 * (1.0 + ((1.0 - e * e) / e) * atanh_safe(e)))
	proj_dict = {"proj": "sinu", "R": Rq, "lon_0": lambda_center}
	proj = pyproj.Proj(**proj_dict)

	xmin, ymin = np.inf, np.inf
	xmax, ymax = -np.inf, -np.inf
	for navigation in (navigation_g16, navigation_g18):
		x_projected, y_projected = proj(navigation["lon"], navigation["lat"])
		finite = np.isfinite(x_projected) & np.isfinite(y_projected)
		if np.any(finite):
			xmin = min(xmin, np.nanmin(x_projected[finite]))
			xmax = max(xmax, np.nanmax(x_projected[finite]))
			ymin = min(ymin, np.nanmin(y_projected[finite]))
			ymax = max(ymax, np.nanmax(y_projected[finite]))

	res_m = res_km * 1000.0
	width = int(np.ceil((xmax - xmin) / res_m))
	height = int(np.ceil((ymax - ymin) / res_m))
	xmax = xmin + width * res_m
	ymax = ymin + height * res_m
	area_def = geometry.AreaDefinition(
		area_id="abi_sinu_union_g16_g18",
		description="ABI (G16+G18) union on sinusoidal equal-area grid (authalic)",
		proj_id="sinu_auth_union",
		projection=proj_dict,
		width=width,
		height=height,
		area_extent=(xmin, ymin, xmax, ymax),
	)

	output_path = (
		f"data/preprocessed_files/abi_{day}_res{int(res_km)}km_step{step}.nc"
	)
	writer = PreprocessedWriter(output_path, height, width)
	try:
		writer.write_scalar("width", width)
		writer.write_scalar("height", height)
		writer.write_scalar("step", step)
		lon_interp_grid, lat_interp_grid = area_def.get_lonlats()
		writer.write_2d("lat_interp_grid", lat_interp_grid)
		writer.write_2d("lon_interp_grid", lon_interp_grid)
		del lon_interp_grid, lat_interp_grid

		roi_m = 20000.0
		g16_grid = geometry.SwathDefinition(
			lons=navigation_g16["lon"], lats=navigation_g16["lat"]
		)
		g16_corr_grid = geometry.SwathDefinition(
			lons=navigation_g16["lon_corr"], lats=navigation_g16["lat_corr"]
		)
		process_satellite(
			files[:10], navigation_g16, g16_grid, g16_corr_grid,
			area_def, roi_m, writer, "G16", step,
		)
		del navigation_g16, g16_grid, g16_corr_grid

		g18_grid = geometry.SwathDefinition(
			lons=navigation_g18["lon"], lats=navigation_g18["lat"]
		)
		g18_corr_grid = geometry.SwathDefinition(
			lons=navigation_g18["lon_corr"], lats=navigation_g18["lat_corr"]
		)
		process_satellite(
			files[10:], navigation_g18, g18_grid, g18_corr_grid,
			area_def, roi_m, writer, "G18", step,
		)
	finally:
		writer.close()
