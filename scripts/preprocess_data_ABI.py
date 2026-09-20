import numpy as np
import matplotlib.pyplot as plt
import h5py
import xarray as xr
import math
import argparse
import os
from pathlib import Path
from utils import *
from scipy.interpolate import griddata, RegularGridInterpolator
from pyresample import geometry, kd_tree
import pyproj
import time
from netCDF4 import Dataset

def outer_frame(lon2d, lat2d):
    """Return the four sides of the swath as 1D arrays (lon, lat)."""
    # top, bottom, left, right
    top_lon,    top_lat    = lon2d[0, :],      lat2d[0, :]
    bottom_lon, bottom_lat = lon2d[-1, :],     lat2d[-1, :]
    left_lon,   left_lat   = lon2d[:, 0],      lat2d[:, 0]
    right_lon,  right_lat  = lon2d[:, -1],     lat2d[:, -1]
    # concatenate sides
    lon = np.concatenate([top_lon, bottom_lon, left_lon, right_lon])
    lat = np.concatenate([top_lat, bottom_lat, left_lat, right_lat])
    return lon, lat

# Helper: normalize longitudes relative to lon_0 so we avoid wrap issues
def wrap_lon(lon_deg):
	# returns longitudes in (lon0-180, lon0+180]
	return ((lon_deg + 180.0) % 360.0) - 180.0

def atanh_safe(x):
    # robust atanh that returns 0 when x ~ 0 to avoid 0/0 issues
    return 0.0 if np.isclose(x, 0.0) else 0.5*np.log((1.0 + x)/(1.0 - x))

def rad_to_T(rad,ds):
	ratio = ds.planck_fk1.data/rad
	a = ratio + 1
	b = np.log(a)
	c = ds.planck_fk2.data/b
	d = c - ds.planck_bc1.data
	return d/ds.planck_bc2.data

def T_to_alt(BT):
	if np.isnan(BT) or np.isinf(BT) or BT > 288.15: return 0
	elif BT < 288.15 and BT > 216.65: return (1000/6.5)*(288.15-BT)
	else: return 11000
T_to_alt = np.vectorize(T_to_alt)

def latlon_to_xyGOES(lat, lon, goes_imager_projection):
	# Define GOES geometrical parameters	
	lambda0 = math.radians(goes_imager_projection.longitude_of_projection_origin)
	r_eq = goes_imager_projection.semi_major_axis
	r_pol = goes_imager_projection.semi_minor_axis
	H = goes_imager_projection.perspective_point_height + r_eq
	f = 1./goes_imager_projection.inverse_flattening
	e = math.sqrt(f*(2-f))

	phi_c = v_compute_phi_c(r_pol,r_eq,lat)
	rc = v_compute_rc(r_pol,e,phi_c)
	sx_2 = v_compute_sx_2(H,rc,phi_c,lon,lambda0)
	sy_2 = v_compute_sy_2(rc,phi_c,lon,lambda0)
	sz_2 = v_compute_sz_2(rc,phi_c)

	# Convert from lat-lon to GOES x-y
	xGOES = v_compute_x(sx_2,sy_2,sz_2)
	yGOES = v_compute_y(sx_2,sz_2)

	return xGOES, yGOES

def parallax_correction(lat, lon, CTH, goes_imager_projection):
	alt = np.nan_to_num(CTH)

	# Define GOES geometrical parameters	
	lambda0 = math.radians(goes_imager_projection.longitude_of_projection_origin)
	r_eq = goes_imager_projection.semi_major_axis
	r_pol = goes_imager_projection.semi_minor_axis
	H = goes_imager_projection.perspective_point_height + r_eq

	# Satellite cartesian coordinates
	x_sat = H
	y_sat = 0
	z_sat = 0
	# Pixel cartesian coordinates
	R_ratio = r_eq/r_pol
	lat_geoc = np.arctan(np.tan(lat)/math.pow(R_ratio,2))
	R_local = r_eq/np.sqrt(np.power(np.cos(lat_geoc),2)+np.power(R_ratio*np.sin(lat_geoc),2))
	x_GOES = R_local*np.cos(lat_geoc)*np.cos(lon-lambda0)
	y_GOES = R_local*np.cos(lat_geoc)*np.sin(lon-lambda0)
	z_GOES = R_local*np.sin(lat_geoc)
	# Compute difference vector
	x_diff = x_sat - x_GOES
	y_diff = y_sat - y_GOES
	z_diff = z_sat - z_GOES
	# R_ratio local
	R_ratio_local = np.power((r_eq+alt)/(r_pol+alt),2)
	# Compute correction factor
	quad_a = np.power(x_diff,2) + np.power(y_diff,2) + R_ratio_local*np.power(z_diff,2)
	quad_b = 2*(x_GOES*x_diff + y_GOES*y_diff + R_ratio_local*z_GOES*z_diff)
	quad_c = np.power(x_GOES,2) + np.power(y_GOES,2) + R_ratio_local*np.power(z_GOES,2) - np.power(r_eq+alt,2)
	c = (np.sqrt(np.power(quad_b,2)-4*quad_a*quad_c) - quad_b)/(2*quad_a)
	# Compute corrected cartesian coordinates
	x_corr = x_GOES + c*x_diff
	y_corr = y_GOES + c*y_diff
	z_corr = z_GOES + c*z_diff
	# Compute corrected latitude and longitude
	lat_corr = np.arctan2(R_ratio_local*z_corr,np.sqrt(np.power(x_corr,2)+np.power(y_corr,2)))
	lon_corr = np.arctan2(y_corr,x_corr) + lambda0

	return lat_corr, lon_corr

def xyGOES_to_latlonGRID(xGOES, yGOES, goes_imager_projection, CTH):
	# Recover BT and add it to the Earth radii
	alt = np.nan_to_num(CTH)

	# Define GOES geometrical parameters	
	lambda0 = math.radians(goes_imager_projection.longitude_of_projection_origin)
	r_eq = goes_imager_projection.semi_major_axis
	r_pol = goes_imager_projection.semi_minor_axis
	H = goes_imager_projection.perspective_point_height + r_eq

	# Compute GOES x-y-dependent geometrical parameters
	rs = v_compute_rs(xGOES,yGOES,H,r_eq,r_pol)
	sx_1 = v_compute_sx_1(rs,xGOES,yGOES)
	sy_1 = v_compute_sy_1(rs,xGOES)
	sz_1 = v_compute_sz_1(rs,xGOES,yGOES)

	# Convert from GOES x-y to lat-lon
	lat_GOES = v_compute_lat(sx_1,sy_1,sz_1,H,r_eq,r_pol)
	lon_GOES = v_compute_lon(sx_1,sy_1,H,lambda0)

	# Compute local zenith angle (viewing zenith angle from the satellite)
	lza = v_compute_lza(H, lat_GOES, lon_GOES, lambda0, r_eq)

	# Satellite cartesian coordinates
	x_sat = H
	y_sat = 0
	z_sat = 0
	# Pixel cartesian coordinates
	R_ratio = r_eq/r_pol
	lat_GOES_geoc = np.arctan(np.tan(lat_GOES)/math.pow(R_ratio,2))
	R_local = r_eq/np.sqrt(np.power(np.cos(lat_GOES_geoc),2)+np.power(R_ratio*np.sin(lat_GOES_geoc),2))
	x_GOES = R_local*np.cos(lat_GOES_geoc)*np.cos(lon_GOES-lambda0)
	y_GOES = R_local*np.cos(lat_GOES_geoc)*np.sin(lon_GOES-lambda0)
	z_GOES = R_local*np.sin(lat_GOES_geoc)
	# Compute difference vector
	x_diff = x_sat - x_GOES
	y_diff = y_sat - y_GOES
	z_diff = z_sat - z_GOES
	# R_ratio local
	R_ratio_local = np.power((r_eq+alt)/(r_pol+alt),2)
	# Compute correction factor
	quad_a = np.power(x_diff,2) + np.power(y_diff,2) + R_ratio_local*np.power(z_diff,2)
	quad_b = 2*(x_GOES*x_diff + y_GOES*y_diff + R_ratio_local*z_GOES*z_diff)
	quad_c = np.power(x_GOES,2) + np.power(y_GOES,2) + R_ratio_local*np.power(z_GOES,2) - np.power(r_eq+alt,2)
	c = (np.sqrt(np.power(quad_b,2)-4*quad_a*quad_c) - quad_b)/(2*quad_a)
	# Compute corrected cartesian coordinates
	x_corr = x_GOES + c*x_diff
	y_corr = y_GOES + c*y_diff
	z_corr = z_GOES + c*z_diff
	# Compute corrected latitude and longitude
	lat_GOES_corr = np.arctan2(R_ratio_local*z_corr,np.sqrt(np.power(x_corr,2)+np.power(y_corr,2)))
	lon_GOES_corr = np.arctan2(y_corr,x_corr) + lambda0

	return np.degrees(lat_GOES), np.degrees(lon_GOES), np.degrees(lat_GOES_corr), np.degrees(lon_GOES_corr), np.degrees(lza)

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
	BT_C08 = rad_to_T(rad_C08,ds_C08)
	ds_C09 = xr.open_dataset(os.path.join(files[1][0]))
	rad_C09 = ds_C09.Rad[::step,::step].data
	planck_C09 = [ds_C09.planck_fk1.data,ds_C09.planck_fk2.data,ds_C09.planck_bc1.data,ds_C09.planck_bc2.data]
	BT_C09 = rad_to_T(rad_C09,ds_C09)
	ds_C10 = xr.open_dataset(os.path.join(files[2][0]))
	rad_C10 = ds_C10.Rad[::step,::step].data
	planck_C10 = [ds_C10.planck_fk1.data,ds_C10.planck_fk2.data,ds_C10.planck_bc1.data,ds_C10.planck_bc2.data]
	BT_C10 = rad_to_T(rad_C10,ds_C10)
	ds_C11 = xr.open_dataset(os.path.join(files[3][0]))
	rad_C11 = ds_C11.Rad[::step,::step].data
	planck_C11 = [ds_C11.planck_fk1.data,ds_C11.planck_fk2.data,ds_C11.planck_bc1.data,ds_C11.planck_bc2.data]
	BT_C11 = rad_to_T(rad_C11,ds_C11)
	ds_C12 = xr.open_dataset(os.path.join(files[4][0]))
	rad_C12 = ds_C12.Rad[::step,::step].data
	planck_C12 = [ds_C12.planck_fk1.data,ds_C12.planck_fk2.data,ds_C12.planck_bc1.data,ds_C12.planck_bc2.data]
	BT_C12 = rad_to_T(rad_C12,ds_C12)
	ds_C13 = xr.open_dataset(os.path.join(files[5][0]))
	rad_C13 = ds_C13.Rad[::step,::step].data
	planck_C13 = [ds_C13.planck_fk1.data,ds_C13.planck_fk2.data,ds_C13.planck_bc1.data,ds_C13.planck_bc2.data]
	BT_C13 = rad_to_T(rad_C13,ds_C13)
	ds_C14 = xr.open_dataset(os.path.join(files[6][0]))
	rad_C14 = ds_C14.Rad[::step,::step].data
	planck_C14 = [ds_C14.planck_fk1.data,ds_C14.planck_fk2.data,ds_C14.planck_bc1.data,ds_C14.planck_bc2.data]
	BT_C14 = rad_to_T(rad_C14,ds_C14)
	ds_C15 = xr.open_dataset(os.path.join(files[7][0]))
	rad_C15 = ds_C15.Rad[::step,::step].data
	planck_C15 = [ds_C15.planck_fk1.data,ds_C15.planck_fk2.data,ds_C15.planck_bc1.data,ds_C15.planck_bc2.data]
	BT_C15 = rad_to_T(rad_C15,ds_C15)
	ds_C16 = xr.open_dataset(os.path.join(files[8][0]))
	rad_C16 = ds_C16.Rad[::step,::step].data
	planck_C16 = [ds_C16.planck_fk1.data,ds_C16.planck_fk2.data,ds_C16.planck_bc1.data,ds_C16.planck_bc2.data]
	BT_C16 = rad_to_T(rad_C16,ds_C16)
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
			brightness_temperature = rad_to_T(radiance, dataset)

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
