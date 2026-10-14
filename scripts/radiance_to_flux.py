"""ABI radiances -> ADM-corrected narrowband BT and spectral irradiance.

Applies the pooled held-in scene ADM library (component_adms/adm_k{K}.joblib,
fitted on all training days of the reviewed scene taxonomy) to every native
2 km pixel. Writes one file per spectral channel with BT_G16/BT_G18 (K) and
spectral_irradiance_G16/spectral_irradiance_G18 (W m-2 (cm-1)-1).

Pixels are NaN when the scene is unidentified (both satellites must have six
valid BTs), the radiance is invalid, or that satellite's viewing zenith angle
exceeds --max-vza (default 70 degrees, the ADM fitting domain).
Spectral irradiance is pi times the isotropic-equivalent ADM-corrected ABI
radiance, with the native mW radiance converted to W; it is not integrated
over the channel spectral response.

Memory: one slab of --chunk-rows rows (~256 x 12471 x 6 float64 per satellite,
~150 MB), a float32 spectral-irradiance output slab (~75 MB), plus netCDF
chunk caches; outputs are written slab by slab.
"""

import argparse

import numpy as np
from netCDF4 import Dataset

from abi_product import (
    SATELLITES, create_product, grid_shape, load_adm_library, planck_coefficients,
    read_satellite, row_slabs, satellite_chain, scene_labels,
)
from sensitivity import angular_integral
from product_paths import scene_id_path


def narrowband_path(day, resolution, channel):
	return f"data/narrowband_flux/narrowband_flux_{day}_res{resolution}km_C{channel}.nc"


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("-d", "--day", type=int, required=True)
	parser.add_argument("-r", "--resolution", type=int, default=2)
	parser.add_argument("--n-components", type=int, required=True)
	parser.add_argument("--adm-library", required=True,
	                    help="Pooled scene ADM library joblib (with .json sidecar)")
	parser.add_argument("--max-vza", type=float, default=70.)
	parser.add_argument("--chunk-rows", type=int, default=256)
	parser.add_argument("-l", "--lambda_center", type=float, default=-106)
	args = parser.parse_args()

	library, form, metadata = load_adm_library(args.adm_library, args.n_components)
	integral = angular_integral(library, form)
	radiance_file = f"data/preprocessed_files/abi_{args.day}_res{args.resolution}km_step1.nc"
	scene_file = scene_id_path(args.day, args.resolution, args.n_components)
	attrs = {
		"product": "ECO ADM-corrected narrowband BT and spectral irradiance",
		"spectral_irradiance": (
			"pi times isotropic-equivalent ADM-corrected ABI spectral radiance; "
			"converted from mW to W, not integrated over channel response"
		),
		"adm_library": str(args.adm_library), "adm_form": form,
		"adm_components": int(args.n_components), "adm_seed": int(metadata.get("seed", -1)),
		"max_vza_deg": float(args.max_vza),
		"masking": ("NaN where scene unidentified, radiance invalid, or satellite "
		            "VZA > max_vza_deg"),
	}
	with Dataset(radiance_file) as source, Dataset(scene_file) as scene:
		height, width = grid_shape(source)
		planck = {sat: planck_coefficients(source, sat) for sat in SATELLITES}
		targets = [create_product(narrowband_path(args.day, args.resolution, channel), height, width,
		                          ("BT_G16", "BT_G18",
		                           "spectral_irradiance_G16",
		                           "spectral_irradiance_G18", "lat", "lon"),
		                          {**attrs, "channel": channel}, args.chunk_rows)
		           for channel in range(6)]
		try:
			for start, stop in row_slabs(height, args.chunk_rows):
				rows = slice(start, stop)
				labels = scene_labels(scene, start * width, stop * width)
				lat = np.asarray(source.variables["lat_interp_grid"][rows, :])
				lon = np.asarray(source.variables["lon_interp_grid"][rows, :])
				for target in targets:
					target.variables["lat"][rows, :] = lat
					target.variables["lon"][rows, :] = lon
				for sat in SATELLITES:
					radiance, angles = read_satellite(source, sat, rows)
					spectral_irradiance = np.empty(
						(radiance.shape[0], 6), dtype=np.float32
					)
					bt, _ = satellite_chain(
						radiance, angles, labels, planck[sat], library, form,
						integral, None, args.max_vza,
						spectral_irradiance_out=spectral_irradiance,
					)
					bt = bt.reshape(stop - start, width, 6)
					spectral_irradiance = spectral_irradiance.reshape(
						stop - start, width, 6
					)
					for channel, target in enumerate(targets):
						target.variables[f"BT_{sat}"][rows, :] = bt[..., channel].astype(np.float32)
						target.variables[f"spectral_irradiance_{sat}"][
							rows, :
						] = spectral_irradiance[..., channel]
		finally:
			for target in targets:
				target.close()


if __name__ == "__main__":
	main()
