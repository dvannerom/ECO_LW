"""Narrowband BTs -> daily broadband LW flux (W m-2) per satellite.

Evaluates the N2B polynomial (default: cubic, data/models/
narrowband_to_broadband_coeffs.json) in float64 and F = sigma*T_eff^4.
NaN where any channel BT is NaN (unidentified scene, invalid radiance or
VZA > max) or flux is outside [50, 1000] W m-2.

Memory: one slab of --chunk-rows rows per satellite ((n, 6) float64).
"""

import argparse

import numpy as np
from netCDF4 import Dataset

from abi_product import SATELLITES, broadband_flux, create_product, row_slabs
from broadband import load_regression_coefficients


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("-d", "--day", type=int, required=True)
	parser.add_argument("-r", "--resolution", type=int, default=2)
	parser.add_argument("-l", "--lambda_center", type=float, default=-106)
	parser.add_argument("--coefficients-file", default="data/models/narrowband_to_broadband_coeffs.json")
	parser.add_argument("--form", default="cubic", choices=("linear", "quadratic", "cubic"))
	parser.add_argument("--chunk-rows", type=int, default=256)
	args = parser.parse_args()

	regression = load_regression_coefficients(args.coefficients_file, args.form)
	inputs = [Dataset(f"data/narrowband_flux/narrowband_flux_{args.day}_res{args.resolution}km_C{channel}.nc")
	          for channel in range(6)]
	try:
		height, width = inputs[0].variables["lat"].shape
		attrs = {"product": "ECO daily broadband flux", "n2b_form": args.form,
		         "n2b_coefficients": args.coefficients_file,
		         "flux_range_w_m2": "50 1000", "precision": "float64 evaluation, float32 storage"}
		for name in ("adm_library", "adm_form", "max_vza_deg"):
			if name in inputs[0].ncattrs():
				attrs[name] = inputs[0].getncattr(name)
		target = create_product(f"data/broadband_flux/broadband_flux_{args.day}_res{args.resolution}km.nc",
		                        height, width, ("flux_G16", "flux_G18", "lat", "lon"), attrs, args.chunk_rows)
		try:
			for start, stop in row_slabs(height, args.chunk_rows):
				rows = slice(start, stop)
				target.variables["lat"][rows, :] = inputs[0].variables["lat"][rows, :]
				target.variables["lon"][rows, :] = inputs[0].variables["lon"][rows, :]
				for sat in SATELLITES:
					bt = np.empty((stop - start, width, 6))
					for channel, source in enumerate(inputs):
						bt[..., channel] = np.ma.filled(
							source.variables[f"BT_{sat}"][rows, :].astype(np.float64), np.nan)
					bt = bt.reshape(-1, 6)
					flux = np.full(len(bt), np.nan)
					finite = np.all(np.isfinite(bt), axis=1)
					if finite.any():
						flux[finite] = broadband_flux(bt[finite], regression)
					target.variables[f"flux_{sat}"][rows, :] = flux.reshape(stop - start, width).astype(np.float32)
		finally:
			target.close()
	finally:
		for source in inputs:
			source.close()


if __name__ == "__main__":
	main()
