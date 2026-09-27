"""Fit narrowband-to-broadband regression coefficients from the SBDART "Sunny" scene library.

For every simulated clear-sky/cloudy scene in data/Sunny/, the TOA broadband
flux and per-GOES-channel fluxes are integrated from the upward spectral flux,
converted to an effective narrowband brightness temperature via a
sensor-emulation power law, and regressed (linear/quadratic/cubic polynomial)
against the broadband-equivalent temperature. The fitted cubic coefficients
are what src/broadband.py's cubic_regression() applies operationally.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import integrate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from spectral_response import CHANNELS, load_goes_filters, load_power_law

SIGMA = 5.670374e-08  # Stefan-Boltzmann constant, W/m2/K4

# Fallback power-law sensor-emulation coefficients (radiance[W/m2/um/sr] =
# (a * BT[K])**b), used only if --power-law-file (scripts/fit_irradiance.py
# output) is unavailable. Used only to turn simulated SBDART nadir radiances
# into an effective narrowband BT; the operational pipeline instead converts
# real ABI radiances to BT with the exact per-file Planck coefficients
# (src/radiometry.py). The two conversions are independent approximations of
# the same channel and are not guaranteed to agree bit-for-bit; this is a
# known modeling simplification.
DEFAULT_CHANNEL_A = np.array([4.63213883e-03, 5.05040878e-03, 5.27956064e-03, 0.00645188, 0.00693793, 0.00632944])
DEFAULT_CHANNEL_B = np.array([7.99453640e+00, 6.07740875e+00, 5.41373985e+00, 4.75053341, 4.38290917, 4.10502937])


def load_sunny_scene(path, lambdas_goes, response_goes):
	"""Integrate one SBDART spectrum into broadband and GOES-channel fluxes.

	:param path: (Path) sunny_lw_* file. Columns: wavelength[um], upward TOA
		flux[W/m2/um], nadir (0 deg VZA) radiance[W/m2/um/sr], ...
	:param lambdas_goes: (list[ndarray]) per-channel filter wavelengths[um]
	:param response_goes: (list[ndarray]) per-channel filter responses
	:return: (float, ndarray[6]) broadband flux[W/m2], SRF-weighted channel fluxes[W/m2]
	"""
	data = np.loadtxt(path)
	wavelength = data[:, 0]
	flux = data[:, 1]
	flux_bb = integrate.simpson(flux, wavelength, axis=0)

	response = np.stack(
		[
			np.interp(wavelength, lam, resp, left=0, right=0)
			for lam, resp in zip(lambdas_goes, response_goes)
		],
		axis=1,
	)
	flux_nb = integrate.simpson(flux[:, None] * response, wavelength, axis=0)
	return flux_bb, flux_nb


def fit_regressions(t_narrowband, t_broadband):
	"""Fit linear/quadratic/cubic polynomial regressions of broadband BT on channel BTs.

	:param t_narrowband: (ndarray, shape (n_scenes, 6)) per-channel effective BT[K]
	:param t_broadband: (ndarray, shape (n_scenes,)) broadband-equivalent BT[K]
	:return: dict of {"linear"|"quadratic"|"cubic": {"coefficients": [...], "intercept": float}}
	"""
	models = {}
	for name, degree in (("linear", 1), ("quadratic", 2), ("cubic", 3)):
		poly = PolynomialFeatures(degree=degree, include_bias=False)
		pipeline = make_pipeline(poly, LinearRegression())
		pipeline.fit(t_narrowband, t_broadband)
		linreg = pipeline.named_steps["linearregression"]
		models[name] = {
			"coefficients": linreg.coef_.tolist(),
			"intercept": float(linreg.intercept_),
		}
	return models


def main():
	parser = argparse.ArgumentParser(
		description="Fit narrowband-to-broadband regression coefficients from the SBDART Sunny scene library."
	)
	parser.add_argument("--sunny-dir", type=Path, default=ROOT / "data" / "Sunny",
	                     help="Directory holding radiance_lw_cs/ and radiance_lw_cl/ subfolders")
	parser.add_argument("--filter-dir", type=Path, default=ROOT / "data" / "goes_channels",
	                     help="Directory holding per-channel GOES ABI spectral response files")
	parser.add_argument("--output", type=Path,
	                     default=ROOT / "data" / "models" / "narrowband_to_broadband_coeffs.json",
	                     help="Output JSON path for fitted coefficients")
	parser.add_argument("--power-law-file", type=Path,
	                     default=ROOT / "data" / "models" / "channel_radiance_power_law.json",
	                     help="JSON from scripts/fit_irradiance.py; falls back to built-in defaults if missing")
	args = parser.parse_args()

	lambdas_goes, response_goes = load_goes_filters(args.filter_dir, CHANNELS)
	lambdas_goes = [lam / 1000.0 for lam in lambdas_goes]  # nm -> um, to match the Sunny wavelength grid

	if args.power_law_file.exists():
		_, channel_a, channel_b = load_power_law(args.power_law_file)
	else:
		print(f"{args.power_law_file} not found, using built-in default power-law coefficients")
		channel_a, channel_b = DEFAULT_CHANNEL_A, DEFAULT_CHANNEL_B

	scene_files = sorted((args.sunny_dir / "radiance_lw_cs").glob("*")) + \
		sorted((args.sunny_dir / "radiance_lw_cl").glob("*"))
	if not scene_files:
		raise FileNotFoundError(f"No Sunny scene files found under {args.sunny_dir}")

	flux_bb = np.empty(len(scene_files))
	radiance_nb = np.empty((len(scene_files), len(CHANNELS)))
	for i, scene_file in enumerate(scene_files):
		flux_bb[i], radiance_nb[i] = load_sunny_scene(scene_file, lambdas_goes, response_goes)

	t_broadband = np.power(flux_bb / SIGMA, 1.0 / 4)
	t_narrowband = np.power(radiance_nb, 1.0 / channel_b) / channel_a

	models = fit_regressions(t_narrowband, t_broadband)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	with args.output.open("w") as handle:
		json.dump({"channels": CHANNELS, "n_scenes": len(scene_files), **models}, handle, indent=2)
	print(f"Wrote {args.output}")

	cubic_poly = PolynomialFeatures(degree=3, include_bias=False)
	t_narrowband_poly = cubic_poly.fit_transform(t_narrowband)
	t_broadband_cubic_pred = (
		t_narrowband_poly @ np.asarray(models["cubic"]["coefficients"]) + models["cubic"]["intercept"]
	)
	flux_cubic_pred = SIGMA * np.power(t_broadband_cubic_pred, 4)
	flux_err = (flux_bb - flux_cubic_pred) / flux_bb
	print(f"Cubic fit: OLR error mean={np.mean(flux_err):.4f}, stddev={np.std(flux_err):.4f}")


if __name__ == "__main__":
	main()

