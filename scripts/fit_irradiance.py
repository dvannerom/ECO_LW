"""Fit per-channel radiance-to-BT power law coefficients (radiance = (a*BT)**b).

For each GOES ABI LW channel, the Planck blackbody spectral radiance is
integrated over 4-20 um weighted by the channel's spectral response function
(SRF), for a range of blackbody temperatures. A power law is then fit to the
resulting (temperature, band-integrated radiance) curve. This a/b pair is the
sensor-emulation model used by scripts/compute_temperature_SBDART.py to turn
simulated SBDART nadir radiances into an effective narrowband BT; it is a
distinct approximation from the operational per-file Planck-coefficient BT
conversion in src/radiometry.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from spectral_response import CHANNELS, load_goes_filters

H = 6.62607015e-34  # Planck constant, J*s
C = 299792458.0  # speed of light, m/s
KB = 1.380649e-23  # Boltzmann constant, J/K

WL_MIN = 4e-6  # integration lower bound, m
WL_MAX = 20e-6  # integration upper bound, m


def blackbody_spectral_radiance(wl, T):
	"""Planck's law spectral radiance.

	:param wl: (float) wavelength, m
	:param T: (float) temperature, K
	:return: (float) spectral radiance, W/m2/m/sr
	"""
	factor1 = (2 * np.pi * H * C ** 2) / wl ** 5
	factor2 = 1.0 / (np.exp((H * C) / (wl * KB * T)) - 1.0)
	return factor1 * factor2


def band_radiance(lambdas_nm, response, T):
	"""Band-integrated (channel-weighted) blackbody radiance at temperature T.

	:param lambdas_nm: (ndarray) SRF wavelength grid, nm
	:param response: (ndarray) SRF relative response, unitless
	:param T: (float) temperature, K
	:return: (float) channel radiance, W/m2/sr
	"""
	def integrand(wl):
		return blackbody_spectral_radiance(wl, T) * np.interp(wl * 1e9, lambdas_nm, response, left=0, right=0)

	# Integrate over the SRF's own support rather than the full 4-20 um range:
	# quad's default subdivision limit can't resolve a narrow response band
	# sitting inside a much wider, mostly-zero integration interval.
	wl_lo = max(WL_MIN, lambdas_nm.min() * 1e-9)
	wl_hi = min(WL_MAX, lambdas_nm.max() * 1e-9)
	return quad(integrand, wl_lo, wl_hi, limit=200)[0]


def power_law(x, a, b):
	return np.power(a * x, b)


def fit_channel_power_law(lambdas_nm, response, temperatures):
	"""Fit radiance = (a*T)**b against the band-integrated Planck curve.

	:param lambdas_nm: (ndarray) SRF wavelength grid, nm
	:param response: (ndarray) SRF relative response, unitless
	:param temperatures: (ndarray) temperature grid, K
	:return: (float, float, ndarray) a, b, radiance[len(temperatures)]
	"""
	radiance = np.array([band_radiance(lambdas_nm, response, T) for T in temperatures])
	(a, b), _ = curve_fit(power_law, temperatures, radiance, p0=(1e-2, 5.0))
	return a, b, radiance


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--filter-dir", type=Path, default=ROOT / "data" / "goes_channels",
	                     help="Directory holding per-channel GOES ABI spectral response files")
	parser.add_argument("--t-min", type=float, default=200.0, help="Lowest fitted temperature, K")
	parser.add_argument("--t-max", type=float, default=320.0, help="Highest fitted temperature, K")
	parser.add_argument("--t-step", type=float, default=1.0, help="Temperature grid step, K")
	parser.add_argument("--output", type=Path,
	                     default=ROOT / "data" / "models" / "channel_radiance_power_law.json",
	                     help="Output JSON path for fitted a/b coefficients")
	args = parser.parse_args()

	temperatures = np.arange(args.t_min, args.t_max + args.t_step, args.t_step)
	lambdas_goes, response_goes = load_goes_filters(args.filter_dir, CHANNELS)

	a_coeffs, b_coeffs = [], []
	for channel, lambdas_nm, response in zip(CHANNELS, lambdas_goes, response_goes):
		a, b, radiance = fit_channel_power_law(lambdas_nm, response, temperatures)
		print(f"{channel}: a={a:.8e}, b={b:.8f}")
		a_coeffs.append(a)
		b_coeffs.append(b)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	with args.output.open("w") as handle:
		json.dump({"channels": CHANNELS, "a": a_coeffs, "b": b_coeffs}, handle, indent=2)
	print(f"Wrote {args.output}")


if __name__ == "__main__":
	main()
