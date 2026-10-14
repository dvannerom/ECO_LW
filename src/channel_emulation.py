"""ABI band-integrated blackbody power-law emulation, distinct from operational BT.

DEFAULT_CHANNEL_A/B retain the legacy optional emulation coefficients; production
fits and passes its saved coefficients instead of relying on those defaults.
"""

import numpy as np
from scipy.integrate import quad, simpson
from scipy.optimize import curve_fit

H = 6.62607015e-34  # Planck constant, J*s
C = 299792458.0  # speed of light, m/s
KB = 1.380649e-23  # Boltzmann constant, J/K

WL_MIN = 4e-6  # integration lower bound, m
WL_MAX = 20e-6  # integration upper bound, m

SIGMA = 5.670374e-08  # Stefan-Boltzmann constant, W/m2/K4
DEFAULT_CHANNEL_A = np.array([4.63213883e-03, 5.05040878e-03, 5.27956064e-03, 0.00645188, 0.00693793, 0.00632944])
DEFAULT_CHANNEL_B = np.array([7.99453640e+00, 6.07740875e+00, 5.41373985e+00, 4.75053341, 4.38290917, 4.10502937])


def load_sunny_scene(path, lambdas_goes, response_goes):
	"""Integrate one Sunny file into broadband and SRF-weighted channel fluxes.

	File columns: wavelength [um], upward TOA spectral flux [W m-2 um-1],
	then directional radiances. Filter wavelengths must be in um.
	Returns broadband scalar and channel[6] fluxes, both in W m-2.
	Memory scales with one spectrum and its wavelength-by-channel responses.
	"""
	data = np.loadtxt(path)
	wavelength, flux = data[:, 0], data[:, 1]
	response = np.stack([
		np.interp(wavelength, lam, resp, left=0, right=0)
		for lam, resp in zip(lambdas_goes, response_goes)
	], axis=1)
	return (simpson(flux, x=wavelength),
	        simpson(flux[:, None] * response, x=wavelength, axis=0))


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
