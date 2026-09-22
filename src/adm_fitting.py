"""ADM parameter fitting, matching the production fit_ADM.py calculation."""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

from adm import radiance_integrand, radiance_linear, radiance_linear_ratio


def fit_adm_scene(lza16, lza18, rad16, rad18):
    """Fit the ADM b-parameter and unity-integral normalization for one scene/channel."""
    ratio = np.divide(rad16, rad18, out=np.zeros_like(rad16), where=rad18 != 0)
    popt, _ = curve_fit(radiance_linear_ratio, (lza16, lza18), ratio)
    b = float(popt[0])
    norm = 1.0 / quad(radiance_integrand, 0, 90, args=(b,))[0]
    return b, norm


def correct_radiance(rad, lza, b, norm):
    """Apply the fitted ADM correction, matching radiance_to_flux.py."""
    return rad / (norm * radiance_linear(lza, b))
