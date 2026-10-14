"""Shared ABI production operators: scene-ADM correction, BT and N2B flux.

The production scripts (radiance_to_flux.py, narrowband_to_broadband.py) and
the production-aligned sensitivity study both call these functions, so the
sensitivity baseline *is* the production algorithm and closes on the
written daily product.

Chain for one satellite and pixel (scene s, channel c, viewing zenith θ):

    L_iso[c] = L[c] * I[s, c] / (1 + basis(θ) . p[s, c])
    BT[c]    = Planck^-1(L_iso[c])                                  [K]
    T_eff    = poly(BT[0..5])                                       [K]
    F        = σ T_eff^4                                            [W m-2]

with I the hemispheric integral of the scene ADM shape (sensitivity.
angular_integral). Pixels are not eligible (NaN outputs) when the scene
identification is missing, the radiance is non-positive or non-finite, or
θ exceeds ``max_vza_deg``. Fluxes outside FLUX_RANGE_W_M2 are NaN.

Memory: functions work on flat (n, 6) chunks; callers stream row slabs so the
peak is a few (n, 6) float64 arrays per chunk, never a full-grid stack.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from netCDF4 import Dataset

from broadband import polynomial_regression
from scene_features import SPECTRAL_CHANNEL_INDICES
from sensitivity import angular_basis, angular_integral

SIGMA_W_M2_K4 = 5.670374e-8
FLUX_RANGE_W_M2 = (50., 1000.)
SATELLITES = ("G16", "G18")
ADM_PARAMETERS = {"regularized": 1, "quadratic": 2}


def load_adm_library(path, components=None):
    """Return (library[K, 6, P], form, metadata) from a joblib ADM library.

    The sidecar ``path.json`` records the angular ``form`` (default
    "regularized", P=1; "quadratic" has P=2).
    """
    path = Path(path)
    metadata = json.loads(Path(str(path) + ".json").read_text())
    form = metadata.get("form", "regularized")
    if form not in ADM_PARAMETERS:
        raise ValueError(f"Unknown ADM form {form} in {path}")
    library = np.asarray(joblib.load(path), dtype=np.float64)
    if (library.ndim != 3 or library.shape[1:] != (6, ADM_PARAMETERS[form])
            or not np.isfinite(library).all()):
        raise ValueError(f"Invalid {form} ADM library {path}: shape {library.shape}")
    if components is not None and library.shape[0] != components:
        raise ValueError(f"ADM library {path} has K={library.shape[0]}, expected {components}")
    angular_integral(library, form)  # positivity over the hemisphere
    return library, form, metadata


def adm_correct(radiance, angles, labels, library, form, integral=None):
    """Isotropic-equivalent radiance[n, 6] (input units) for valid labels.

    radiance[n, 6] ABI radiance; angles[n] viewing zenith in degrees;
    labels[n] scene index in [0, K); integral[K, 6] from angular_integral.
    """
    if integral is None:
        integral = angular_integral(library, form)
    basis = angular_basis(np.asarray(angles, dtype=np.float64), form)
    shape = 1. + np.einsum("np,ncp->nc", basis, library[labels])
    if np.any(shape <= 0):
        raise ValueError("Non-positive ADM shape at an eligible pixel")
    return radiance * integral[labels] / shape


def brightness_temperature(radiance, planck):
    """BT[n, 6] in K from radiance[n, 6] and Planck coefficients planck[6, 4].

    Non-positive or non-finite radiance gives NaN (no warning).
    """
    fk1, fk2, bc1, bc2 = np.asarray(planck, dtype=np.float64).T
    radiance = np.asarray(radiance, dtype=np.float64)
    positive = np.isfinite(radiance) & (radiance > 0)
    safe = np.where(positive, radiance, 1.)
    bt = (fk2 / np.log1p(fk1 / safe) - bc1) / bc2
    bt[~positive] = np.nan
    return bt


def broadband_flux(bt, regression):
    """Broadband flux[n] in W m-2 from BT[n, 6]; regression=(coeff, intercept, degree).

    Evaluated in float64. NaN where any BT is NaN, the effective temperature is
    non-positive, or the flux is outside FLUX_RANGE_W_M2.
    """
    effective = polynomial_regression(bt, *regression)
    flux = SIGMA_W_M2_K4 * effective**4
    lower, upper = FLUX_RANGE_W_M2
    flux[~((effective > 0) & (flux >= lower) & (flux <= upper))] = np.nan
    return flux


def eligible_pixels(radiance, angles, labels, max_vza_deg):
    """Boolean[n]: scene identified, VZA in [0, max], all six radiances > 0."""
    return ((labels >= 0) & np.isfinite(angles) & (angles >= 0) & (angles <= max_vza_deg)
            & np.all(np.isfinite(radiance) & (radiance > 0), axis=1))


def satellite_chain(radiance, angles, labels, planck, library, form, integral,
                    regression, max_vza_deg, spectral_irradiance_out=None):
    """Production chain for one satellite on a flat chunk.

    Returns (bt[n, 6] float64, flux[n] float64), NaN where not eligible.
    ``regression=None`` skips the broadband step (flux is None). If provided,
    spectral_irradiance_out[n, 6] receives pi times the ADM-corrected spectral
    radiance, converted from ABI mW to W, in W m-2 (cm-1)-1.
    """
    n = len(labels)
    if spectral_irradiance_out is not None:
        if spectral_irradiance_out.shape != (n, 6):
            raise ValueError(
                "spectral_irradiance_out must have shape "
                f"{(n, 6)}, got {spectral_irradiance_out.shape}"
            )
        spectral_irradiance_out[...] = np.nan
    bt = np.full((n, 6), np.nan)
    eligible = eligible_pixels(radiance, angles, labels, max_vza_deg)
    if eligible.any():
        corrected = adm_correct(radiance[eligible], angles[eligible], labels[eligible],
                                library, form, integral)
        bt[eligible] = brightness_temperature(corrected, planck)
        if spectral_irradiance_out is not None:
            corrected *= np.pi / 1000.
            spectral_irradiance_out[eligible] = corrected
    if regression is None:
        return bt, None
    flux = np.full(n, np.nan)
    finite = np.all(np.isfinite(bt), axis=1)
    if finite.any():
        flux[finite] = broadband_flux(bt[finite], regression)
    return bt, flux


def scene_labels(scene, start, stop):
    """Flat labels[start:stop] as int64 with -1 where the scene is unidentified."""
    labels = np.asarray(scene.variables["labels"][start:stop], dtype=np.int64)
    identified = np.isfinite(np.ma.filled(
        scene.variables["sorted_labels"][start:stop].astype(np.float64), np.nan))
    labels[~identified] = -1
    return labels


def planck_coefficients(source, satellite):
    """Planck coefficients[6, 4] for the six spectral channels."""
    return np.asarray(source.variables[f"planck_{satellite}"][:],
                      dtype=np.float64)[list(SPECTRAL_CHANNEL_INDICES)]


def read_satellite(source, satellite, rows, name="rad_{}_interp_corr"):
    """Flat (n, 6) float64 spectral-channel values and (n,) VZA for row slice."""
    values = np.ma.filled(source.variables[name.format(satellite)][
        rows, :, list(SPECTRAL_CHANNEL_INDICES)].astype(np.float64), np.nan)
    angles = np.ma.filled(source.variables[f"lza_{satellite}_interp_corr"][
        rows, :].astype(np.float64), np.nan)
    return values.reshape(-1, 6), angles.ravel()


def row_slabs(height, rows):
    """(start, stop) row ranges covering [0, height)."""
    if rows < 1:
        raise ValueError("rows must be positive")
    return [(start, min(start + rows, height)) for start in range(0, height, rows)]


def grid_shape(source):
    return source.variables["lat_interp_grid"].shape


def create_product(path, height, width, variables, attrs, chunk_rows):
    """Open a NetCDF4 product with (y, x) float32 variables for slab writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = Dataset(path, "w", format="NETCDF4")
    target.createDimension("y", height)
    target.createDimension("x", width)
    chunks = (min(chunk_rows, height), min(256, width))
    for name in variables:
        variable = target.createVariable(name, "f4", ("y", "x"), zlib=True, complevel=1,
                                         shuffle=True, chunksizes=chunks, fill_value=np.nan)
        if name.startswith("BT_"):
            variable.units = "K"
        elif name.startswith("flux_"):
            variable.units = "W m-2"
        elif name.startswith("spectral_irradiance_"):
            variable.units = "W m-2 (cm-1)-1"
        elif name in ("lat", "lon"):
            variable.units = "degrees"
    target.setncatts(attrs)
    return target
