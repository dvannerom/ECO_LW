"""Population-weighted scene-library retrieval and error-class statistics."""

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar

from sensitivity import angular_basis, angular_integral


def integrate_domain(wavelength, values, lower, upper):
    """Integrate spectral flux in an exact bounded domain with interpolated edges."""
    wavelength, values = np.asarray(wavelength), np.asarray(values)
    if (wavelength.ndim != 1 or values.shape != wavelength.shape
            or np.any(~np.isfinite(values)) or np.any(np.diff(wavelength) <= 0)
            or not wavelength[0] <= lower < upper <= wavelength[-1]):
        raise ValueError("Invalid spectral grid, values or unsupported reference domain")
    selected = (wavelength > lower) & (wavelength < upper)
    grid = np.r_[lower, wavelength[selected], upper]
    flux = np.r_[np.interp(lower, wavelength, values), values[selected],
                 np.interp(upper, wavelength, values)]
    return float(simpson(flux, x=grid))


def file_scene(probabilities):
    """Majority of view labels; count ties use mean posterior then lowest ID."""
    probabilities = np.asarray(probabilities)
    if (probabilities.ndim != 3 or np.any(~np.isfinite(probabilities))
            or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=-1), 1)):
        raise ValueError("Expected valid posteriors[file,view,component]")
    labels = probabilities.argmax(axis=-1)
    counts = np.stack([(labels == k).sum(axis=1)
                       for k in range(probabilities.shape[-1])], axis=1)
    tied = counts == counts.max(axis=1, keepdims=True)
    chosen = np.where(tied, probabilities.mean(axis=1), -1).argmax(axis=1)
    return chosen, counts.max(axis=1)/labels.shape[1]


def population_weights(labels, population):
    """Equal weight within scenes, matching supported ABI scene masses.

    Missing support is explicitly returned. Results conditional on covered
    scenes must never be described as a full-population budget.
    """
    labels, population = np.asarray(labels), np.asarray(population, dtype=float)
    if (labels.ndim != 1 or np.any(labels < 0) or np.any(labels >= len(population))
            or np.any(population < 0) or not np.isclose(population.sum(), 1)):
        raise ValueError("Invalid scene labels or population")
    counts = np.bincount(labels, minlength=len(population))
    covered = counts > 0
    coverage = float(population[covered].sum())
    if coverage == 0:
        raise ValueError("No supported ABI population")
    weights = population[labels]/counts[labels]/coverage
    return weights, {
        "covered_population": coverage, "missing_scenes": np.flatnonzero(~covered & (population > 0)).tolist(),
        "scene_counts": counts.tolist(),
        "effective_sample_size": float(1/np.square(weights).sum()),
    }


def fit_pooled_shape(radiance, angles):
    """Fit one shared shape/channel with separate per-file amplitudes.

    Equal relative radiance residual weighting gives each file equal influence,
    independent of its brightness. Input shape (file,view,channel).
    """
    radiance = np.asarray(radiance, dtype=float)
    if radiance.ndim != 3 or not len(radiance) or np.any(radiance <= 0) or np.any(~np.isfinite(radiance)):
        raise ValueError("Pooled ADM requires positive finite training radiances")
    basis = angular_basis(angles, "regularized")[:, 0]
    grid = angular_basis(np.linspace(0, 90, 901), "regularized")[:, 0]
    lower = np.max((1e-6-1)/grid[grid > 0])
    upper = np.min((1e-6-1)/grid[grid < 0])
    coefficients = []
    for channel in range(radiance.shape[2]):
        target = radiance[:, :, channel]/radiance[:, :, channel].mean(axis=1, keepdims=True)

        def objective(b):
            profile = 1+b*basis
            amplitude = target @ profile / np.dot(profile, profile)
            return np.mean((target-amplitude[:, None]*profile)**2)

        fit = minimize_scalar(objective, bounds=(lower, upper), method="bounded",
                              options={"xatol": 1e-10})
        if not fit.success:
            raise RuntimeError("Pooled shape fit failed")
        coefficients.append(fit.x)
    return np.asarray(coefficients)[:, None]


def library_flux(radiance, angles, labels, library):
    """Fixed scene profile with equal-weight least-squares amplitude per channel."""
    parameters = library[np.asarray(labels)]
    shape = 1+np.einsum("vp,scp->svc", angular_basis(angles, "regularized"), parameters)
    amplitude = (radiance*shape).sum(axis=1)/np.square(shape).sum(axis=1)
    flux = np.pi*amplitude*angular_integral(library, "regularized")[labels]
    if np.any(~np.isfinite(flux)) or np.any(flux <= 0):
        raise ValueError("Invalid scene-library flux")
    return flux


def error_classes(errors, weights):
    """Population moments for errors[file,realization]; correct finite-MC scene variance.

    Random variance is estimated with ddof=1 within each file. The weighted
    variance of file means is corrected for finite Monte Carlo mean noise.
    Negative corrected estimates are disclosed, not silently interpreted.
    """
    errors = np.asarray(errors, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (errors.ndim != 2 or len(weights) != len(errors) or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1) or np.any(~np.isfinite(errors))):
        raise ValueError("Invalid residual ensemble or normalized weights")
    count = errors.shape[1]
    mean = errors.mean(axis=1)
    bias = float(weights @ mean)
    variance = errors.var(axis=1, ddof=1) if count > 1 else np.zeros(len(mean))
    random = float(weights @ variance)
    raw_scene = float(weights @ np.square(mean-bias))
    correction = float(np.sum(weights*(1-weights)*variance)/count) if count > 1 else 0.
    corrected = raw_scene-correction
    return {
        "bias_w_m2": bias, "scene_sd_w_m2": float(np.sqrt(max(corrected, 0))),
        "random_sd_w_m2": float(np.sqrt(random)),
        "ensemble_rmse_w_m2": float(np.sqrt(weights @ np.mean(errors**2, axis=1))),
        "class_rmse_w_m2": float(np.sqrt(bias*bias+max(corrected, 0)+random)),
        "raw_scene_variance": raw_scene, "mc_scene_variance_correction": correction,
        "negative_corrected_scene_variance": bool(corrected < 0),
        "realizations": count,
    }


def independent_budget_sum(first, second):
    """Add signed biases and independent variances from different populations.

    RMSE uses ensemble second moments, not finite-MC-corrected class variances.
    This is an assumption-based combination, not a directly paired ensemble.
    """
    bias = first["bias_w_m2"]+second["bias_w_m2"]
    scene_variance = first["scene_sd_w_m2"]**2+second["scene_sd_w_m2"]**2
    random_variance = first["random_sd_w_m2"]**2+second["random_sd_w_m2"]**2
    variance = sum(row["ensemble_rmse_w_m2"]**2-row["bias_w_m2"]**2
                   for row in (first, second))
    if variance < -1e-10:
        raise ValueError("Invalid source second moments for independent combination")
    return {
        "bias_w_m2": bias, "scene_sd_w_m2": float(np.sqrt(scene_variance)),
        "random_sd_w_m2": float(np.sqrt(random_variance)),
        "ensemble_rmse_w_m2": float(np.sqrt(bias*bias+max(variance, 0))),
        "class_rmse_w_m2": float(np.sqrt(bias*bias+scene_variance+random_variance)),
        "negative_corrected_scene_variance": False,
        "combination": "Independent spatial increment; signed biases added; not a paired ensemble",
    }
