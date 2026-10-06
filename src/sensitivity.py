"""Bounded sensitivity operators; radiances and fluxes retain physical units."""

from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import LinearConstraint, minimize, minimize_scalar
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adm import elmer
from adm_fitting import fit_adm_scene
from broadband import cubic_regression
from radiometry import radiance_to_brightness_temperature
from scene_features import _local_valid_stats, spectral_features_from_bt


def read_settings(path):
    import yaml

    with Path(path).open() as source:
        settings = yaml.safe_load(source)
    if settings["schema_version"] != 1:
        raise ValueError("Unsupported sensitivity configuration")
    abi = settings["abi"]
    if set(settings["training_days"]) & set(settings["evaluation_days"]):
        raise ValueError("Training and evaluation days must be disjoint")
    if not settings["training_days"] or not settings["evaluation_days"]:
        raise ValueError("Training and evaluation days must be nonempty")
    for value in (
        settings["realizations"], abi["tiles_per_day"], abi["tile_size"],
        abi["training_points"], abi["block_size"], abi["gmm_initializations"],
        abi["max_candidate_tiles"],
    ):
        if not isinstance(value, int) or value < 1:
            raise ValueError("Counts and dimensions must be positive integers")
    if abi["halo"] < 4 * abi["block_size"]:
        raise ValueError("Halo must support the coarse 9x9 texture window")
    if abi["tile_size"] % abi["block_size"] or abi["halo"] % abi["block_size"]:
        raise ValueError("Tile and halo must be divisible by block size")
    if not 0 <= abi["second_choice_fraction"] <= 1:
        raise ValueError("Assignment fraction must be in [0,1]")
    if not 0 < abi["minimum_valid_fraction"] <= 1:
        raise ValueError("Minimum valid fraction must be in (0,1]")
    if abi["baseline_components"] not in abi["components"] or min(abi["components"]) < 2:
        raise ValueError("Include a baseline and at least two GMM components")
    if len(set(abi["components"])) != len(abi["components"]):
        raise ValueError("Component counts must be unique")
    if settings["sunny"]["noise_requirement"] not in ("goal", "threshold"):
        raise ValueError("Select goal or threshold noise requirements")
    return settings


def block_mean(values, block_size):
    """Mean complete equal-area blocks; (..., y, x, channels) is not supported."""
    values = np.asarray(values)
    height, width = values.shape[:2]
    if height % block_size or width % block_size:
        raise ValueError("Comparison requires complete spatial blocks")
    shaped = values.reshape(
        height // block_size, block_size, width // block_size, block_size,
        *values.shape[2:],
    )
    return shaped.mean(axis=(1, 3))


def tile_features(bt):
    """Production 22-feature schema from BT[tile_y,tile_x,satellite,channel]."""
    valid = np.all(np.isfinite(bt) & (bt > 0) & (bt < 1000), axis=(2, 3))
    mean_bt = bt.mean(axis=2)
    stacked = np.moveaxis(mean_bt, -1, 0)
    _, std5 = _local_valid_stats(stacked, valid, half_window=2)
    _, std9 = _local_valid_stats(stacked, valid, half_window=4)
    features = np.concatenate(
        (spectral_features_from_bt(mean_bt), np.moveaxis(std5, 0, -1),
         np.moveaxis(std9, 0, -1)), axis=-1,
    )
    return features, valid & np.all(np.isfinite(features), axis=-1)


def fit_gmm(features, components, seed, initializations):
    model = make_pipeline(
        StandardScaler(), PCA(n_components=0.98, svd_solver="full"),
        GaussianMixture(n_components=components, n_init=initializations,
                        random_state=seed, reg_covar=1e-6, max_iter=300),
    )
    model.fit(np.asarray(features, dtype=np.float64))
    if not model[-1].converged_:
        raise RuntimeError(f"{components}-component GMM did not converge")
    return model


def switch_labels(probabilities, fraction, seed):
    """Switch floor(fraction*n) distinct valid records; component IDs stay native."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Need at least two components")
    if (not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Invalid posterior probabilities")
    if not 0 <= fraction <= 1:
        raise ValueError("Assignment fraction must be in [0,1]")
    order = np.argsort(-probabilities, axis=1, kind="stable")
    labels = order[:, 0].copy()
    count = int(np.floor(fraction * labels.size))
    selected = np.random.default_rng(seed).choice(labels.size, count, replace=False)
    labels[selected] = order[selected, 1]
    return labels, selected


def angular_basis(angles, form):
    first = elmer(np.asarray(angles)) - 1
    if form == "regularized":
        return first[..., None]
    if form == "quadratic":
        cosine = np.cos(np.deg2rad(angles))
        return np.stack((first, cosine**2 - np.cos(np.deg2rad(55))**2), axis=-1)
    raise ValueError(f"Unknown ADM form {form}")


def angular_integral(parameters, form):
    """Return 2*integral(shape*cos(theta)*sin(theta) dtheta)."""
    angles = np.linspace(0, 90, 901)
    basis = angular_basis(angles, form)
    parameters = np.asarray(parameters)
    flat = parameters.reshape(-1, parameters.shape[-1])
    if not np.all(np.isfinite(flat)):
        raise ValueError("Non-finite ADM parameters")
    for start in range(0, len(flat), 2048):
        shape = 1 + flat[start:start + 2048] @ basis.T
        if np.any(shape <= 0):
            raise ValueError("ADM angular profile is not positive over the hemisphere")
    theta = np.deg2rad(angles)
    weights = 2 * np.cos(theta) * np.sin(theta)
    integrated_basis = simpson(basis * weights[:, None], x=theta, axis=0)
    return simpson(weights, x=theta) + parameters @ integrated_basis


def fit_scene_adm(angles, radiance, form):
    """Fit pooled paired ABI ratios; shape parameters are shared by a scene."""
    if radiance.shape[0] < 100:
        raise ValueError("Scene ADM needs at least 100 training pairs")
    if np.std(angles[:, 0] - angles[:, 1]) < 0.1:
        raise ValueError("Scene ADM training angles lack diversity")
    if form == "regularized":
        b, _ = fit_adm_scene(
            angles[:, 0], angles[:, 1], radiance[:, 0], radiance[:, 1]
        )
        parameters = np.array([b])
        grid = angular_basis(np.linspace(0, 90, 901), form)[:, 0]
        if np.any(1 + b * grid < 1e-6):
            lower = np.max((1e-6 - 1) / grid[grid > 0])
            upper = np.min((1e-6 - 1) / grid[grid < 0])
            basis = angular_basis(angles, form)[..., 0]
            ratio = radiance[:, 0] / radiance[:, 1]

            def objective(value):
                residual = (1 + basis[:, 0] * value) / (1 + basis[:, 1] * value) - ratio
                return np.mean(residual**2)

            result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
            if not result.success:
                raise RuntimeError("Positive regularized ABI ADM fit failed")
            parameters = np.array([result.x])
    else:
        b16 = angular_basis(angles[:, 0], form)
        b18 = angular_basis(angles[:, 1], form)
        ratio = radiance[:, 0] / radiance[:, 1]
        grid = angular_basis(np.linspace(0, 90, 901), form)

        def objective(parameters):
            numerator = 1 + b16 @ parameters
            denominator = 1 + b18 @ parameters
            residual = numerator / denominator - ratio
            jacobian = (b16 * denominator[:, None] - numerator[:, None] * b18) / denominator[:, None]**2
            return .5 * np.mean(residual**2), jacobian.T @ residual / len(residual)

        result = minimize(
            objective, np.zeros(2), jac=True, method="SLSQP",
            constraints=[LinearConstraint(grid, 1e-6 - 1, np.inf)],
            options={"ftol": 1e-15, "maxiter": 200},
        )
        if not result.success:
            raise RuntimeError("Higher-order pooled ADM fit failed or is unidentifiable")
        parameters = result.x
        numerator = 1 + b16 @ parameters
        denominator = 1 + b18 @ parameters
        jacobian = (b16 * denominator[:, None] - numerator[:, None] * b18) / denominator[:, None]**2
        if np.linalg.cond(jacobian) > 1e8:
            raise ValueError("Higher-order pooled ABI ADM is ill-conditioned")
    angular_integral(parameters, form)
    return parameters


def retrieve_sunny(radiance, angles, form, diagnostics=None):
    """Return band flux[scene,channel] from radiance[scene,angle,channel]."""
    reference = np.flatnonzero(np.isclose(angles, 55))
    if reference.size != 1:
        raise ValueError("Sunny fit requires exactly one 55-degree view")
    amplitude = radiance[:, reference[0], :]
    if np.any(~np.isfinite(radiance)) or np.any(radiance <= 0):
        raise ValueError("Non-positive or invalid perturbed radiance")
    basis = angular_basis(angles, form)
    if np.linalg.matrix_rank(basis) != basis.shape[1]:
        raise ValueError("ADM geometry does not identify the shape parameters")
    normalized = radiance / amplitude[:, None, :]
    parameters = np.ascontiguousarray(np.einsum(
        "pa,sac->scp", np.linalg.pinv(basis), normalized - 1
    ))
    grid = angular_basis(np.linspace(0, 90, 901), form)
    flat = parameters.reshape(-1, parameters.shape[-1])
    invalid = []
    for start in range(0, len(flat), 2048):
        bad = np.flatnonzero(np.any(1 + flat[start:start + 2048] @ grid.T < 1e-6, axis=1))
        invalid.extend((bad + start).tolist())
    for index in invalid:
        scene, channel = np.unravel_index(index, parameters.shape[:2])
        target = normalized[scene, :, channel] - 1
        if form == "regularized":
            values = grid[:, 0]
            lower = np.max((1e-6 - 1) / values[values > 0])
            upper = np.min((1e-6 - 1) / values[values < 0])
            flat[index, 0] = np.clip(flat[index, 0], lower, upper)
        else:
            def objective(value):
                residual = basis @ value - target
                return .5 * np.dot(residual, residual)

            result = minimize(
                objective, np.zeros(2),
                jac=lambda value: basis.T @ (basis @ value - target),
                constraints=[LinearConstraint(grid, 1e-6 - 1, np.inf)],
                method="SLSQP", options={"ftol": 1e-12, "maxiter": 200},
            )
            if not result.success:
                raise RuntimeError(f"Positive Sunny ADM fit failed: {result.message}")
            flat[index] = result.x
    if diagnostics is not None:
        diagnostics["positivity_constrained_scene_channels"] = len(invalid)
        diagnostics["scene_channels"] = len(flat)
        diagnostics["positivity_grid_deg"] = 0.1
    integral = angular_integral(parameters, form)
    fitted = 1 + np.einsum("ap,scp->sac", basis, parameters)
    if np.any(fitted <= 0):
        raise ValueError("ADM fitted retrieval profile is non-positive")
    return np.pi * np.mean(radiance / fitted, axis=1) * integral


def abi_broadband(radiance, angles, planck, labels, library, form, coefficients):
    """ABI chain on valid records: radiance[n,2,6] -> flux[n,2] W m-2."""
    params = library[labels]  # n, channel, parameter
    norm = angular_integral(library, form)[labels]
    basis = angular_basis(angles, form)
    shape = 1 + np.einsum("nsp,ncp->nsc", basis, params)
    corrected = radiance * norm[:, None, :] / shape
    if np.any(corrected <= 0) or np.any(~np.isfinite(corrected)):
        raise ValueError("Invalid corrected ABI radiance")
    flux = np.empty((radiance.shape[0], 2))
    for satellite in range(2):
        bt = np.column_stack([
            radiance_to_brightness_temperature(
                corrected[:, satellite, channel], planck[satellite, channel]
            ) for channel in range(6)
        ])
        effective_bt = cubic_regression(bt, *coefficients)
        if np.any(~np.isfinite(effective_bt)) or np.any(effective_bt <= 0):
            raise ValueError("Invalid ABI broadband effective temperature")
        flux[:, satellite] = 5.670374e-8 * effective_bt**4
    return flux


def planck_derivative(wavelength_um, temperature):
    """Spectral radiance derivative [W m-2 sr-1 um-1 K-1]."""
    wavelength = np.asarray(wavelength_um, dtype=float) * 1e-6
    h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    exponent = h * c / (wavelength * k * temperature)
    inverse = np.exp(-exponent)
    planck = (2 * h * c**2 / wavelength**5) * inverse / (1 - inverse) * 1e-6
    return planck * exponent / (temperature * (1 - inverse))


def summary(values):
    """Signed mean, sample SD and RMSE; invalid values fail rather than disappear."""
    values = np.asarray(values, dtype=float).ravel()
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Need at least two finite paired residuals")
    return {
        "count": int(values.size), "bias": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "rmse": float(np.sqrt(np.mean(values**2))),
    }


def paired_summary(variant, baseline, reference=None):
    if variant.shape != baseline.shape:
        raise ValueError("Paired outputs have different shapes")
    denominator = baseline if reference is None else reference
    if np.any(denominator <= 0):
        raise ValueError("Relative residual needs positive reference flux")
    return {
        "w_m2": summary(variant - baseline),
        "percent": summary((variant - baseline) / denominator * 100),
    }
