"""Bounded-memory spectral-only ABI/Sunny scene comparison utilities."""

from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from parallel_gmm import ParallelGaussianMixture
from radiometry import radiance_to_brightness_temperature
from scene_features import iter_abi_spectral_features, scene_label_sort_order
from spectral_response import CHANNELS, load_goes_filters

SUNNY_VIEW_ANGLES_DEG = np.arange(0, 90, 5)


def sample_abi_file(path, points_per_file, seed, chunk_rows=64, pixel_step=10):
    """Uniformly sample valid strided ABI pixels without accumulating the file.

    Random-priority reservoir: memory is O(points_per_file + chunk_pixels),
    and each valid strided pixel is visited once. Returns features[n,10],
    source flat indices[n], and the total number of valid strided pixels.
    """
    if points_per_file < 1:
        raise ValueError("points_per_file must be positive")
    rng = np.random.default_rng(seed)
    sampled = np.empty((0, 10), dtype=np.float32)
    positions = np.empty(0, dtype=np.int64)
    priorities = np.empty(0, dtype=np.float64)
    seen = 0
    for features, indices in iter_abi_spectral_features(path, chunk_rows, pixel_step):
        seen += features.shape[0]
        keys = rng.random(features.shape[0])
        if keys.size > points_per_file:
            selected = np.argpartition(keys, points_per_file - 1)[:points_per_file]
            keys, features, indices = keys[selected], features[selected], indices[selected]
        priorities = np.concatenate((priorities, keys))
        sampled = np.concatenate((sampled, features), axis=0)
        positions = np.concatenate((positions, indices))
        if priorities.size > points_per_file:
            selected = np.argpartition(priorities, points_per_file - 1)[:points_per_file]
            priorities, sampled, positions = (
                priorities[selected], sampled[selected], positions[selected]
            )
    if not seen:
        raise ValueError(f"No valid spectral pixels in {path}")
    order = np.argsort(positions)
    return sampled[order], positions[order], seen


def fit_spectral_model(features, n_components=7, seed=42, n_jobs=1):
    """Fit ABI-only Scale/PCA/GMM; PCA removes exact BT/BTD dependencies."""
    if features.ndim != 2 or features.shape[1] != 10 or not np.all(np.isfinite(features)):
        raise ValueError("Expected finite training features with shape (n, 10)")
    if n_components < 1 or features.shape[0] < max(n_components, 2):
        raise ValueError("GMM needs at least two rows and one row per requested component")
    if n_jobs < 1:
        raise ValueError("GMM jobs must be positive")
    if np.any(np.std(features[:, :6], axis=0) == 0):
        raise ValueError("ABI training sample must vary in every BT channel")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.98, svd_solver="full")),
        ("gmm", ParallelGaussianMixture(
            n_components=n_components, n_init=5, covariance_type="full",
            random_state=seed, reg_covar=1e-6, max_iter=300, n_jobs=n_jobs,
        )),
    ])
    model.fit(features.astype(np.float64, copy=False))
    if not model.named_steps["gmm"].converged_:
        raise RuntimeError("Spectral GMM did not converge; do not interpret this model")
    means_scaled = model.named_steps["pca"].inverse_transform(
        model.named_steps["gmm"].means_
    )
    order = scene_label_sort_order(means_scaled, method="c14")
    return model, order


def classify_spectral_features(model, scene_order, features):
    """Return ordered scene IDs, posterior probabilities, and PCA-space log density."""
    transformed = model.named_steps["pca"].transform(
        model.named_steps["scaler"].transform(features)
    )
    gmm = model.named_steps["gmm"]
    probabilities = gmm.predict_proba(transformed)[:, scene_order]
    return probabilities.argmax(axis=1), probabilities, gmm.score_samples(transformed)


def summarize_classification(features, labels, probabilities, scores, threshold):
    """Summarize occupancy and mean/SD BT/BTD; no classification accuracy is implied."""
    classes = []
    for scene in range(probabilities.shape[1]):
        selected = labels == scene
        count = int(np.count_nonzero(selected))
        classes.append({
            "scene_id": scene,
            "count": count,
            "fraction": count / len(labels),
            "feature_mean_K": features[selected].mean(axis=0).tolist() if count else None,
            "feature_sd_K": (
                features[selected].std(axis=0, ddof=1).tolist() if count > 1 else None
            ),
        })
    return {
        "n_records": len(labels),
        "mean_max_posterior": float(probabilities.max(axis=1).mean()),
        "fraction_below_abi_log_density_p01": float(np.mean(scores < threshold)),
        "classes": classes,
    }


def abi_response_on_grid(wavelength_um, filter_dir):
    """Return SRFs[n_wavelength,6] and wavenumber normalization[6].

    Sunny L_lambda is W m-2 sr-1 um-1. ABI L_nu is
    mW m-2 sr-1 (cm-1)-1: 1000 * integral(L_lambda R d_lambda) /
    integral(R d_nu), with |d_nu/d_lambda| = 10000/lambda_um**2.
    """
    wavelength = np.asarray(wavelength_um)
    if (
        wavelength.ndim != 1 or wavelength.size < 3
        or not np.all(np.isfinite(wavelength))
        or np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0)
    ):
        raise ValueError("Sunny wavelength grid must be finite, positive and increasing")
    filter_wavelengths, filter_responses = load_goes_filters(Path(filter_dir))
    responses = []
    for channel, lam_nm, response in zip(CHANNELS, filter_wavelengths, filter_responses):
        lam = lam_nm / 1000.0
        if (
            not np.all(np.isfinite(lam)) or not np.all(np.isfinite(response))
            or np.any(np.diff(lam) <= 0) or np.any(response < 0)
            or not np.any(response > 0)
        ):
            raise ValueError(f"Invalid ABI response for {channel}")
        if wavelength[0] > lam[0] or wavelength[-1] < lam[-1]:
            raise ValueError(f"Sunny wavelength grid does not cover the {channel} SRF")
        responses.append(np.interp(wavelength, lam, response, left=0, right=0))
    responses = np.stack(responses, axis=1)
    normalization = simpson(
        responses * (10000.0 / wavelength[:, None] ** 2), x=wavelength, axis=0
    )
    if np.any(normalization <= 0) or not np.all(np.isfinite(normalization)):
        raise ValueError("Sunny wavelength sampling does not resolve every ABI channel")
    return responses, normalization


def convolve_abi_radiances(wavelength_um, directional_radiances, responses, normalization):
    """Return synthetic ABI spectral-density radiances[view,6], not band integrals."""
    radiances = np.asarray(directional_radiances)
    if (
        radiances.ndim != 2 or radiances.shape[0] != len(wavelength_um)
        or not np.all(np.isfinite(radiances)) or np.any(radiances < 0)
    ):
        raise ValueError("Expected non-negative finite Sunny radiances[wavelength,view]")
    result = np.empty((radiances.shape[1], len(CHANNELS)), dtype=np.float64)
    for channel in range(len(CHANNELS)):
        result[:, channel] = 1000.0 * simpson(
            radiances * responses[:, channel, None], x=wavelength_um, axis=0
        ) / normalization[channel]
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("Synthetic ABI channel radiances must be finite and positive")
    return result


def synthetic_brightness_temperature(radiances, planck):
    """Convert ABI-unit radiances[view,6] using reference Planck coefficients[6,4]."""
    coefficients = np.asarray(planck)
    if (
        coefficients.shape != (6, 4) or not np.all(np.isfinite(coefficients))
        or np.any(coefficients[:, :2] <= 0) or np.any(coefficients[:, 3] <= 0)
    ):
        raise ValueError("Expected valid reference Planck coefficients with shape (6, 4)")
    bt = radiance_to_brightness_temperature(radiances, coefficients.T)
    if not np.all(np.isfinite(bt)) or np.any(bt <= 0) or np.any(bt >= 1000):
        raise ValueError("Synthetic BT is outside the ABI feature validity range")
    return bt


def blackbody_roundtrip(wavelength_um, responses, normalization, planck):
    """Recover blackbody temperatures 180..330 K; return maximum absolute error[K]."""
    temperatures = np.arange(180.0, 331.0, 10.0)
    wavelength_m = wavelength_um[:, None] * 1e-6
    spectra = (
        2 * 6.62607015e-34 * 299792458.0**2 / wavelength_m**5
        / np.expm1(
            6.62607015e-34 * 299792458.0
            / (wavelength_m * 1.380649e-23 * temperatures)
        ) * 1e-6
    )
    radiances = convolve_abi_radiances(
        wavelength_um, spectra, responses, normalization
    )
    recovered = synthetic_brightness_temperature(radiances, planck)
    return float(np.max(np.abs(recovered - temperatures[:, None])))


def sunny_scene_files(sunny_dir):
    """Return paired clear/cloud spectrum paths with shared IDs as grouping keys."""
    files = []
    ids_by_regime = []
    for regime, folder in (("clear_sky", "radiance_lw_cs"), ("cloudy", "radiance_lw_cl")):
        paths = sorted((Path(sunny_dir) / folder).glob("sunny_lw_*"))
        if not paths:
            raise FileNotFoundError(f"No Sunny spectra in {Path(sunny_dir) / folder}")
        ids = []
        for path in paths:
            scene_id = path.name.rsplit("_", 1)[-1]
            if not scene_id.isdigit():
                raise ValueError(f"Invalid Sunny scene index in {path}")
            ids.append(scene_id)
            files.append((path, regime, scene_id))
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate Sunny indices in {folder}")
        ids_by_regime.append(set(ids))
    if ids_by_regime[0] != ids_by_regime[1]:
        raise ValueError("Sunny clear/cloud spectra must have matching scene indices")
    return files
