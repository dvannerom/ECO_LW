"""Assess ECO narrowband-flux to broadband-OLR reconstruction on Sunny spectra."""

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml
from scipy.integrate import quad, simpson
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eco_spectral_response import channel_response_matrix, load_channel_scenarios
from adm import elmer, radiance_integrand
from nominal_budget import classify, fit_n2bc, population_weights
from uncertainty import build_provenance, sha256_file


RESPONSE_CATALOG = ROOT / "config" / "eco_channel_scenarios.yaml"
ERROR_BUDGET = ROOT / "config" / "error_budget.yaml"
DEFAULT_SUNNY_DIR = ROOT / "data/inputs/sunny"
DEFAULT_METRICS = ROOT / "data/nominal_production/diagnostics/eco_spectral_reconstruction.json"
DEFAULT_RESIDUALS = ROOT / "data/nominal_production/diagnostics/eco_spectral_residuals.csv"
DEFAULT_N2BC_MODELS = ROOT / "data/nominal_production/models/eco_n2bc_cv_models.joblib"
DEFAULT_N2BC_METRICS = ROOT / "data/nominal_production/diagnostics/eco_n2bc_stage_metrics.json"
DEFAULT_ADM_DIR = ROOT / "data/nominal_production/products/spectral_assessment/narrowband_flux"
DEFAULT_ADM_METRICS = ROOT / "data/nominal_production/diagnostics/eco_adm_stage_metrics.json"
DEFAULT_SCENE_MODEL = ROOT / "data/scene_classification/models/spectral_gmm.joblib"
NOMINAL_DIR = ROOT / "data/nominal_production/uncertainties/nominal_budget"
DEFAULT_SUNNY_ABI = NOMINAL_DIR / "sunny.npz"
TRAINING_MODE = "abi_population_weighted"
REGRESSION_DEGREES = (1, 2, 3)


def evaluation_provenance(args, input_paths):
    return build_provenance(
        root=ROOT,
        code_paths=(
            Path(__file__).resolve(),
            ROOT / "src" / "adm.py",
            ROOT / "src" / "eco_spectral_response.py",
            ROOT / "src" / "nominal_budget.py",
            ROOT / "src" / "abi_sunny_comparison.py",
            ROOT / "src" / "scene_features.py",
        ),
        configuration_paths=(
            ROOT / "config.yaml",
            args.channel_scenarios.resolve(),
            args.error_budget.resolve(),
        ),
        input_paths=input_paths,
    )


def load_sunny_library(sunny_dir, available_view_angles_deg):
    """Read broadband spectra and directional radiance spectra by scene."""
    available_view_angles_deg = np.asarray(available_view_angles_deg, dtype=np.float64)
    scene_files = []
    for regime, directory in (
        ("clear_sky", "radiance_lw_cs"),
        ("cloudy", "radiance_lw_cl"),
    ):
        scene_files.extend(
            (path, regime)
            for path in sorted((sunny_dir / directory).glob("sunny_lw_*"))
            if path.is_file()
        )
    if not scene_files:
        raise FileNotFoundError(f"No Sunny LW spectra found under {sunny_dir}")

    scene_ids = []
    regimes = []
    spectral_data = []
    broadband_flux = np.empty(len(scene_files), dtype=np.float64)
    for index, (path, regime) in enumerate(scene_files):
        scene_id = path.stem.rsplit("_", 1)[-1]
        if not scene_id.isdigit():
            raise ValueError(f"Cannot parse Sunny scene index from {path.name}")
        data = np.loadtxt(path, usecols=tuple(range(2 + available_view_angles_deg.size)))
        if data.ndim != 2 or data.shape[1] != 2 + available_view_angles_deg.size:
            raise ValueError(f"Expected flux and all configured radiance-angle columns in {path}")
        if not np.all(np.isfinite(data)) or np.any(data[:, 1:] < 0):
            raise ValueError(f"Invalid Sunny radiance/flux values in {path}")
        if np.any(np.diff(data[:, 0]) <= 0):
            raise ValueError(f"Wavelength grid must be strictly increasing in {path}")

        spectral_data.append(
            (data[:, 0].copy(), data[:, 1].copy(), data[:, 2:].copy())
        )
        broadband_flux[index] = simpson(data[:, 1], x=data[:, 0])
        scene_ids.append(scene_id)
        regimes.append(regime)

    scene_ids = np.asarray(scene_ids)
    regimes = np.asarray(regimes)
    unique_ids, id_counts = np.unique(scene_ids, return_counts=True)
    if np.any(id_counts != 2):
        raise ValueError("Each Sunny scene index must have one clear and one cloudy spectrum")
    for scene_id in unique_ids:
        if set(regimes[scene_ids == scene_id]) != {"clear_sky", "cloudy"}:
            raise ValueError(f"Sunny scene index {scene_id} lacks a clear/cloud record")

    if np.any(~np.isfinite(broadband_flux)) or np.any(broadband_flux <= 0):
        raise ValueError("Reference broadband OLR must be finite and positive")
    return spectral_data, broadband_flux, scene_ids, regimes, available_view_angles_deg


def simulate_channel_radiances_and_fluxes(
    spectral_data, scenario, default_edge_slope
):
    """Convolve directional radiances and hemispheric flux with ECO channel SRFs."""
    channel_count = len(scenario["channel_names"])
    angle_count = spectral_data[0][2].shape[1]
    channel_radiances = np.empty(
        (len(spectral_data), angle_count, channel_count), dtype=np.float64
    )
    true_channel_fluxes = np.empty(
        (len(spectral_data), channel_count), dtype=np.float64
    )
    for index, (wavelength_um, spectral_flux, directional_radiances) in enumerate(
        spectral_data
    ):
        responses = channel_response_matrix(
            wavelength_um, scenario, default_edge_slope
        )
        weighted_radiances = directional_radiances[:, :, None] * responses[:, None, :]
        channel_radiances[index] = simpson(
            weighted_radiances, x=wavelength_um, axis=0
        )
        true_channel_fluxes[index] = simpson(
            spectral_flux[:, None] * responses, x=wavelength_um, axis=0
        )
    if (
        np.any(~np.isfinite(channel_radiances))
        or np.any(channel_radiances <= 0)
        or np.any(~np.isfinite(true_channel_fluxes))
        or np.any(true_channel_fluxes <= 0)
    ):
        raise ValueError("ECO channel radiances/fluxes must be finite and positive")
    return channel_radiances, true_channel_fluxes


def integrate_true_channel_fluxes(spectral_data, scenario, default_edge_slope):
    """Integrate ECO channel fluxes from Sunny hemispheric flux spectra only."""
    channel_count = len(scenario["channel_names"])
    channel_fluxes = np.empty((len(spectral_data), channel_count), dtype=np.float64)
    for index, (wavelength_um, spectral_flux, _) in enumerate(spectral_data):
        responses = channel_response_matrix(
            wavelength_um, scenario, default_edge_slope
        )
        channel_fluxes[index] = simpson(
            spectral_flux[:, None] * responses, x=wavelength_um, axis=0
        )
    if np.any(~np.isfinite(channel_fluxes)) or np.any(channel_fluxes <= 0):
        raise ValueError("True ECO channel fluxes must be finite and positive")
    return channel_fluxes


def retrieve_narrowband_fluxes(channel_radiances, viewing_angles_deg):
    """Fit the repo limb-darkening shape per scene/channel and retrieve flux."""
    viewing_angles = np.asarray(viewing_angles_deg, dtype=np.float64)
    if viewing_angles.ndim != 1 or viewing_angles.size < 2:
        raise ValueError("at least two configured viewing angles are required to fit an ADM")
    reference_matches = np.flatnonzero(np.isclose(viewing_angles, 55.0))
    if reference_matches.size != 1:
        raise ValueError("the ADM fit angles must include exactly one 55-degree view")
    reference_index = int(reference_matches[0])

    angular_basis = elmer(viewing_angles) - 1.0
    denominator = float(np.dot(angular_basis, angular_basis))
    if denominator <= 0:
        raise ValueError("viewing angles do not constrain the ADM shape parameter")

    reference_radiance = channel_radiances[:, reference_index, :]
    normalized_radiance = channel_radiances / reference_radiance[:, None, :]
    shape_parameters = np.einsum(
        "a,sac->sc", angular_basis, normalized_radiance - 1.0
    ) / denominator
    fitted_shape = 1.0 + shape_parameters[:, None, :] * angular_basis[None, :, None]
    if np.any(fitted_shape <= 0):
        raise ValueError("fitted ADM has non-positive radiance ratios")

    reference_integral = quad(radiance_integrand, 0.0, 90.0, args=(0.0,))[0]
    shape_integral = (
        quad(radiance_integrand, 0.0, 90.0, args=(1.0,))[0]
        - reference_integral
    )
    normalization = 1.0 / (
        reference_integral + shape_parameters * shape_integral
    )
    corrected_radiance = channel_radiances / (
        normalization[:, None, :] * fitted_shape
    )
    narrowband_flux = np.pi * np.mean(corrected_radiance, axis=1)
    fit_residual_percent = (
        (normalized_radiance - fitted_shape) / normalized_radiance * 100.0
    )
    adm_fit_rms_percent = np.sqrt(np.mean(fit_residual_percent**2, axis=1))
    if np.any(~np.isfinite(narrowband_flux)) or np.any(narrowband_flux <= 0):
        raise ValueError("ADM-retrieved ECO narrowband fluxes must be finite and positive")
    return narrowband_flux, shape_parameters, adm_fit_rms_percent


def summarize_adm_retrieval(
    estimated_flux,
    true_flux,
    shape_parameters,
    fit_rms_percent,
    regimes,
    weights,
    channel_names,
):
    """Summarize ABI-population-weighted ADM diagnostics by channel and regime."""
    weights = np.asarray(weights, dtype=np.float64)
    if (
        weights.shape != regimes.shape
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0)
        or not weights.sum() > 0
    ):
        raise ValueError("ADM diagnostic weights must match Sunny scenes and be non-negative")

    def weighted_values(values, subset_weights, include_sd=False, include_p95=False):
        if not subset_weights.sum() > 0:
            raise ValueError("ADM diagnostic subset has no ABI population weight")
        values = np.asarray(values, dtype=np.float64)
        normalized = subset_weights / subset_weights.sum()
        mean = float(normalized @ values)
        squared_weight_sum = float(normalized @ normalized)
        summary = {
            "mean": mean,
            "effective_sample_size": 1.0 / squared_weight_sum,
        }
        if include_sd:
            variance = (
                float(normalized @ np.square(values - mean))
                / (1.0 - squared_weight_sum)
                if squared_weight_sum < 1.0
                else np.nan
            )
            summary["standard_deviation"] = float(np.sqrt(variance))
        if include_p95:
            order = np.argsort(values)
            sorted_values = values[order]
            sorted_weights = normalized[order]
            midpoint_cdf = np.cumsum(sorted_weights) - 0.5 * sorted_weights
            summary["p95"] = float(
                np.interp(
                    0.95,
                    midpoint_cdf,
                    sorted_values,
                    left=sorted_values[0],
                    right=sorted_values[-1],
                )
            )
        return summary

    channel_results = {}
    scene_weights = weights / weights.sum()
    for channel_index, channel_name in enumerate(channel_names):
        relative_error_percent = (
            estimated_flux[:, channel_index] - true_flux[:, channel_index]
        ) / true_flux[:, channel_index] * 100.0
        scene_weights = weights / weights.sum()
        channel_results[channel_name] = {
            "shape_parameter": weighted_values(
                shape_parameters[:, channel_index], scene_weights, include_sd=True
            ),
            "fit_residual_rms_percent": weighted_values(
                fit_rms_percent[:, channel_index],
                scene_weights,
                include_p95=True,
            ),
            "narrowband_flux_error_percent": {
                "overall": summarize_errors(relative_error_percent, scene_weights),
                "by_regime": {
                    regime: summarize_errors(
                        relative_error_percent[regimes == regime],
                        scene_weights[regimes == regime],
                    )
                    for regime in np.unique(regimes)
                },
            },
        }
    return channel_results


def summarize_errors(error_percent, weights=None):
    """Summarize relative OLR errors [%] for error_percent[n].

    weights[n] >= 0 are ABI-population importance weights (renormalized here);
    None means equal weights. The SD uses the unbiased reliability-weight
    estimator, which reduces to ddof=1 for equal weights.
    """
    error_percent = np.asarray(error_percent, dtype=np.float64)
    weights = (
        np.ones_like(error_percent)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if (
        weights.shape != error_percent.shape
        or np.any(weights < 0)
        or not weights.sum() > 0
    ):
        raise ValueError("Error weights must be non-negative with positive sum")
    weights = weights / weights.sum()
    squared_weight_sum = float(weights @ weights)
    bias = float(weights @ error_percent)
    variance = (
        float(weights @ np.square(error_percent - bias)) / (1.0 - squared_weight_sum)
        if squared_weight_sum < 1.0
        else np.nan
    )
    standard_deviation = float(np.sqrt(variance))
    return {
        "n_spectra": int(error_percent.size),
        "effective_sample_size": 1.0 / squared_weight_sum,
        "bias_percent": bias,
        "standard_deviation_percent": standard_deviation,
        "k2_scatter_percent": 2.0 * standard_deviation,
        "rmse_percent": float(np.sqrt(weights @ np.square(error_percent))),
        "max_absolute_error_percent": float(
            np.max(np.abs(error_percent[weights > 0]))
        ),
    }


def weighted_breakdown(error_percent, weights, regimes, abi_labels, population):
    """Population-weighted overall, per-Sunny-regime and per-ABI-scene summaries.

    Regime and scene subsets renormalize the same weights; population_fraction
    is the subset share of the weighted (covered) population.
    """
    weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)

    def subset(mask):
        fraction = float(weights[mask].sum())
        if fraction <= 0:
            return {"population_fraction": 0.0, "n_spectra": int(mask.sum())}
        return {
            "population_fraction": fraction,
            **summarize_errors(error_percent[mask], weights[mask]),
        }

    return {
        "overall": summarize_errors(error_percent, weights),
        "by_regime": {
            str(regime): subset(regimes == regime) for regime in np.unique(regimes)
        },
        "by_abi_scene": {
            str(scene): {
                "abi_population_mass": float(population[scene]),
                **subset(abi_labels == scene),
            }
            for scene in np.unique(abi_labels)
        },
    }


def support_summary(support):
    return {
        key: support[key]
        for key in ("covered_population", "missing_scenes", "effective_sample_size")
    }


def find_requirement(error_budget, requirement_id):
    for requirement in error_budget.get("requirements", {}).values():
        if requirement.get("rfmA_id") == requirement_id:
            return requirement
    raise KeyError(f"Requirement {requirement_id!r} not found in error budget")


def requirement_summary(error_budget, requirement_id):
    requirement = find_requirement(error_budget, requirement_id)
    return {
        "id": requirement_id,
        "source": requirement["source"],
        "quantity": requirement.get("quantity"),
        "aggregation": requirement.get("aggregation"),
        "units": requirement.get("units"),
        "wavelength_range": requirement.get("wavelength_range"),
        "goal": requirement.get("goal"),
        "threshold": requirement.get("threshold"),
        "maximum": requirement.get("maximum"),
        "confidence_level": requirement.get("confidence_level"),
    }


def predict_n2bc(model, narrowband_flux):
    """Apply a fitted (scaler, polynomial, regression) N2BC to flux[file, channel]."""
    scaler, polynomial, regression = model
    return regression.predict(polynomial.transform(scaler.transform(narrowband_flux)))


def load_abi_scene_labels(scene_model_path, sunny_abi_path, scene_ids, regimes):
    """Classify each Sunny file with the nominal ABI-trained GMM.

    Uses the SRF-independent ABI-convolved Sunny radiances abi_radiance[file,
    view, 6] prepared by the nominal budget; each file takes the majority-vote
    scene over its views. Returns labels[file], vote_fraction[file] and the ABI
    scene population[K] (fractions summing to one).
    """
    fitted = joblib.load(scene_model_path)
    with np.load(sunny_abi_path) as source:
        abi_radiance = source["abi_radiance"]
        planck = source["planck"]
        abi_scene_ids = source["groups"]
        abi_regimes = source["regimes"]
    if not (
        np.array_equal(abi_scene_ids, scene_ids)
        and np.array_equal(abi_regimes, regimes)
    ):
        raise ValueError(
            "Sunny file ordering differs between the nominal ABI library and this assessment"
        )
    labels, vote_fraction = classify(fitted["model"], abi_radiance, planck)
    return labels, vote_fraction, np.asarray(fitted["population"], dtype=np.float64)


def evaluate_scenario(
    true_channel_fluxes, broadband_flux, scene_ids, regimes, abi_labels, population, folds
):
    """Grouped-CV global N2BC fits weighted by ABI scene population.

    Each training fold renormalizes population weights over its own supported
    scenes, as in the nominal budget; known clear/cloud regimes are never used
    for fitting or prediction.
    """
    fold_splits = list(
        GroupKFold(n_splits=folds).split(true_channel_fluxes, groups=scene_ids)
    )
    degree_models = {}
    for degree in REGRESSION_DEGREES:
        folds_for_degree = []
        for fold_index, (train_indices, test_indices) in enumerate(fold_splits, start=1):
            if set(scene_ids[train_indices]) & set(scene_ids[test_indices]):
                raise RuntimeError("scene-index leakage across validation folds")
            train_weights, train_support = population_weights(
                abi_labels[train_indices], population
            )
            model = fit_n2bc(
                true_channel_fluxes[train_indices],
                broadband_flux[train_indices],
                train_weights,
                degree,
            )
            folds_for_degree.append(
                {
                    "fold": fold_index,
                    "test_indices": test_indices.copy(),
                    "model": model,
                    "training_support": support_summary(train_support),
                }
            )
        degree_models[f"degree_{degree}"] = folds_for_degree
    return {TRAINING_MODE: degree_models}


def apply_fold_models(
    features,
    broadband_flux,
    scene_ids,
    regimes,
    abi_labels,
    weights,
    population,
    fold_models,
    assessment_stage,
):
    """Apply saved grouped-CV N2BC models to one narrowband-flux input set.

    features[file, channel] in W m-2; weights[file] are the full-sample ABI
    population weights used for all reported statistics.
    """
    results = {}
    residual_rows = []
    weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    for training_mode, degree_models in fold_models.items():
        mode_results = {}
        for degree_name, folds in degree_models.items():
            predicted_flux = np.full(broadband_flux.shape, np.nan, dtype=np.float64)
            fold_number = np.full(broadband_flux.shape, -1, dtype=np.int16)
            fold_metrics = []
            for fold in folds:
                test_indices = fold["test_indices"]
                predicted_flux[test_indices] = predict_n2bc(
                    fold["model"], features[test_indices]
                )
                fold_number[test_indices] = fold["fold"]
                fold_error = (
                    predicted_flux[test_indices] - broadband_flux[test_indices]
                ) / broadband_flux[test_indices] * 100.0
                fold_metrics.append(
                    {
                        "fold": int(fold["fold"]),
                        **summarize_errors(fold_error, weights[test_indices]),
                        "training_support": fold["training_support"],
                    }
                )

            if (
                np.any(~np.isfinite(predicted_flux))
                or np.any(predicted_flux <= 0)
                or np.any(fold_number < 1)
            ):
                raise RuntimeError(f"{assessment_stage} CV did not predict every record")
            error_percent = (predicted_flux - broadband_flux) / broadband_flux * 100.0
            mode_results[degree_name] = {
                **weighted_breakdown(
                    error_percent, weights, regimes, abi_labels, population
                ),
                "fold_metrics": fold_metrics,
            }
            degree = degree_name.removeprefix("degree_")
            for index in range(broadband_flux.size):
                residual_rows.append(
                    {
                        "assessment_stage": assessment_stage,
                        "training_mode": training_mode,
                        "regression_degree": degree,
                        "scene_index": scene_ids[index],
                        "regime": regimes[index],
                        "abi_scene": int(abi_labels[index]),
                        "population_weight": float(weights[index]),
                        "fold": int(fold_number[index]),
                        "reference_olr_w_m2": float(broadband_flux[index]),
                        "predicted_olr_w_m2": float(predicted_flux[index]),
                        "relative_error_percent": float(error_percent[index]),
                    }
                )
        results[training_mode] = mode_results
    return results, residual_rows


def scenario_metadata(scenario_id, scenario, channel_catalog):
    return {
        "source": scenario["source"],
        "description": scenario["description"],
        "channel_names": scenario["channel_names"],
        "channel_bands_um": scenario["channel_bands_um"],
        "channel_count": len(scenario["channel_names"]),
        "response_model": scenario["response_model"],
        "edge_slope_per_um": scenario.get(
            "edge_slope_per_um", channel_catalog["default_edge_slope_per_um"]
        ),
        "idealized_response_note": (
            "RfMA band limits do not specify flight SRFs; logistic band edges are scenario assumptions."
        ),
    }


def assessment_metadata(args, channel_catalog, broadband_flux, scene_ids, spectral_data):
    error_budget = yaml.safe_load(args.error_budget.read_text())
    requirement = find_requirement(error_budget, "ObsReq_16")
    retrieval_geometry = channel_catalog["retrieval_geometry"]
    unique_scene_ids = np.unique(scene_ids)
    return {
        "target": "ECO",
        "requirement": {
            "id": "ObsReq_16",
            "source": requirement["source"],
            "maximum_percent": requirement["maximum"],
            "confidence_level": requirement.get("confidence_level"),
        },
        "channel_scenario_catalog": str(args.channel_scenarios),
        "reference_data": {
            "library": "GERB/Clerbaux Sunny LW spectra",
            "path": str(args.sunny_dir),
            "spectrum_count": int(broadband_flux.size),
            "shared_scene_index_count": int(unique_scene_ids.size),
            "records_per_index": "one clear-sky and one cloudy spectrum; grouped together in validation",
            "spectral_flux_column": "upward hemispheric TOA spectral flux (W m-2 um-1)",
            "directional_radiance_columns": "TOA spectral radiance at configured angles (W m-2 um-1 sr-1)",
            "available_view_angles_deg": channel_catalog["retrieval_geometry"][
                "available_view_angles_deg"
            ],
            "wavelength_range_um": [
                float(min(wavelength[0] for wavelength, _, _ in spectral_data)),
                float(max(wavelength[-1] for wavelength, _, _ in spectral_data)),
            ],
            "wavelength_grid_lengths": {
                str(length): int(
                    sum(wavelength.size == length for wavelength, _, _ in spectral_data)
                )
                for length in sorted(
                    {wavelength.size for wavelength, _, _ in spectral_data}
                )
            },
            "integration": "Simpson integration over the supplied spectrum, including its longwave extension to 500 um.",
            "angular_retrieval": retrieval_geometry["retrieval_note"],
            "adm_method": "Per-scene, per-channel least-squares fit of the repository radiance_linear(theta,b) profile using the 55-degree radiance as amplitude reference; retrieved fluxes use the repository radiance_integrand normalization.",
            "limitation": "Shared clear/cloud indices are conservatively grouped but not assumed to be identical atmospheric profiles; the clear/cloudy truth is reported as a diagnostic only.",
        },
        "validation": {
            "method": "GroupKFold out-of-fold predictions of one global, ABI-population-weighted N2BC",
            "group_key": "four-digit shared clear/cloud scene index",
            "folds": args.folds,
            "k2_convention": "2 times sample standard deviation; provisional comparison convention, not confirmed identical to the RfMA fitting-noise definition.",
        },
        "provenance": evaluation_provenance(
            args,
            {
                "sunny_library": args.sunny_dir.resolve(),
                "abi_scene_model": args.scene_model.resolve(),
                "sunny_abi_radiances": args.sunny_abi.resolve(),
            },
        ),
    }


def scene_weighting_metadata(args, regimes, vote_fraction, population, weights, support):
    return {
        "method": (
            "Each Sunny file is assigned the majority-vote ABI-GMM scene over its views; "
            "weight = ABI population mass of that scene / Sunny files in that scene, "
            "renormalized over supported scenes. Fits use training-fold weights; "
            "all statistics use full-sample weights."
        ),
        "training_mode": TRAINING_MODE,
        "scene_model": str(args.scene_model),
        "sunny_abi_radiances": str(args.sunny_abi),
        "abi_population": population.tolist(),
        "sunny_scene_counts": support["scene_counts"],
        "covered_population": support["covered_population"],
        "missing_scenes": support["missing_scenes"],
        "effective_sample_size": support["effective_sample_size"],
        "view_vote_fraction": {
            "mean": float(np.mean(vote_fraction)),
            "minimum": float(np.min(vote_fraction)),
        },
        "clear_sky_population_fraction": float(weights[regimes == "clear_sky"].sum()),
    }


def fit_n2bc_stage(args, channel_catalog, scenarios):
    """Fit N2BC fold models using true integrated ECO band fluxes only."""
    spectral_data, broadband_flux, scene_ids, regimes, _ = load_sunny_library(
        args.sunny_dir,
        channel_catalog["retrieval_geometry"]["available_view_angles_deg"],
    )
    if args.folds > np.unique(scene_ids).size:
        raise ValueError("--folds exceeds the number of grouped Sunny indices")
    result_metadata = assessment_metadata(
        args, channel_catalog, broadband_flux, scene_ids, spectral_data
    )
    abi_labels, vote_fraction, population = load_abi_scene_labels(
        args.scene_model, args.sunny_abi, scene_ids, regimes
    )
    weights, support = population_weights(abi_labels, population)
    scene_weighting = scene_weighting_metadata(
        args, regimes, vote_fraction, population, weights, support
    )
    result_metadata["scene_weighting"] = scene_weighting
    model_bundle = {
        "schema_version": 2,
        "folds": args.folds,
        "requirement_id": "ObsReq_16",
        "training_mode": TRAINING_MODE,
        "abi_labels": abi_labels,
        "population_weights": weights,
        "abi_population": population,
        "scene_weighting": scene_weighting,
        "scenarios": {},
    }
    metrics = {**result_metadata, "stage": "n2bc_only_true_band_flux", "scenarios": {}}

    for scenario_id, scenario in scenarios.items():
        print(f"Fitting independent N2BC models for {scenario_id}", flush=True)
        true_narrowband_fluxes = integrate_true_channel_fluxes(
            spectral_data, scenario, channel_catalog["default_edge_slope_per_um"]
        )
        fold_models = evaluate_scenario(
            true_narrowband_fluxes,
            broadband_flux,
            scene_ids,
            regimes,
            abi_labels,
            population,
            args.folds,
        )
        regressions, _ = apply_fold_models(
            true_narrowband_fluxes,
            broadband_flux,
            scene_ids,
            regimes,
            abi_labels,
            weights,
            population,
            fold_models,
            "n2bc_only",
        )
        metadata = scenario_metadata(scenario_id, scenario, channel_catalog)
        metrics["scenarios"][scenario_id] = {
            **metadata,
            "regressions": regressions,
        }
        model_bundle["scenarios"][scenario_id] = {
            **metadata,
            "broadband_flux": broadband_flux,
            "true_narrowband_fluxes": true_narrowband_fluxes,
            "scene_ids": scene_ids,
            "regimes": regimes,
            "fold_models": fold_models,
        }

    args.n2bc_models_output.parent.mkdir(parents=True, exist_ok=True)
    args.n2bc_metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, args.n2bc_models_output, compress=3)
    with args.n2bc_metrics_output.open("w") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Wrote independent ECO N2BC models: {args.n2bc_models_output}", flush=True)
    print(f"Wrote N2BC-only metrics: {args.n2bc_metrics_output}", flush=True)


def retrieve_adm_stage(args, channel_catalog, scenarios):
    """Fit ECO ADM shape per simulated scene/channel from angular radiances."""
    spectral_data, broadband_flux, scene_ids, regimes, available_view_angles = (
        load_sunny_library(
            args.sunny_dir,
            channel_catalog["retrieval_geometry"]["available_view_angles_deg"],
        )
    )
    abi_labels, vote_fraction, population = load_abi_scene_labels(
        args.scene_model, args.sunny_abi, scene_ids, regimes
    )
    weights, support = population_weights(abi_labels, population)
    scene_weighting = scene_weighting_metadata(
        args, regimes, vote_fraction, population, weights, support
    )
    retrieval_geometry = channel_catalog["retrieval_geometry"]
    retrieval_view_angles = np.asarray(
        retrieval_geometry["retrieval_view_angles_deg"], dtype=np.float64
    )
    retrieval_indices = []
    for angle in retrieval_view_angles:
        matches = np.flatnonzero(np.isclose(available_view_angles, angle))
        if matches.size != 1:
            raise ValueError(f"Configured retrieval view {angle:g} deg is unavailable")
        retrieval_indices.append(int(matches[0]))
    retrieval_indices = np.asarray(retrieval_indices, dtype=np.intp)
    stage_metrics = {
        "stage": "adm_retrieval_to_narrowband_flux",
        "target": "ECO",
        "geometry": retrieval_geometry,
        "scene_index_count": int(np.unique(scene_ids).size),
        "scene_weighting": scene_weighting,
        "scenarios": {},
        "provenance": evaluation_provenance(
            args,
            {
                "sunny_library": args.sunny_dir.resolve(),
                "abi_scene_model": args.scene_model.resolve(),
                "sunny_abi_radiances": args.sunny_abi.resolve(),
            },
        ),
    }
    args.adm_output_dir.mkdir(parents=True, exist_ok=True)

    for scenario_id, scenario in scenarios.items():
        print(f"Retrieving ADM-corrected narrowband flux for {scenario_id}", flush=True)
        channel_radiances, true_narrowband_fluxes = simulate_channel_radiances_and_fluxes(
            spectral_data, scenario, channel_catalog["default_edge_slope_per_um"]
        )
        retrieved_fluxes, shape_parameters, fit_rms_percent = retrieve_narrowband_fluxes(
            channel_radiances[:, retrieval_indices, :], retrieval_view_angles
        )
        adm_metrics = summarize_adm_retrieval(
            retrieved_fluxes,
            true_narrowband_fluxes,
            shape_parameters,
            fit_rms_percent,
            regimes,
            weights,
            scenario["channel_names"],
        )
        stage_metrics["scenarios"][scenario_id] = adm_metrics
        np.savez_compressed(
            args.adm_output_dir / f"{scenario_id}.npz",
            scene_ids=scene_ids,
            regimes=regimes,
            retrieved_narrowband_fluxes=retrieved_fluxes,
            shape_parameters=shape_parameters,
            adm_fit_rms_percent=fit_rms_percent,
        )

    args.adm_metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.adm_metrics_output.open("w") as handle:
        json.dump(stage_metrics, handle, indent=2)
    print(f"Wrote ADM-stage metrics: {args.adm_metrics_output}", flush=True)


def combine_chain_stages(
    args, channel_catalog, n2bc_metrics, adm_metrics, abi_adm_evidence
):
    """Join independent N2BC/ADM outputs only for held-out end-to-end scoring."""
    model_bundle = joblib.load(args.n2bc_models_input)
    combined = {
        "assessment": "ECO simulated angular-radiance to ADM narrowband-flux to broadband-OLR chain",
        **{key: value for key, value in n2bc_metrics.items() if key != "scenarios"},
        "branch_provenance": {
            "n2bc": n2bc_metrics.get("provenance"),
            "adm": adm_metrics.get("provenance"),
            "abi_proxy": abi_adm_evidence.get("provenance"),
        },
        "provenance": evaluation_provenance(
            args,
            {
                "n2bc_models": args.n2bc_models_input.resolve(),
                "n2bc_metrics": args.n2bc_metrics_input.resolve(),
                "adm_metrics": args.adm_metrics_input.resolve(),
                "adm_retrieval_products": args.adm_output_dir.resolve(),
                "abi_adm_proxy_summary": args.abi_adm_evidence.resolve(),
            },
        ),
        "dependency_structure": {
            "parallel_branches": {
                "n2bc_coefficients": "Fit on true spectrally integrated channel fluxes and broadband OLR.",
                "adm_retrieval": "Fit the ADM shape from directional channel radiances, then retrieve narrowband flux.",
            },
            "join": "Apply each held-out N2BC model to ADM-retrieved narrowband flux to measure end-to-end error.",
        },
        "additional_proxy_evidence": {
            "abi_adm_variability": abi_adm_evidence,
            "combination_policy": "ABI proxy variability is retained as separate evidence and is not added to Sunny-derived ECO uncertainty without channel/scene/geometry transfer validation.",
        },
        "scenarios": {},
    }
    error_budget = yaml.safe_load(args.error_budget.read_text())
    combined.pop("requirement", None)
    combined["requirement_mappings"] = {
        "n2bc_only_true_band_flux": requirement_summary(
            error_budget, "ObsReq_16"
        ),
        "end_to_end_adm_plus_n2bc": requirement_summary(
            error_budget, "ObsReq_15"
        ),
        "spectral_reference_definition": requirement_summary(
            error_budget, "ObsReq_8"
        ),
    }
    for scenario_id, model_data in model_bundle["scenarios"].items():
        adm_path = args.adm_output_dir / f"{scenario_id}.npz"
        with np.load(adm_path) as adm_data:
            scene_ids = adm_data["scene_ids"]
            regimes = adm_data["regimes"]
            retrieved_fluxes = adm_data["retrieved_narrowband_fluxes"]
        if not np.array_equal(scene_ids, model_data["scene_ids"]):
            raise ValueError(f"Sunny scene ordering differs between branches for {scenario_id}")
        if not np.array_equal(regimes, model_data["regimes"]):
            raise ValueError(f"Sunny regime ordering differs between branches for {scenario_id}")

        weighting = (
            model_bundle["abi_labels"],
            model_bundle["population_weights"],
            model_bundle["abi_population"],
        )
        n2bc_only_results, _ = apply_fold_models(
            model_data["true_narrowband_fluxes"],
            model_data["broadband_flux"],
            scene_ids,
            regimes,
            *weighting,
            model_data["fold_models"],
            "n2bc_only",
        )
        end_to_end_results, _ = apply_fold_models(
            retrieved_fluxes,
            model_data["broadband_flux"],
            scene_ids,
            regimes,
            *weighting,
            model_data["fold_models"],
            "end_to_end",
        )
        final_regressions = {}
        for mode, degree_results in n2bc_only_results.items():
            final_regressions[mode] = {}
            for degree, n2bc_result in degree_results.items():
                final_regressions[mode][degree] = {
                    "n2bc_only": n2bc_result,
                    "end_to_end": end_to_end_results[mode][degree],
                }
        combined["scenarios"][scenario_id] = {
            **{
                key: value
                for key, value in model_data.items()
                if key not in {
                    "broadband_flux",
                    "true_narrowband_fluxes",
                    "scene_ids",
                    "regimes",
                    "fold_models",
                }
            },
            "angular_retrieval": adm_metrics["scenarios"][scenario_id],
            "regressions": final_regressions,
        }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.residuals_output.parent.mkdir(parents=True, exist_ok=True)
    with args.residuals_output.open("w", newline="") as residual_file:
        fieldnames = [
            "scenario_id",
            "assessment_stage",
            "training_mode",
            "regression_degree",
            "scene_index",
            "regime",
            "abi_scene",
            "population_weight",
            "fold",
            "reference_olr_w_m2",
            "predicted_olr_w_m2",
            "relative_error_percent",
        ]
        writer = csv.DictWriter(residual_file, fieldnames=fieldnames)
        writer.writeheader()
        for scenario_id, scenario_residuals in model_bundle["scenarios"].items():
            model_data = scenario_residuals
            with np.load(args.adm_output_dir / f"{scenario_id}.npz") as adm_data:
                retrieved_fluxes = adm_data["retrieved_narrowband_fluxes"]
            for feature_values, assessment_stage in (
                (model_data["true_narrowband_fluxes"], "n2bc_only"),
                (retrieved_fluxes, "end_to_end"),
            ):
                _, rows = apply_fold_models(
                    feature_values,
                    model_data["broadband_flux"],
                    model_data["scene_ids"],
                    model_data["regimes"],
                    model_bundle["abi_labels"],
                    model_bundle["population_weights"],
                    model_bundle["abi_population"],
                    model_data["fold_models"],
                    assessment_stage,
                )
                for row in rows:
                    writer.writerow({"scenario_id": scenario_id, **row})

    residual_path = args.residuals_output.resolve()
    try:
        residual_path_display = residual_path.relative_to(ROOT).as_posix()
    except ValueError:
        residual_path_display = str(residual_path)
    combined["derived_residuals"] = {
        "path": residual_path_display,
        "sha256": sha256_file(residual_path),
    }
    with args.metrics_output.open("w") as handle:
        json.dump(combined, handle, indent=2)
    print(f"Wrote joined ECO chain metrics: {args.metrics_output}", flush=True)
    print(f"Wrote stage-separated OLR residuals: {args.residuals_output}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate independent ECO ADM and N2BC stages on Sunny spectra."
    )
    parser.add_argument(
        "--stage",
        choices=("fit-n2bc", "retrieve-adm", "combine"),
        required=True,
    )
    parser.add_argument("--sunny-dir", type=Path, default=DEFAULT_SUNNY_DIR)
    parser.add_argument("--channel-scenarios", type=Path, default=RESPONSE_CATALOG)
    parser.add_argument("--error-budget", type=Path, default=ERROR_BUDGET)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--residuals-output", type=Path, default=DEFAULT_RESIDUALS)
    parser.add_argument("--n2bc-models-output", type=Path, default=DEFAULT_N2BC_MODELS)
    parser.add_argument("--n2bc-metrics-output", type=Path, default=DEFAULT_N2BC_METRICS)
    parser.add_argument("--n2bc-models-input", type=Path, default=DEFAULT_N2BC_MODELS)
    parser.add_argument("--n2bc-metrics-input", type=Path, default=DEFAULT_N2BC_METRICS)
    parser.add_argument("--scene-model", type=Path, default=DEFAULT_SCENE_MODEL)
    parser.add_argument("--sunny-abi", type=Path, default=DEFAULT_SUNNY_ABI)
    parser.add_argument("--adm-output-dir", type=Path, default=DEFAULT_ADM_DIR)
    parser.add_argument("--adm-metrics-output", type=Path, default=DEFAULT_ADM_METRICS)
    parser.add_argument("--adm-metrics-input", type=Path, default=DEFAULT_ADM_METRICS)
    parser.add_argument(
        "--abi-adm-evidence",
        type=Path,
        default=ROOT / "data/abi_proxy/diagnostics/adm_proxy_summary.json",
    )
    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be at least 2")
    with args.channel_scenarios.open() as handle:
        channel_catalog = yaml.safe_load(handle)
    scenarios = load_channel_scenarios(args.channel_scenarios)

    if args.stage == "fit-n2bc":
        fit_n2bc_stage(args, channel_catalog, scenarios)
    elif args.stage == "retrieve-adm":
        retrieve_adm_stage(args, channel_catalog, scenarios)
    else:
        with args.n2bc_metrics_input.open() as handle:
            n2bc_metrics = json.load(handle)
        with args.adm_metrics_input.open() as handle:
            adm_metrics = json.load(handle)
        with args.abi_adm_evidence.open() as handle:
            abi_adm_evidence = json.load(handle)
        combine_chain_stages(
            args, channel_catalog, n2bc_metrics, adm_metrics, abi_adm_evidence
        )


if __name__ == "__main__":
    main()
