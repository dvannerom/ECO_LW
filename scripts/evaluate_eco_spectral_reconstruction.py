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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eco_spectral_response import channel_response_matrix, load_channel_scenarios
from adm import elmer, radiance_integrand
from uncertainty import build_provenance, sha256_file


RESPONSE_CATALOG = ROOT / "config" / "eco_channel_scenarios.yaml"
ERROR_BUDGET = ROOT / "config" / "error_budget.yaml"
DEFAULT_SUNNY_DIR = ROOT / "data" / "Sunny"
DEFAULT_METRICS = ROOT / "data" / "uncertainty" / "eco_spectral_reconstruction.json"
DEFAULT_RESIDUALS = ROOT / "data" / "uncertainty" / "eco_spectral_residuals.csv"
DEFAULT_N2BC_MODELS = ROOT / "data" / "uncertainty" / "eco_n2bc_cv_models.joblib"
DEFAULT_N2BC_METRICS = ROOT / "data" / "uncertainty" / "eco_n2bc_stage_metrics.json"
DEFAULT_ADM_DIR = ROOT / "data" / "uncertainty" / "eco_adm_retrieval"
DEFAULT_ADM_METRICS = ROOT / "data" / "uncertainty" / "eco_adm_stage_metrics.json"


def evaluation_provenance(args, input_paths):
    return build_provenance(
        root=ROOT,
        code_paths=(
            Path(__file__).resolve(),
            ROOT / "src" / "adm.py",
            ROOT / "src" / "eco_spectral_response.py",
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
    estimated_flux, true_flux, shape_parameters, fit_rms_percent, regimes, channel_names
):
    """Summarize ADM-fit and narrowband-flux errors by ECO channel and regime."""
    channel_results = {}
    for channel_index, channel_name in enumerate(channel_names):
        relative_error_percent = (
            estimated_flux[:, channel_index] - true_flux[:, channel_index]
        ) / true_flux[:, channel_index] * 100.0
        channel_results[channel_name] = {
            "shape_parameter": {
                "mean": float(np.mean(shape_parameters[:, channel_index])),
                "standard_deviation": float(np.std(shape_parameters[:, channel_index], ddof=1)),
            },
            "fit_residual_rms_percent": {
                "mean": float(np.mean(fit_rms_percent[:, channel_index])),
                "p95": float(np.percentile(fit_rms_percent[:, channel_index], 95)),
            },
            "narrowband_flux_error_percent": {
                "overall": summarize_errors(relative_error_percent),
                "by_regime": {
                    regime: summarize_errors(relative_error_percent[regimes == regime])
                    for regime in np.unique(regimes)
                },
            },
        }
    return channel_results


def summarize_errors(error_percent):
    """Summarize relative reconstruction errors, retaining bias and scatter."""
    standard_deviation = float(np.std(error_percent, ddof=1))
    return {
        "n_spectra": int(error_percent.size),
        "bias_percent": float(np.mean(error_percent)),
        "standard_deviation_percent": standard_deviation,
        "k2_scatter_percent": 2.0 * standard_deviation,
        "rmse_percent": float(np.sqrt(np.mean(error_percent**2))),
        "max_absolute_error_percent": float(np.max(np.abs(error_percent))),
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


def evaluate_scenario(
    true_channel_fluxes,
    broadband_flux,
    scene_ids,
    regimes,
    folds,
    requirement_id,
    adm_retrieved_fluxes=None,
):
    splitter = GroupKFold(n_splits=folds)
    model_results = {}
    fold_models = {}
    residual_rows = []
    fold_splits = list(
        splitter.split(true_channel_fluxes, broadband_flux, groups=scene_ids)
    )
    for training_mode in ("unstratified", "clear_cloud_stratified"):
        training_mode_results = {}
        training_mode_models = {}
        for degree in (1, 2, 3):
            predicted_n2bc_flux = np.full(
                broadband_flux.shape, np.nan, dtype=np.float64
            )
            predicted_end_to_end_flux = (
                np.full(broadband_flux.shape, np.nan, dtype=np.float64)
                if adm_retrieved_fluxes is not None
                else None
            )
            fold_number = np.full(broadband_flux.shape, -1, dtype=np.int16)
            n2bc_fold_metrics = []
            end_to_end_fold_metrics = []
            degree_fold_models = []
            for fold_index, (train_indices, test_indices) in enumerate(
                fold_splits, start=1
            ):
                if set(scene_ids[train_indices]) & set(scene_ids[test_indices]):
                    raise RuntimeError("scene-index leakage across validation folds")

                if training_mode == "unstratified":
                    model = make_pipeline(
                        StandardScaler(),
                        PolynomialFeatures(degree=degree, include_bias=False),
                        LinearRegression(),
                    )
                    model.fit(
                        true_channel_fluxes[train_indices], broadband_flux[train_indices]
                    )
                    predicted_n2bc_flux[test_indices] = model.predict(
                        true_channel_fluxes[test_indices]
                    )
                    fold_models_for_test = {"all": model}
                    if predicted_end_to_end_flux is not None:
                        predicted_end_to_end_flux[test_indices] = model.predict(
                            adm_retrieved_fluxes[test_indices]
                        )
                else:
                    fold_models_for_test = {}
                    for regime in np.unique(regimes):
                        train_regime = train_indices[regimes[train_indices] == regime]
                        test_regime = test_indices[regimes[test_indices] == regime]
                        model = make_pipeline(
                            StandardScaler(),
                            PolynomialFeatures(degree=degree, include_bias=False),
                            LinearRegression(),
                        )
                        model.fit(
                            true_channel_fluxes[train_regime], broadband_flux[train_regime]
                        )
                        predicted_n2bc_flux[test_regime] = model.predict(
                            true_channel_fluxes[test_regime]
                        )
                        fold_models_for_test[str(regime)] = model
                        if predicted_end_to_end_flux is not None:
                            predicted_end_to_end_flux[test_regime] = model.predict(
                                adm_retrieved_fluxes[test_regime]
                            )

                fold_number[test_indices] = fold_index
                n2bc_fold_error_percent = (
                    predicted_n2bc_flux[test_indices] - broadband_flux[test_indices]
                ) / broadband_flux[test_indices] * 100.0
                n2bc_fold_metrics.append(
                    {
                        "fold": fold_index,
                        **summarize_errors(n2bc_fold_error_percent),
                    }
                )
                if predicted_end_to_end_flux is not None:
                    end_to_end_fold_error_percent = (
                        predicted_end_to_end_flux[test_indices]
                        - broadband_flux[test_indices]
                    ) / broadband_flux[test_indices] * 100.0
                    end_to_end_fold_metrics.append(
                        {
                            "fold": fold_index,
                            **summarize_errors(end_to_end_fold_error_percent),
                        }
                    )
                degree_fold_models.append(
                    {
                        "fold": fold_index,
                        "test_indices": test_indices.copy(),
                        "models": fold_models_for_test,
                    }
                )

            if (
                np.any(~np.isfinite(predicted_n2bc_flux))
                or np.any(fold_number < 1)
            ):
                raise RuntimeError("cross-validation did not predict every Sunny spectrum")
            if predicted_end_to_end_flux is not None and np.any(
                ~np.isfinite(predicted_end_to_end_flux)
            ):
                raise RuntimeError("end-to-end CV did not predict every Sunny spectrum")

            n2bc_error_percent = (
                predicted_n2bc_flux - broadband_flux
            ) / broadband_flux * 100.0
            n2bc_regime_metrics = {
                regime: summarize_errors(n2bc_error_percent[regimes == regime])
                for regime in np.unique(regimes)
            }
            training_mode_results[f"degree_{degree}"] = {
                "requirement_id": requirement_id,
                "n2bc_only": {
                    "overall": summarize_errors(n2bc_error_percent),
                    "by_regime": n2bc_regime_metrics,
                    "fold_metrics": n2bc_fold_metrics,
                },
                "end_to_end": None,
            }
            if predicted_end_to_end_flux is not None:
                end_to_end_error_percent = (
                    predicted_end_to_end_flux - broadband_flux
                ) / broadband_flux * 100.0
                training_mode_results[f"degree_{degree}"]["end_to_end"] = {
                    "overall": summarize_errors(end_to_end_error_percent),
                    "by_regime": {
                        regime: summarize_errors(
                            end_to_end_error_percent[regimes == regime]
                        )
                        for regime in np.unique(regimes)
                    },
                    "fold_metrics": end_to_end_fold_metrics,
                }
            training_mode_models[f"degree_{degree}"] = degree_fold_models
            for index in range(broadband_flux.size):
                residual_rows.append(
                    {
                        "requirement_id": requirement_id,
                        "training_mode": training_mode,
                        "regression_degree": degree,
                        "scene_index": scene_ids[index],
                        "regime": regimes[index],
                        "fold": int(fold_number[index]),
                        "assessment_stage": "n2bc_only",
                        "reference_olr_w_m2": float(broadband_flux[index]),
                        "predicted_olr_w_m2": float(predicted_n2bc_flux[index]),
                        "relative_error_percent": float(n2bc_error_percent[index]),
                    }
                )
                if predicted_end_to_end_flux is not None:
                    residual_rows.append(
                        {
                            "requirement_id": requirement_id,
                            "training_mode": training_mode,
                            "regression_degree": degree,
                            "scene_index": scene_ids[index],
                            "regime": regimes[index],
                            "fold": int(fold_number[index]),
                            "assessment_stage": "end_to_end",
                            "reference_olr_w_m2": float(broadband_flux[index]),
                            "predicted_olr_w_m2": float(
                                predicted_end_to_end_flux[index]
                            ),
                            "relative_error_percent": float(
                                end_to_end_error_percent[index]
                            ),
                        }
                    )
        model_results[training_mode] = training_mode_results
        fold_models[training_mode] = training_mode_models
    return model_results, residual_rows, fold_models


def apply_fold_models(
    features, broadband_flux, scene_ids, regimes, fold_models, assessment_stage
):
    """Apply saved grouped-CV N2BC models to one narrowband-flux input set."""
    results = {}
    residual_rows = []
    for training_mode, degree_models in fold_models.items():
        mode_results = {}
        for degree_name, folds in degree_models.items():
            predicted_flux = np.full(broadband_flux.shape, np.nan, dtype=np.float64)
            fold_number = np.full(broadband_flux.shape, -1, dtype=np.int16)
            fold_metrics = []
            for fold in folds:
                test_indices = fold["test_indices"]
                models = fold["models"]
                if "all" in models:
                    predicted_flux[test_indices] = models["all"].predict(
                        features[test_indices]
                    )
                else:
                    for regime, model in models.items():
                        test_regime = test_indices[regimes[test_indices] == regime]
                        predicted_flux[test_regime] = model.predict(features[test_regime])
                fold_number[test_indices] = fold["fold"]
                fold_error = (
                    predicted_flux[test_indices] - broadband_flux[test_indices]
                ) / broadband_flux[test_indices] * 100.0
                fold_metrics.append(
                    {"fold": int(fold["fold"]), **summarize_errors(fold_error)}
                )

            if np.any(~np.isfinite(predicted_flux)) or np.any(fold_number < 1):
                raise RuntimeError(f"{assessment_stage} CV did not predict every record")
            error_percent = (predicted_flux - broadband_flux) / broadband_flux * 100.0
            mode_results[degree_name] = {
                "overall": summarize_errors(error_percent),
                "by_regime": {
                    regime: summarize_errors(error_percent[regimes == regime])
                    for regime in np.unique(regimes)
                },
                "fold_metrics": fold_metrics,
            }
            for index in range(broadband_flux.size):
                residual_rows.append(
                    {
                        "assessment_stage": assessment_stage,
                        "training_mode": training_mode,
                        "regression_degree": degree_name.removeprefix("degree_"),
                        "scene_index": scene_ids[index],
                        "regime": regimes[index],
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
        "envelope_band_um": scenario.get("envelope_band_um"),
        "envelope_edge_slope_per_um": scenario.get(
            "envelope_edge_slope_per_um"
        ),
        "idealized_response_note": (
            "RfMA band limits do not specify flight SRFs; sigmoid or triangular responses are scenario assumptions."
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
            "limitation": "Library metadata supports clear/cloudy stratification only; shared indices are conservatively grouped but not assumed to be identical atmospheric profiles.",
        },
        "validation": {
            "method": "GroupKFold out-of-fold predictions",
            "group_key": "four-digit shared clear/cloud scene index",
            "folds": args.folds,
            "k2_convention": "2 times sample standard deviation; provisional comparison convention, not confirmed identical to the RfMA fitting-noise definition.",
        },
        "provenance": evaluation_provenance(
            args, {"sunny_library": args.sunny_dir.resolve()}
        ),
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
    model_bundle = {
        "schema_version": 1,
        "folds": args.folds,
        "requirement_id": "ObsReq_16",
        "scenarios": {},
    }
    metrics = {**result_metadata, "stage": "n2bc_only_true_band_flux", "scenarios": {}}

    for scenario_id, scenario in scenarios.items():
        print(f"Fitting independent N2BC models for {scenario_id}", flush=True)
        true_narrowband_fluxes = integrate_true_channel_fluxes(
            spectral_data, scenario, channel_catalog["default_edge_slope_per_um"]
        )
        regressions, _, fold_models = evaluate_scenario(
            true_narrowband_fluxes,
            broadband_flux,
            scene_ids,
            regimes,
            args.folds,
            "ObsReq_16",
        )
        metadata = scenario_metadata(scenario_id, scenario, channel_catalog)
        metrics["scenarios"][scenario_id] = {
            **metadata,
            "regressions": {
                mode: {
                    degree: values["n2bc_only"]
                    for degree, values in degree_results.items()
                }
                for mode, degree_results in regressions.items()
            },
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
        "scenarios": {},
        "provenance": evaluation_provenance(
            args, {"sunny_library": args.sunny_dir.resolve()}
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

        n2bc_only_results, _ = apply_fold_models(
            model_data["true_narrowband_fluxes"],
            model_data["broadband_flux"],
            scene_ids,
            regimes,
            model_data["fold_models"],
            "n2bc_only",
        )
        end_to_end_results, _ = apply_fold_models(
            retrieved_fluxes,
            model_data["broadband_flux"],
            scene_ids,
            regimes,
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
    parser.add_argument("--adm-output-dir", type=Path, default=DEFAULT_ADM_DIR)
    parser.add_argument("--adm-metrics-output", type=Path, default=DEFAULT_ADM_METRICS)
    parser.add_argument("--adm-metrics-input", type=Path, default=DEFAULT_ADM_METRICS)
    parser.add_argument(
        "--abi-adm-evidence",
        type=Path,
        default=ROOT / "data" / "uncertainty" / "abi_adm_proxy_summary.json",
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
