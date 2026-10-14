"""Isolated ECO multi-view geometry diagnostic on the nominal Sunny/ECO estimator.

Uses the nominal budget Sunny library (ECO goal channels, sunny.npz) and its
grouped held-out global N2BC fold models (cv_models.joblib), so the dense
regularized row is the nominal angular configuration with per-file fits.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import yaml

from geometry_sensitivity import retrieve_geometry
from sensitivity import summary
from uncertainty import build_provenance, check_provenance

SETTINGS = ROOT / "config/sensitivity_extensions.yaml"


def settings():
    with SETTINGS.open() as source:
        options = yaml.safe_load(source)
    if options["schema_version"] != 2:
        raise ValueError("Extension settings must use schema_version 2 (geometry only)")
    return options


def provenance(inputs):
    return build_provenance(
        root=ROOT,
        code_paths=[Path(__file__), ROOT / "src/geometry_sensitivity.py",
                    *[ROOT / f"src/{name}.py" for name in (
                        "sensitivity", "uncertainty", "adm", "adm_fitting",
                    )]],
        configuration_paths=[SETTINGS, ROOT / "config.yaml"],
        input_paths={str(i): Path(path) for i, path in enumerate(inputs)},
    )


def save(path, report, inputs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report["provenance"] = provenance(inputs)
    with path.open("w") as target:
        json.dump(report, target, indent=2, allow_nan=False)


def current(path, provenance_file=None):
    record = json.loads(Path(provenance_file or str(path) + ".json").read_text())
    freshness = check_provenance(record["provenance"], ROOT)
    if freshness["status"] != "current":
        raise ValueError(f"Stale nominal input {path}: {freshness}")
    return record


def nominal_predict(flux, models):
    """Grouped held-out prediction[n] with each fold's global weighted N2BC."""
    prediction = np.full(len(flux), np.nan)
    for fold in models:
        test = fold["test"]
        prediction[test] = fold["regression"].predict(fold["poly"].transform(fold["scaler"].transform(flux[test])))
    if np.any(~np.isfinite(prediction)):
        raise RuntimeError("Nominal CV models did not predict every Sunny record")
    return prediction


def weighted(values, weights):
    """Population-weighted (ABI scene masses) bias and RMSE, as in the nominal budget."""
    return {"bias": float(weights @ values), "rmse": float(np.sqrt(weights @ values**2))}


def geometry(args, options):
    current(args.cache)
    budget = current(args.budget, args.budget)
    models = joblib.load(args.model)
    with np.load(args.cache) as source:
        data = {key: source[key] for key in source.files}
    with np.load(args.residuals) as source:
        weights = source["weights"]
    data["radiance"], data["noise_sd"], data["true_flux"] = (
        data.pop("eco_radiance"), data.pop("eco_noise_sd"), data.pop("true_band_flux"))
    dense, _ = retrieve_geometry(data["radiance"], data["angles"], "regularized")
    baseline = nominal_predict(dense, models)
    spectral = nominal_predict(data["true_flux"], models)
    rows = []
    for name, angles in options["geometry_sets"].items():
        indices = []
        for angle in angles:
            match = np.flatnonzero(data["angles"] == angle)
            if len(match) != 1:
                raise ValueError(f"Angle {angle} absent from Sunny cache")
            indices.append(match[0])
        for form in ("regularized", "quadratic"):
            if len(angles) < (2 if form == "regularized" else 3):
                rows.append({"geometry": name, "angles": angles, "form": form,
                             "status": "unidentifiable", "reason": "Too few views for amplitude and shape"})
                continue
            clean_prediction = None
            for noise_seed in [None, *options["noise_seeds"]]:
                radiance = data["radiance"]
                if noise_seed is not None:
                    # Same full-angle noise realization across geometries/forms.
                    radiance = radiance + np.random.default_rng(noise_seed).normal(
                        size=radiance.shape
                    ) * data["noise_sd"][:, None, :]
                flux, diagnostics = retrieve_geometry(radiance[:, indices], angles, form)
                prediction = nominal_predict(flux, models)
                if np.any(prediction <= 0):
                    raise ValueError("Geometry broadband prediction is non-positive")
                if noise_seed is None:
                    clean_prediction = prediction
                error = prediction - data["reference"]
                rows.append({
                    "geometry": name, "angles": angles, "form": form, "status": "quantified",
                    "noise_seed": noise_seed, "diagnostics": diagnostics,
                    "truth_error_w_m2": summary(error),
                    "population_weighted_truth_error_w_m2": weighted(error, weights),
                    "truth_error_percent": summary(error / data["reference"] * 100),
                    "angular_increment_w_m2": summary(prediction - spectral),
                    "delta_to_dense_regularized_w_m2": summary(prediction - baseline),
                    "noise_only_delta_w_m2": summary(prediction - clean_prediction),
                    "band_relative_error_percent": summary((flux-data["true_flux"])/data["true_flux"]*100),
                    "regimes": {str(regime): summary(error[data["regimes"] == regime])
                                for regime in np.unique(data["regimes"])},
                })
            print(f"Geometry {name}/{form} completed", flush=True)
    save(args.output, {
        "settings": options, "rows": rows,
        "scene_count": len(data["reference"]), "group_count": len(np.unique(data["groups"])),
        "scenario": budget["settings"]["scenario"],
        "spectral_only_truth_error_w_m2": summary(spectral-data["reference"]),
        "population_weighted_spectral_only_truth_error_w_m2": weighted(spectral-data["reference"], weights),
        "limitations": [
            "Idealized angle sets, not orbit-derived sampling; maximum 15 distinct Sunny views.",
            "Equal-weight fits; no single-view fallback, scene priors or angular weighting.",
            "Minimum-view fits have no residual degrees of freedom; identifiability is not accuracy.",
            "Gaussian channel/view noise independent at requirement reference temperature.",
            "Fixed grouped held-out nominal global N2BC fold models (population-weighted fit).",
            "Rows report unweighted Sunny-file statistics plus ABI-population-weighted bias/RMSE.",
            "Noisy repetitions measure conditional variability, not mission confidence intervals.",
            "Sunny library realism and reference spectral-domain mismatch remain unresolved.",
        ],
    }, [args.cache, str(args.cache)+".json", args.model, args.budget, args.residuals])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("geometry",))
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True, help="nominal sunny.npz")
    parser.add_argument("--model", required=True, help="nominal cv_models.joblib")
    parser.add_argument("--budget", required=True, help="nominal budget_report.json (provenance of cv_models)")
    parser.add_argument("--residuals", required=True, help="nominal residuals.npz (population weights)")
    args = parser.parse_args()
    geometry(args, settings())


if __name__ == "__main__":
    main()
