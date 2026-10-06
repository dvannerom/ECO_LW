"""ABI-informed, grouped-held-out ECO uncertainty budget."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

import joblib
import numpy as np
import yaml
from scipy.integrate import simpson
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score
from itertools import combinations
from netCDF4 import Dataset

from abi_sunny_comparison import (
    abi_response_on_grid, convolve_abi_radiances, synthetic_brightness_temperature,
    sunny_scene_files, blackbody_roundtrip,
)
from eco_spectral_response import load_channel_scenarios, channel_response_matrix
from geometry_sensitivity import retrieve_geometry
from nominal_budget import (
    integrate_domain, file_scene, population_weights, fit_pooled_shape, library_flux, error_classes,
    independent_budget_sum,
)
from run_sensitivity_convergence import collect_training, open_cache
from scene_features import spectral_features_from_bt, SPECTRAL_CHANNEL_INDICES
from radiometry import radiance_to_brightness_temperature
from sensitivity import fit_gmm, planck_derivative, fit_scene_adm, abi_broadband
from broadband import load_cubic_coefficients
from average_resolution import block_average_chunk
from find_nComponents import evaluate_candidate, evaluate_exact_candidate
from uncertainty import build_provenance, check_provenance

SETTINGS = ROOT/"config/nominal_budget.yaml"


def options():
    with SETTINGS.open() as source:
        settings = yaml.safe_load(source)
    with (ROOT/"config.yaml").open() as source:
        config = yaml.safe_load(source)
    if settings["schema_version"] != 1 or settings["noise_realizations"] < 2:
        raise ValueError("Unsupported nominal budget settings")
    return settings, config


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as target:
        json.dump(value, target, indent=2, allow_nan=False)


def provenance(inputs):
    return build_provenance(
        root=ROOT,
        code_paths=[Path(__file__), ROOT/"src/nominal_budget.py", ROOT/"src/geometry_sensitivity.py",
                    ROOT/"scripts/find_nComponents.py", ROOT/"scripts/average_resolution.py",
                    ROOT/"scripts/preprocess_data_ABI.py", ROOT/"src/broadband.py",
                    ROOT/"scripts/run_sensitivity_convergence.py",
                    *[ROOT/f"src/{name}.py" for name in (
                        "abi_sunny_comparison", "eco_spectral_response", "sensitivity",
                        "scene_features", "radiometry", "uncertainty", "adm", "adm_fitting",
                        "spectral_response",
                    )]],
        configuration_paths=[SETTINGS, ROOT/"config.yaml", ROOT/"config/eco_channel_scenarios.yaml"],
        input_paths={str(i): Path(path) for i, path in enumerate(inputs)},
    )


def verify(path):
    metadata = json.loads(Path(str(path)+".json").read_text())
    freshness = check_provenance(metadata["provenance"], ROOT)
    if freshness["status"] != "current":
        raise ValueError(f"Stale nominal budget input {path}: {freshness}")


def cache_paths(settings, days):
    return [Path(settings["cache_root"])/str(day) for day in days]


def select_nominal_candidate(eligible, components):
    """Select initialization at a manually configured count; never override it."""
    matching = [row for row in eligible if row["components"] == components]
    if not matching:
        raise ValueError(f"Nominal GMM count {components} has no adequately occupied candidate")
    return max(matching, key=lambda row: row["heldout_mean_log_likelihood"])


def train(settings):
    training = cache_paths(settings, settings["training_days"])
    validation = cache_paths(settings, settings["validation_days"])
    if set(training) & set(validation):
        raise ValueError("ABI day leakage")
    features = collect_training(training, settings["tiles_per_day"], settings["sample_points"],
                                5, 42)["features"][:, :10]
    validation_samples = [
        collect_training([path], settings["tiles_per_day"],
                         settings["sample_points"]//len(validation), 5, 73)["features"][:, :10]
        for path in validation
    ]
    heldout = np.concatenate(validation_samples)
    rows, candidates, assignments = [], {}, {}
    for components in settings["components"]:
        for seed in settings["seeds"]:
            model = fit_gmm(features, components, seed, 2)
            transformed = model[1].transform(model[0].transform(heldout))
            likelihood = float(model[-1].score(transformed))
            labels = model.predict(heldout)
            counts = np.bincount(labels, minlength=components)
            per_day = [float(model[-1].score(model[1].transform(model[0].transform(sample))))
                       for sample in validation_samples]
            row = {"components": components, "seed": seed, "heldout_mean_log_likelihood": likelihood,
                   "heldout_day_log_likelihood": per_day,
                   "heldout_counts": counts.tolist(), "minimum_occupancy": float(counts.min()/len(labels))}
            rows.append(row)
            if np.all(counts >= 100):
                candidates[(components, seed)] = model
                assignments[(components, seed)] = labels
            print(f"Spectral GMM k={components}, seed={seed}: held-out likelihood {likelihood:.4f}", flush=True)
    if not candidates:
        raise ValueError("No occupied, converged GMM candidate")
    eligible = [row for row in rows if (row["components"], row["seed"]) in candidates]
    best = max(eligible, key=lambda row: row["heldout_mean_log_likelihood"])
    day_se = float(np.std(best["heldout_day_log_likelihood"], ddof=1)/np.sqrt(len(validation)))
    threshold = best["heldout_mean_log_likelihood"]-day_se
    suggested = min(row["components"] for row in eligible
                    if row["heldout_mean_log_likelihood"] >= threshold)
    components = settings["nominal_components"]
    selected = select_nominal_candidate(eligible, components)
    seed = selected["seed"]
    model = candidates[(components, seed)]
    stability = [{
        "seed_a": first, "seed_b": second,
        "adjusted_rand": float(adjusted_rand_score(assignments[(components, first)],
                                                  assignments[(components, second)])),
    } for first in settings["seeds"] for second in settings["seeds"]
       if first < second and (components, first) in assignments and (components, second) in assignments]
    # Count complete cached eligible population, rather than the fitting sample.
    population_counts = np.zeros(components, dtype=np.int64)
    by_day = {}
    for path in validation:
        index, data = open_cache(path)
        counts = np.zeros(components, dtype=np.int64)
        for tile in range(settings["tiles_per_day"]):
            valid = np.repeat(np.repeat(data["valid_blocks"][tile], 5, axis=0), 5, axis=1)
            counts += np.bincount(model.predict(data["features"][tile][valid, :10]), minlength=components)
        by_day[str(index["day"])] = counts.tolist()
        population_counts += counts
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    path = output/"spectral_gmm.joblib"
    joblib.dump({"model": model, "population": population_counts/population_counts.sum()}, path)
    metadata = {
        "provenance": provenance([*training, *validation]), "components": components, "seed": seed,
        "selection": "Manually configured nominal count; best held-out initialization at that count",
        "diagnostic_one_se_components": suggested,
        "best_likelihood_components": best["components"], "day_standard_error": day_se,
        "likelihood_threshold": threshold, "initialization_stability": stability,
        "best_at_search_boundary": best["components"] == max(settings["components"]),
        "heldout_log_density_p01": float(np.quantile(
            model[-1].score_samples(model[1].transform(model[0].transform(heldout))), .01)),
        "population_counts": population_counts.tolist(), "by_day": by_day,
        "population_scope": "Eligible complete overlap tiles on four held-out days, equal native-pixel weight; not global climatology",
        "selection_caution": "Held-out days used for selection, not an independent final classifier accuracy test; stability diagnostics retained",
    }
    write(str(path)+".json", metadata)
    write(Path(settings["diagnostic_dir"])/"gmm_selection.json", {"candidates": rows, **metadata})


def diagnostic_observations(records, planck):
    """Adapt cached six-channel pairs to the existing nine-channel diagnostics."""
    radiance = np.full((len(records["features"]), 2, 9), np.nan)
    radiance[:, :, list(SPECTRAL_CHANNEL_INDICES)] = records["radiance"]
    coefficients = np.full((2, 9, 4), np.nan)
    coefficients[:, list(SPECTRAL_CHANNEL_INDICES)] = planck
    return (records["angles"][:, 0], records["angles"][:, 1],
            radiance[:, 0], radiance[:, 1], coefficients[0], coefficients[1])


def diagnostics(settings):
    """Reuse production component diagnostics on texture-free cached samples."""
    training_paths = cache_paths(settings, settings["training_days"])
    validation_paths = cache_paths(settings, settings["validation_days"])
    training = collect_training(training_paths, settings["tiles_per_day"],
                                settings["sample_points"], 5, 42)
    train_features = training["features"][:, :10]
    _, cache = open_cache(training_paths[0])
    train_obs = diagnostic_observations(training, cache["planck"])
    validation = []
    for path in validation_paths:
        records = collect_training([path], settings["tiles_per_day"],
                                   settings["sample_points"]//len(validation_paths), 5, 73)
        _, cache = open_cache(path)
        validation.append((records["features"][:, :10], str(path),
                           diagnostic_observations(records, cache["planck"])))
    features = np.concatenate([entry[0] for entry in validation])
    # Each day retains its own Planck coefficients for exact broadband scoring.
    observations = tuple(np.concatenate([entry[2][i] for entry in validation])
                         for i in range(4))+validation[0][2][4:]
    rows = []
    previous = None
    for count in settings["components"]:
        row, state = evaluate_candidate(
            count, train_features, features, train_obs, observations,
            [(entry[0], entry[1]) for entry in validation],
            True, .98, 42, 3, settings["nominal_components"])
        exact = evaluate_exact_candidate(count, train_obs, state["train_labels"],
            [(entry[2], state["validation_labels"][entry[1]]) for entry in validation])
        for name, value in zip((
            "exact_raw_flux_std", "exact_corrected_flux_std", "exact_flux_improvement",
            "exact_adm_ratio_rmse", "exact_worst_channel_adm_rmse",
            "exact_worst_scene_adm_rmse"), exact):
            row[name] = value
        if not all(np.isfinite(row[key]) for key in (
                "adm_ratio_rmse", "exact_adm_ratio_rmse", "ari_min", "exact_corrected_flux_std")):
            raise ValueError(f"Component {count} diagnostics lack valid ADM/flux/stability scores")
        row["shortlisted"] = True  # All budget candidates receive the exact stage.
        improvement = (previous-row["adm_ratio_rmse"])/previous if previous is not None else np.inf
        row.update(
            criterion_adm_plateau=bool(improvement < .01),
            criterion_occupancy=bool(row["test_min_fraction"] >= .005),
            criterion_stability=bool(row["occupancy_js"] <= .1 and row["centroid_shift"] <= 1
                                     and row["ari_min"] >= .8),
            criterion_angular_coverage=bool(row["minimum_angular_coverage"] >= .7),
            criterion_adm_coverage=bool(np.isfinite(row["worst_scene_adm_rmse"])),
            criterion_exact_adm_coverage=bool(np.isfinite(row["exact_worst_scene_adm_rmse"])),
            criterion_exact_improvement=bool(row["exact_flux_improvement"] >= 0),
        )
        row["all_criteria"] = all(value for key, value in row.items() if key.startswith("criterion_"))
        previous = row["adm_ratio_rmse"]
        rows.append(row)
        print(f"Four-panel diagnostic k={count} completed", flush=True)
    output = Path(settings["diagnostic_dir"])/"gmm_component_diagnostics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(str(output)+".json", {
        "provenance": provenance([*training_paths, *validation_paths]),
        "manual_components": settings["nominal_components"],
        "scope": "Existing find_nComponents scoring, spectral-only features, cached overlap population",
        "notes": [
            "Seed 42 diagnostic fits use existing five-initialization fitter and three bootstrap refits.",
            "Diagnostic fits are separate from the nominal two-initialization fit; no automatic count override.",
            "All candidates receive exact production ADM and cubic ABI broadband diagnostics.",
            "Pooled training uses first-day Planck coefficients, as in existing diagnostic tooling.",
            "Exact held-out flux scoring preserves per-day Planck coefficients.",
            "Coverage uses original nine 10-degree bins despite eligibility restricted to 70 degrees.",
        ],
    })


def spatial_tile(model, raw_radiance, radiance, angles, planck, library, coefficients,
                 valid, block, origin=(0, 0)):
    """Paired ABI footprint difference [W/m2]; all native pixels must be valid.

    Inputs are (y,x,2,6) radiances, (y,x,2) angles, valid[y,x].
    Mean satellite broadband flux is used in both routes. Blocks align to
    the full-grid origin; incomplete tile-edge blocks are excluded.
    """
    height, width = valid.shape
    if not isinstance(block, int) or block < 1:
        raise ValueError("Spatial block size must be a positive integer")
    start_y, start_x = (-int(origin[0])) % block, (-int(origin[1])) % block
    end_y = start_y+(height-start_y)//block*block
    end_x = start_x+(width-start_x)//block*block
    if end_y <= start_y or end_x <= start_x:
        raise ValueError("No complete aligned blocks within tile")
    region = (slice(start_y, end_y), slice(start_x, end_x))
    raw_radiance, radiance, angles, valid = (
        values[region] for values in (raw_radiance, radiance, angles, valid))
    height, width = valid.shape
    complete = block_average_chunk(valid.astype(float), block) == 1
    if not complete.any():
        raise ValueError("No complete matched spatial footprints")
    native = np.repeat(np.repeat(complete, block, axis=0), block, axis=1)
    labels = model.predict(abi_noise_features(raw_radiance[native], planck, 0))
    fine = np.zeros((height, width))
    fine[native] = abi_broadband(radiance[native], angles[native], planck, labels,
                                library, "regularized", coefficients).mean(axis=1)
    reference = block_average_chunk(fine, block)[complete]
    coarse_raw = block_average_chunk(raw_radiance, block)[complete]
    coarse_rad = block_average_chunk(radiance, block)[complete]
    coarse_angles = block_average_chunk(angles, block)[complete]
    coarse_labels = model.predict(abi_noise_features(coarse_raw, planck, 0))
    retrieved = abi_broadband(coarse_rad, coarse_angles, planck, coarse_labels,
                             library, "regularized", coefficients).mean(axis=1)
    return retrieved-reference


def spatial(settings, config):
    """Fit fixed ABI proxy ADMs on training days; score matched held-out tiles."""
    directory = Path(settings["output_dir"])
    model_path = directory/"spectral_gmm.joblib"
    verify(model_path)
    fitted = joblib.load(model_path)
    model, population = fitted["model"], fitted["population"]
    training_paths = cache_paths(settings, settings["training_days"])
    training = collect_training(training_paths, settings["tiles_per_day"],
                                settings["sample_points"], 5, 42)
    labels = model.predict(training["features"][:, :10])
    library = np.empty((len(population), 6, 1))
    for scene in range(len(population)):
        selected = labels == scene
        for channel in range(6):
            library[scene, channel] = fit_scene_adm(
                training["angles"][selected], training["radiance"][selected, :, channel], "regularized")
    coefficient_path = config["narrowband_to_broadband_coeffs_file"]
    coefficients = load_cubic_coefficients(coefficient_path)
    blocks = settings["spatial"]["block_sizes"]
    nominal = settings["spatial"]["nominal_block_size"]
    if (not blocks or blocks[0] != 1 or nominal not in blocks
            or sorted(set(blocks)) != blocks
            or any(not isinstance(block, int) or block < 1 for block in blocks)
            or nominal*settings["spatial"]["native_resolution_km"] != settings["product_resolution_km"]):
        raise ValueError("Spatial ladder must include identity and nominal product footprint")
    residuals = {block: [] for block in blocks}
    day_rows = {}
    inputs = [model_path, str(model_path)+".json", coefficient_path, *training_paths]
    for path in cache_paths(settings, settings["validation_days"]):
        index, data = open_cache(path)
        sources = [entry["path"] for entry in index["provenance"]["inputs"].values()]
        if len(sources) != 1:
            raise ValueError("Expected one preprocessed ABI source per cache")
        source_path = ROOT/sources[0]
        inputs.extend([path, source_path])
        daily = {block: [] for block in blocks}
        with Dataset(source_path) as source:
            for tile in range(settings["tiles_per_day"]):
                valid = np.repeat(np.repeat(data["valid_blocks"][tile], 5, axis=0), 5, axis=1)
                y, x = data["origins"][tile]
                height, width = valid.shape
                raw = np.stack([
                    np.asarray(np.ma.filled(source.variables[f"rad_{sat}_interp"][
                        y:y+height, x:x+width, list(SPECTRAL_CHANNEL_INDICES)], np.nan))
                    for sat in ("G16", "G18")], axis=2)
                for block in blocks:
                    errors = spatial_tile(model, raw, data["radiance"][tile], data["angles"][tile],
                                          data["planck"], library, coefficients, valid, block,
                                          origin=(y, x))
                    daily[block].append(errors)
        day_rows[str(index["day"])] = {}
        for block in blocks:
            errors = np.concatenate(daily[block])
            residuals[block].append(errors)
            day_rows[str(index["day"])][str(block)] = {
                "footprints": len(errors), **error_classes(errors[:, None], np.full(len(errors), 1/len(errors)))}
        print(f"Spatial processing day {index['day']} completed", flush=True)
    rows = []
    combined = {}
    for block in blocks:
        errors = np.concatenate(residuals[block])
        combined[f"block_{block}"] = errors
        rows.append({"block_size": block,
                     "resolution_km": block*settings["spatial"]["native_resolution_km"],
                     "footprints": len(errors),
                     **error_classes(errors[:, None], np.full(len(errors), 1/len(errors)))})
    if np.max(np.abs(combined["block_1"])) > 1e-9:
        raise RuntimeError("Identity spatial route does not close")
    np.savez_compressed(directory/"spatial_residuals.npz", **combined)
    joblib.dump(library, directory/"abi_spatial_adm.joblib")
    write(directory/"spatial_report.json", {
        "provenance": provenance(inputs), "rows": rows, "by_day": day_rows,
        "nominal": next(row for row in rows if row["block_size"] == nominal),
        "assumptions": [
            "Coarse-minus-mean-fine retrieval is a processing difference, not absolute flux truth.",
            "Fixed clean-trained spectral GMM, training-day ABI scene ADMs, existing ABI cubic N2BC.",
            "Uncorrected radiances define features; corrected radiances and angles define retrieval.",
            "Equal-area box averages reuse average_resolution.block_average_chunk; complete valid blocks only.",
            "Blocks align to full-grid origin; incomplete tile-edge blocks excluded independently per resolution.",
            "Resolution-specific footprint counts differ; population consists of the same sampled tiles, not full overlap.",
            "ABI two-view spatial response assumed transferable to ECO fifteen-view retrieval.",
            "Spatial contribution independent of existing Sunny-source errors (user work hypothesis).",
            "Deterministic comparison: random SD zero excludes spatial/noise interactions.",
            "Existing ABI cubic spectral reference differs from nominal ECO 4-100 um regression.",
        ],
    })


def abi_noise_features(radiance, planck, noise):
    """Recompute mean-satellite spectral features from ABI radiances[n,2,6]."""
    noisy = np.asarray(radiance)+noise
    if np.any(~np.isfinite(noisy)) or np.any(noisy <= 0):
        raise ValueError("Added ABI noise produced non-positive or invalid radiance")
    bt = np.stack([synthetic_brightness_temperature(noisy[:, satellite], planck[satellite])
                   for satellite in range(2)], axis=1)
    return spectral_features_from_bt(bt.mean(axis=1))


def assignment(settings):
    """Estimate scene transitions on a bounded uniform sample of held-out ABI."""
    directory = Path(settings["output_dir"])
    model_path = directory/"spectral_gmm.joblib"
    verify(model_path)
    fitted = joblib.load(model_path)
    model, population = fitted["model"], fitted["population"]
    k = len(population)
    capacity = settings["sample_points"]
    rng = np.random.default_rng(settings["noise_seed"])
    keys = np.empty(0)
    records = {name: None for name in ("radiance", "planck", "features", "day")}
    inputs = [model_path, str(model_path)+".json"]
    for path in cache_paths(settings, settings["validation_days"]):
        index, cache = open_cache(path)
        sources = [entry["path"] for entry in index["provenance"]["inputs"].values()]
        if len(sources) != 1:
            raise ValueError("Expected one preprocessed ABI source per cache")
        source_path = ROOT/sources[0]
        inputs.extend([path, source_path])
        with Dataset(source_path) as source:
            planck = np.asarray(cache["planck"])
            for tile in range(settings["tiles_per_day"]):
                valid = np.repeat(np.repeat(cache["valid_blocks"][tile], 5, axis=0), 5, axis=1)
                y, x = cache["origins"][tile]
                height, width = valid.shape
                radiance = np.stack([
                    np.asarray(np.ma.filled(source.variables[f"rad_{sat}_interp"][
                        y:y+height, x:x+width, list(SPECTRAL_CHANNEL_INDICES)], np.nan))
                    for sat in ("G16", "G18")], axis=2)[valid]
                features = abi_noise_features(radiance, planck, 0)
                if not np.allclose(features, cache["features"][tile][valid, :10], atol=.002):
                    raise ValueError("Raw ABI radiance/BT features do not match cached GMM inputs")
                count = len(features)
                arrays = {
                    "radiance": radiance,
                    "planck": np.broadcast_to(planck, (count, 2, 6, 4)),
                    "features": features,
                    "day": np.full(count, index["day"]),
                }
                keys = np.r_[keys, rng.random(count)]
                for name, value in arrays.items():
                    records[name] = value if records[name] is None else np.concatenate((records[name], value))
                if len(keys) > capacity:
                    take = np.argpartition(keys, capacity-1)[:capacity]
                    keys = keys[take]
                    records = {name: value[take] for name, value in records.items()}
    if len(keys) < capacity:
        raise ValueError("Insufficient eligible held-out ABI pixels for noise assessment")
    labels = model.predict(records["features"])
    # ABI spectral-density dL/dT in native radiance units, evaluated at 255 K.
    fk1, fk2, bc1, bc2 = np.moveaxis(records["planck"], -1, 0)
    argument = fk2/(bc1+bc2*255.)
    exponential = np.exp(argument)
    noise_sd = settings["nedt_k_at_255"]*fk1*exponential*fk2*bc2/(
        np.expm1(argument)**2*(bc1+bc2*255.)**2)
    counts = np.zeros((k, k), dtype=np.int64)
    per_day = {str(day): np.zeros((k, k), dtype=np.int64) for day in settings["validation_days"]}
    for repetition in range(settings["noise_realizations"]):
        rng = np.random.default_rng(np.random.SeedSequence([settings["noise_seed"], repetition, 3]))
        noise = rng.normal(size=records["radiance"].shape)*noise_sd
        bt = np.empty_like(records["radiance"])
        for satellite in range(2):
            perturbed = records["radiance"][:, satellite]+noise[:, satellite]
            if np.any(perturbed <= 0) or np.any(~np.isfinite(perturbed)):
                raise ValueError("ABI noise realization has invalid radiances")
            coefficients = np.moveaxis(records["planck"][:, satellite], -1, 0)
            bt[:, satellite] = radiance_to_brightness_temperature(perturbed, coefficients)
        features = spectral_features_from_bt(bt.mean(axis=1))
        if np.any(~np.isfinite(features)) or np.any(bt <= 0) or np.any(bt >= 1000):
            raise ValueError("ABI noise realization has invalid BT features")
        noisy = model.predict(features)
        np.add.at(counts, (labels, noisy), 1)
        for day, matrix in per_day.items():
            selected = records["day"] == int(day)
            np.add.at(matrix, (labels[selected], noisy[selected]), 1)
    if np.any(counts.sum(axis=1) == 0):
        raise ValueError("ABI noise sample lacks a nominal scene")
    probabilities = counts/counts.sum(axis=1, keepdims=True)
    write(directory/"abi_assignment.json", {
        "provenance": provenance(inputs), "transition_counts": counts.tolist(),
        "transition_probabilities": probabilities.tolist(),
        "weighted_change_probability": float(population @ (1-np.diag(probabilities))),
        "sample_change_probability": float(1-np.trace(counts)/counts.sum()),
        "baseline_sample_counts": np.bincount(labels, minlength=k).tolist(),
        "by_day_transition_counts": {day: value.tolist() for day, value in per_day.items()},
        "sample_pixels": len(labels), "noise_realizations": settings["noise_realizations"],
        "nedt_k_at_255": settings["nedt_k_at_255"],
        "assumptions": [
            "Observed ABI radiances treated as truth; existing ABI noise is not removed.",
            "Fixed clean-trained GMM; independent Gaussian noise across satellites, channels and pixels.",
            "Scene-conditional ABI transitions assumed applicable to ECO and within-scene Sunny files.",
            "Transferred scene transitions independent of simulated ECO retrieval radiometric noise.",
            "Sampled eligible native overlap pixels, not global or full-disk coverage.",
        ],
    })


def draw_scene_transitions(labels, probabilities, repetitions, seed):
    """Sample ABI-derived conditional transitions for Sunny files[n]."""
    probabilities = np.asarray(probabilities)
    if (probabilities.ndim != 2 or probabilities.shape[0] != probabilities.shape[1]
            or np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Invalid conditional scene transition matrix")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4]))
    cumulative = probabilities.cumsum(axis=1)
    cumulative[:, -1] = 1
    draws = rng.random((len(labels), repetitions))
    return (draws[:, :, None] > cumulative[np.asarray(labels), None, :]).sum(axis=-1)


def prepare(settings, config):
    files = sunny_scene_files(config["sunny_dir"])
    scenario = load_channel_scenarios(config["eco_channel_scenarios_file"])[settings["scenario"]]
    with Path(config["eco_channel_scenarios_file"]).open() as source:
        catalog = yaml.safe_load(source)
    _, cache = open_cache(cache_paths(settings, settings["training_days"])[0])
    planck = cache["planck"][0]
    arrays = {key: [] for key in ("eco_radiance", "abi_radiance", "true_band_flux", "reference",
                                   "eco_noise_sd", "groups", "regimes")}
    angles = np.asarray(settings["angles_deg"])
    previous = None
    roundtrip = 0.
    for path, regime, group in files:
        data = np.loadtxt(path)
        wavelength = data[:, 0]
        if data.shape[1] != 20 or np.any(~np.isfinite(data)) or np.any(data[:, 1:] < 0):
            raise ValueError(f"Invalid Sunny spectrum {path}")
        if previous is None or not np.array_equal(previous, wavelength):
            responses, normalization = abi_response_on_grid(wavelength, config["goes_filter_dir"])
            roundtrip = max(roundtrip, blackbody_roundtrip(wavelength, responses, normalization, planck))
            if roundtrip > .5:
                raise ValueError("ABI SRF/Planck transfer fails blackbody round-trip")
            eco = channel_response_matrix(wavelength, scenario, catalog["default_edge_slope_per_um"])
            derivative = planck_derivative(wavelength, 255)
            eco_sigma = simpson(derivative[:, None]*eco, x=wavelength, axis=0)*settings["nedt_k_at_255"]
            previous = wavelength.copy()
        indices = (angles/5).astype(int)
        directional = data[:, 2+indices]
        arrays["eco_radiance"].append(simpson(directional[:, :, None]*eco[:, None, :], x=wavelength, axis=0))
        arrays["abi_radiance"].append(convolve_abi_radiances(wavelength, directional, responses, normalization))
        arrays["true_band_flux"].append(simpson(data[:, 1, None]*eco, x=wavelength, axis=0))
        arrays["reference"].append(integrate_domain(wavelength, data[:, 1], *settings["reference_domain_um"]))
        arrays["eco_noise_sd"].append(eco_sigma)
        arrays["groups"].append(group)
        arrays["regimes"].append(regime)
    output = Path(settings["output_dir"])/"sunny.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{key: np.asarray(value) for key, value in arrays.items()},
                        planck=planck, angles=angles)
    write(str(output)+".json", {
        "provenance": provenance([*[path for path, _, _ in files], config["goes_filter_dir"],
                                  cache_paths(settings, settings["training_days"])[0]]),
        "reference_domain_um": settings["reference_domain_um"], "roundtrip_error_K": roundtrip,
        "classifier_noise": "Measured separately on held-out ABI; no noisy Sunny classification",
    })


def classify(model, radiance, planck):
    shape = radiance.shape[:2]
    bt = synthetic_brightness_temperature(radiance.reshape(-1, 6), planck)
    posterior = model.predict_proba(spectral_features_from_bt(bt)).reshape(*shape, -1)
    return file_scene(posterior)


def density(model, radiance, planck):
    bt = synthetic_brightness_temperature(radiance.reshape(-1, 6), planck)
    transformed = model[1].transform(model[0].transform(spectral_features_from_bt(bt)))
    return model[-1].score_samples(transformed).reshape(radiance.shape[:2])


def fit_n2bc(flux, reference, weights):
    scaler = StandardScaler().fit(flux, sample_weight=weights)
    poly = PolynomialFeatures(2, include_bias=False)
    design = poly.fit_transform(scaler.transform(flux))
    regression = LinearRegression().fit(design, reference, sample_weight=weights)
    return scaler, poly, regression


def calculate(settings):
    directory = Path(settings["output_dir"])
    model_path, data_path = directory/"spectral_gmm.joblib", directory/"sunny.npz"
    verify(model_path)
    verify(data_path)
    fitted = joblib.load(model_path)
    model, population = fitted["model"], fitted["population"]
    assignment_path = directory/"abi_assignment.json"
    assignment_report = json.loads(assignment_path.read_text())
    freshness = check_provenance(assignment_report["provenance"], ROOT)
    if freshness["status"] != "current":
        raise ValueError(f"Stale ABI assignment transitions: {freshness}")
    with np.load(data_path) as source:
        data = {key: source[key] for key in source.files}
    labels, consistency = classify(model, data["abi_radiance"], data["planck"])
    weights, support = population_weights(labels, population)
    k = len(population)
    n, repetitions = len(labels), settings["noise_realizations"]
    components = {key: np.zeros((n, repetitions)) for key in (
        "spectral", "angular", "radiometric", "assignment", "total",
    )}
    noisy_labels = draw_scene_transitions(
        labels, assignment_report["transition_probabilities"], repetitions, settings["noise_seed"])
    per_file_flux, _ = retrieve_geometry(data["eco_radiance"], data["angles"], "regularized")
    models, fold_diagnostics = [], []
    per_file_delta = np.zeros(n)
    for fold, (training, test) in enumerate(GroupKFold(settings["folds"]).split(labels, groups=data["groups"])):
        if set(data["groups"][training]) & set(data["groups"][test]):
            raise RuntimeError("Sunny group leakage")
        train_weights, train_support = population_weights(labels[training], population)
        library = np.zeros((k, 6, 1))
        present = np.zeros(k, dtype=bool)
        for scene in np.unique(labels[training]):
            selected = training[labels[training] == scene]
            library[scene] = fit_pooled_shape(data["eco_radiance"][selected], data["angles"])
            present[scene] = True
        if np.any(~present[labels[test]]):
            raise ValueError(f"Fold {fold}: evaluated scene lacks training ADM support")
        probabilities = np.asarray(assignment_report["transition_probabilities"])
        possible_destinations = np.any(probabilities[labels[test]] > 0, axis=0)
        if np.any(possible_destinations & ~present):
            raise ValueError(f"Fold {fold}: nonzero ABI transition probability to an unsupported ADM")
        # Weighted scaler and global N2BC; known clear/cloud labels are not used for prediction.
        scaler, poly, regression = fit_n2bc(data["true_band_flux"][training],
                                           data["reference"][training], train_weights)

        def predict(flux):
            prediction = regression.predict(poly.transform(scaler.transform(flux)))
            if np.any(prediction <= 0) or np.any(~np.isfinite(prediction)):
                raise ValueError("Invalid weighted N2BC prediction")
            return prediction

        spectral = predict(data["true_band_flux"][test])
        clean_flux = library_flux(data["eco_radiance"][test], data["angles"], labels[test], library)
        clean = predict(clean_flux)
        per_file_delta[test] = clean-predict(per_file_flux[test])
        components["spectral"][test] = (spectral-data["reference"][test])[:, None]
        components["angular"][test] = (clean-spectral)[:, None]
        for repetition in range(repetitions):
            rng = np.random.default_rng(np.random.SeedSequence([settings["noise_seed"], repetition, 2]))
            # Shared full-file noise supports exact pairing across folds.
            noise = rng.normal(size=data["eco_radiance"].shape)*data["eco_noise_sd"][:, None, :]
            perturbed = data["eco_radiance"][test]+noise[test]
            fixed = predict(library_flux(perturbed, data["angles"], labels[test], library))
            components["radiometric"][test, repetition] = fixed-clean
            switched = noisy_labels[test, repetition]
            available = present[switched]
            if not available.all():
                raise ValueError(f"Fold {fold}: ABI-transferred assignment destination lacks an ADM")
            changed_flux = library_flux(perturbed, data["angles"], switched, library)
            changed = predict(changed_flux)
            components["assignment"][test, repetition] = changed-fixed
            components["total"][test, repetition] = changed-data["reference"][test]
        models.append({"fold": fold, "test": test, "library": library, "present": present,
                       "training_groups": np.unique(data["groups"][training]),
                       "test_groups": np.unique(data["groups"][test]),
                       "scaler": scaler, "poly": poly, "regression": regression})
        fold_diagnostics.append({"fold": fold, "training_support": train_support,
                                 "test_count": len(test), "scene_shapes": library[:, :, 0].tolist()})
        print(f"Nominal grouped fold {fold+1} completed", flush=True)
    source_keys = ("spectral", "angular", "radiometric", "assignment")
    if not np.allclose(sum(components[key] for key in source_keys), components["total"], atol=1e-9):
        raise RuntimeError("Paired budget decomposition does not close")
    rows = [{"id": key, **error_classes(components[key], weights)}
            for key in (*source_keys, "total")]
    spatial_path = directory/"spatial_report.json"
    spatial_report = json.loads(spatial_path.read_text())
    freshness = check_provenance(spatial_report["provenance"], ROOT)
    if freshness["status"] != "current":
        raise ValueError(f"Stale spatial budget: {freshness}")
    if settings["spatial"]["independent_of_other_sources"] is not True:
        raise ValueError("Cross-population spatial combination requires explicit independence hypothesis")
    rows.extend([
        {"id": "spatial", **spatial_report["nominal"]},
        {"id": "combined", **independent_budget_sum(rows[-1], spatial_report["nominal"])},
    ])
    covariances = {}
    for first, second in combinations(source_keys, 2):
        a, b = components[first], components[second]
        ma, mb = float(weights @ a.mean(axis=1)), float(weights @ b.mean(axis=1))
        covariances[f"{first}:{second}"] = float(weights @ ((a-ma)*(b-mb)).mean(axis=1))
    scene_rows = {}
    for scene in np.unique(labels):
        selected = labels == scene
        mass = float(weights[selected].sum())
        if mass <= 0:
            scene_rows[str(scene)] = {"population_mass": 0, "status": "zero_ABI_population"}
            continue
        scene_rows[str(scene)] = {
            "population_mass": mass, "files": int(selected.sum()),
            "error_classes": {key: error_classes(components[key][selected], weights[selected]/mass)
                              for key in (*source_keys, "total")},
            "within_scene_comparison": error_classes(per_file_delta[selected, None], weights[selected]/mass),
            "assignment_change_probability": float(
                weights[selected]/mass @ (noisy_labels[selected] != labels[selected, None]).mean(axis=1)),
        }
    model_metadata = json.loads(Path(str(model_path)+".json").read_text())
    scores = density(model, data["abi_radiance"], data["planck"])
    low_density = scores < model_metadata["heldout_log_density_p01"]
    transfer = {
        "weighted_fraction_views_below_ABI_density_p01": float(weights @ low_density.mean(axis=1)),
        "weighted_fraction_files_with_any_low_density_view": float(weights @ low_density.any(axis=1)),
        "caution": "Density coverage diagnostic, not calibrated classification accuracy",
    }
    residual_path = directory/"residuals.npz"
    np.savez_compressed(residual_path, **components, weights=weights, labels=labels, noisy_labels=noisy_labels,
                        groups=data["groups"], regimes=data["regimes"], per_file_delta=per_file_delta)
    joblib.dump(models, directory/"cv_models.joblib")
    nominal_library = np.zeros((k, 6, 1))
    nominal_present = np.zeros(k, dtype=bool)
    for scene in np.unique(labels):
        nominal_library[scene] = fit_pooled_shape(data["eco_radiance"][labels == scene], data["angles"])
        nominal_present[scene] = True
    scaler, poly, regression = fit_n2bc(data["true_band_flux"], data["reference"], weights)
    joblib.dump({
        "library": nominal_library, "present": nominal_present, "model": model,
        "scaler": scaler, "poly": poly, "regression": regression, "population": population,
        "note": "All-library nominal model for application, not used for reported held-out errors",
    }, directory/"nominal_models.joblib")
    report = {
        "schema_version": 1, "settings": settings, "rows": rows, "support": support,
        "covariances_w_m4": covariances, "provenance": provenance([model_path, str(model_path)+".json",
                                                                  data_path, str(data_path)+".json",
                                                                  assignment_path, spatial_path]),
        "spatial": spatial_report,
        "assignment_transfer": {
            **assignment_report, "included_in_total": True,
            "simulated_Sunny_change_probability": float(weights @ (
                noisy_labels != labels[:, None]).mean(axis=1)),
            "probability_source": "Fixed-GMM baseline versus added-noise held-out ABI pixels",
        },
        "within_scene_spread": error_classes(per_file_delta[:, None], weights),
        "scene_rows": scene_rows, "transfer_diagnostics": transfer,
        "unweighted_rows": [{ "id": key, **error_classes(components[key], np.full(n, 1/n))}
                            for key in (*source_keys, "total")],
        "limitations": [
            "Conditional library simulation; ABI population is sampled eligible overlap, not global ECO climatology.",
            "ABI-defined spectral taxonomy; operational ECO classifier transfer not established.",
            "ABI noise transitions included under user-assumed similar ECO response; not validated ECO misclassification.",
            "Scene-only transition transfer ignores within-scene dependence on spectra and flux sensitivity.",
            "Transferred assignment draws independent of ECO-channel noise; physical joint covariance not calibrated.",
            "ABI observations treated as truth despite existing instrument noise; only added NEdT assessed.",
            "Weighted budget conditional on covered scenes; missing ABI mass explicitly reported.",
            "ABI 2-to-10 km box-average processing proxy transferred to ECO assuming independent errors.",
            "No ECO spatial PSF simulation; ABI two-view geometry and broadband coefficients differ from ECO.",
            "No CERES, calibration, SRF uncertainty, registration, evolution or library realism uncertainty.",
            "Monthly uncertainty not estimated: temporal covariance and sampling not available.",
            "Noise-free angular errors are scene-dependent; radiometer one-day averaging not assumed.",
            "Six-component nominal count manually configured; diagnostic likelihood recommendations do not override it.",
            "ABI mean-view BT versus individual Sunny-view transfer can change labels with angle.",
            "Scene-frequency weights do not correct within-scene spectral coverage or simulation realism.",
            "Zero random entries for deterministic sources exclude training/library uncertainty, not all possible randomness.",
        ],
    }
    write(directory/"budget_report.json", report)
    write(str(directory/"nominal_models.joblib")+".json", {
        "provenance": report["provenance"], "support": support,
        "use": "Nominal model; uncertainty assessed with separate CV models",
    })
    write(Path(settings["diagnostic_dir"])/"scene_support.json", {
        "support": support, "mean_view_consistency": float(consistency.mean()),
        "transfer": transfer, "scene_rows": scene_rows,
        "folds": fold_diagnostics, "assignment": report["assignment_transfer"],
        "provenance": report["provenance"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "prepare", "assignment", "diagnostics", "spatial", "calculate"))
    args = parser.parse_args()
    settings, config = options()
    {"train": lambda: train(settings), "prepare": lambda: prepare(settings, config),
     "assignment": lambda: assignment(settings), "diagnostics": lambda: diagnostics(settings),
     "spatial": lambda: spatial(settings, config),
     "calculate": lambda: calculate(settings)}[args.stage]()


if __name__ == "__main__":
    main()
