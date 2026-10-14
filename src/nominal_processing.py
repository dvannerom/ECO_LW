"""ABI-informed, grouped-held-out ECO uncertainty budget."""

import argparse
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
from itertools import combinations

from abi_sunny_comparison import (
    abi_response_on_grid, convolve_abi_radiances, synthetic_brightness_temperature,
    sunny_scene_files, blackbody_roundtrip,
)
from eco_spectral_response import load_channel_scenarios, channel_response_matrix
from geometry_sensitivity import retrieve_geometry
from nominal_budget import (
    integrate_domain, population_weights, fit_pooled_shape, library_flux, error_classes,
    classify, fit_n2bc,
    independent_budget_sum, lognormal_radiance,  # noqa: F401 lognormal_radiance is runner API
)
from scene_features import spectral_features_from_bt, SPECTRAL_CHANNEL_INDICES  # noqa: F401 runner API
from sensitivity import planck_derivative, abi_broadband
from spatial_averaging import block_average_chunk
from uncertainty import build_provenance, check_provenance
from nominal_provenance import (
    configuration_identity as nominal_configuration_identity,
    scoped_code_paths as nominal_scoped_code_paths,
    provenance_code_paths as nominal_provenance_code_paths,
)
from nominal_products import write_heldout_products
from product_paths import nominal_model_directory
from workflow_config import load_settings, settings_paths, resolve_components

SETTINGS = ROOT/"config/nominal_production.yaml"
CONFIG_PATH = ROOT/"config.yaml"
TAXONOMY_PATH = ROOT/"config/scene_taxonomy.yaml"


def options():
    from scene_taxonomy import load_scene_taxonomy
    settings = load_settings(SETTINGS)
    with CONFIG_PATH.open() as source:
        config = yaml.safe_load(source)
    if settings["schema_version"] != 1 or settings["noise_realizations"] < 2:
        raise ValueError("Unsupported nominal budget settings")
    if "nominal_components" in settings:
        raise ValueError("nominal_components moved to config/scene_taxonomy.yaml (selection.n_components)")
    # Deliberately not a configuration hash: K-independent stages must stay
    # current when the reviewed K changes; K-dependent outputs record the gate.
    settings["scene_taxonomy_file"] = str(TAXONOMY_PATH)
    settings["channel_scenarios_file"] = config["eco_channel_scenarios_file"]
    settings["abi_input_dir"] = config.get("paths", {}).get("preprocessed", "data/preprocessing/abi")
    taxonomy = load_scene_taxonomy(TAXONOMY_PATH)
    settings["nominal_components"] = taxonomy["selection"]["n_components"]
    sensitivity_path = config.get("sensitivity_settings", ROOT/"config/sensitivity_run.yaml")
    settings = resolve_components(settings, taxonomy, load_settings(sensitivity_path))
    training, validation = settings["training_days"], settings["validation_days"]
    if (not training or len(validation) < 2 or set(training) & set(validation)
            or len(set(training)) != len(training) or len(set(validation)) != len(validation)):
        raise ValueError("Use unique disjoint training days and at least two validation days")
    if settings.get("abi_mode", "full") != "full":
        raise ValueError("Only abi_mode: full is supported; the sampled tile-cache mode is retired")
    from nominal_streaming import validate_settings
    validate_settings(settings, config)
    return settings, config


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as target:
        json.dump(value, target, indent=2, allow_nan=False)


def provenance(inputs, stage=None, identity=None, settings=None):
    if stage is not None:
        if settings is None:
            settings, _ = options()
        selected_identity = nominal_configuration_identity(stage, settings, identity)
        return build_provenance(
            root=ROOT,
            code_paths=nominal_provenance_code_paths(stage, ROOT),
            configuration_paths=(),
            input_paths={str(i): Path(path) for i, path in enumerate(inputs)},
            stage=stage,
            configuration_identity=selected_identity,
            scoped_code_paths=nominal_scoped_code_paths(stage, ROOT),
        )
    return build_provenance(
        root=ROOT,
        code_paths=[Path(__file__), ROOT/"src/nominal_products.py", ROOT/"src/nominal_budget.py",
                    ROOT/"src/geometry_sensitivity.py",
                    ROOT/"src/spatial_averaging.py", ROOT/"src/workflow_config.py",
                    ROOT/"src/product_paths.py", ROOT/"src/broadband.py",
                    ROOT/"src/nominal_streaming.py", ROOT/"src/streaming_gmm.py",
                    ROOT/"src/gmm_scheduler.py",
                    ROOT/"src/gmm_stability.py",
                    *[ROOT/f"src/{name}.py" for name in (
                        "abi_sunny_comparison", "eco_spectral_response", "sensitivity",
                        "scene_features", "scene_taxonomy", "radiometry", "uncertainty", "adm", "adm_fitting",
                        "spectral_response",
                    )]],
        configuration_paths=[*settings_paths(SETTINGS), CONFIG_PATH, ROOT/"config/eco_channel_scenarios.yaml"],
        input_paths={str(i): Path(path) for i, path in enumerate(inputs)},
    )


def provenance_identity(stage, identity=None, settings=None):
    """Return the exact identity expected by a stage freshness check."""
    if settings is None:
        settings, _ = options()
    return nominal_configuration_identity(stage, settings, identity)


def check_stage_provenance(metadata, stage, identity=None, settings=None):
    """Check a provenance object or enclosing stage record against current settings."""
    expected_identity = provenance_identity(stage, identity, settings)
    if isinstance(metadata, dict) and "provenance" in metadata:
        metadata = metadata["provenance"]
    return check_provenance(
        metadata, ROOT, expected_stage=stage,
        expected_configuration_identity=expected_identity,
    )


def verify(path, stage=None, identity=None, settings=None):
    metadata = json.loads(Path(str(path)+".json").read_text())
    if stage is None:
        freshness = check_provenance(metadata["provenance"], ROOT)
    else:
        freshness = check_stage_provenance(
            metadata["provenance"], stage, identity=identity, settings=settings
        )
    if freshness["status"] != "current":
        raise ValueError(f"Stale nominal budget input {path}: {freshness}")


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
    # Incomplete footprints are discarded; avoid averaging their masked NaNs.
    coarse_raw = block_average_chunk(np.where(native[..., None, None], raw_radiance, 0), block)[complete]
    coarse_rad = block_average_chunk(np.where(native[..., None, None], radiance, 0), block)[complete]
    coarse_angles = block_average_chunk(np.where(native[..., None], angles, 0), block)[complete]
    coarse_labels = model.predict(abi_noise_features(coarse_raw, planck, 0))
    retrieved = abi_broadband(coarse_rad, coarse_angles, planck, coarse_labels,
                             library, "regularized", coefficients).mean(axis=1)
    return retrieved-reference


def abi_noise_features(radiance, planck, noise):
    """Recompute mean-satellite spectral features from ABI radiances[n,2,6]."""
    noisy = np.asarray(radiance)+noise
    if np.any(~np.isfinite(noisy)) or np.any(noisy <= 0):
        raise ValueError("Added ABI noise produced non-positive or invalid radiance")
    bt = np.stack([synthetic_brightness_temperature(noisy[:, satellite], planck[satellite])
                   for satellite in range(2)], axis=1)
    return spectral_features_from_bt(bt.mean(axis=1))


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
    from nominal_streaming import cache_path, open_features
    index, _ = open_features(settings, settings["training_days"][0])
    planck = np.asarray(index["planck"])[0]
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
                                  cache_path(settings, settings["training_days"][0]) / "index.json"]),
        "reference_domain_um": settings["reference_domain_um"], "roundtrip_error_K": roundtrip,
        "classifier_noise": "Measured separately on held-out ABI; no noisy Sunny classification",
    })


def density(model, radiance, planck):
    bt = synthetic_brightness_temperature(radiance.reshape(-1, 6), planck)
    transformed = model[1].transform(model[0].transform(spectral_features_from_bt(bt)))
    return model[-1].score_samples(transformed).reshape(radiance.shape[:2])


def calculate(settings):
    directory = Path(settings["output_dir"])
    model_directory = nominal_model_directory(settings)
    model_directory.mkdir(parents=True, exist_ok=True)
    model_path = Path(settings.get(
        "model_path", directory/"spectral_gmm.joblib"
    ))
    data_path = directory/"sunny.npz"
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
    heldout_clean_flux = np.empty((n, 6), dtype=float)
    heldout_clean_broadband = np.empty(n, dtype=float)
    heldout_fold_ids = np.full(n, -1, dtype=np.int64)
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
        heldout_clean_flux[test] = clean_flux
        heldout_clean_broadband[test] = clean
        heldout_fold_ids[test] = fold
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
    if np.any(heldout_fold_ids < 0):
        raise RuntimeError("A Sunny case was not assigned to a held-out fold")
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
    product_dir = Path(settings.get("product_dir") or (directory/"products"))
    products = write_heldout_products(
        product_dir,
        clean_flux=heldout_clean_flux,
        clean_broadband=heldout_clean_broadband,
        reference=data["reference"],
        weights=weights,
        labels=labels,
        groups=data["groups"],
        regimes=data["regimes"],
        fold_ids=heldout_fold_ids,
        scenario=settings["scenario"],
        true_band_flux=data["true_band_flux"],
        angles_deg=data["angles"],
        reference_domain_um=settings["reference_domain_um"],
        channel_definition=load_channel_scenarios(
            settings.get("channel_scenarios_file", ROOT/"config/eco_channel_scenarios.yaml")
        )[settings["scenario"]],
        provenance=provenance([
            model_path, str(model_path)+".json", data_path, str(data_path)+".json",
            assignment_path, spatial_path,
        ]),
    )
    residual_path = directory/"residuals.npz"
    np.savez_compressed(residual_path, **components, weights=weights, labels=labels, noisy_labels=noisy_labels,
                        groups=data["groups"], regimes=data["regimes"], per_file_delta=per_file_delta)
    joblib.dump(models, model_directory/"cv_models.joblib")
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
    }, model_directory/"nominal_models.joblib")
    report = {
        "schema_version": 1, "settings": settings, "rows": rows, "support": support,
        "covariances_w_m4": covariances, "provenance": provenance([model_path, str(model_path)+".json",
                                                                  data_path, str(data_path)+".json",
                                                                  assignment_path, spatial_path]),
        "spatial": spatial_report,
        "products": {
            "directory": str(product_dir),
            "clean_heldout_flux": products,
            "scope": (
                "Noise-free, simulation/evaluation-only Sunny spectra evaluated with "
                "GroupKFold-held-out ECO scene-library and N2BC fits; not flight/"
                "geolocated data or all-training predictions."
            ),
        },
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
            "The reviewed component count is shared; diagnostic likelihood recommendations do not override it.",
            "ABI mean-view BT versus individual Sunny-view transfer can change labels with angle.",
            "Scene-frequency weights do not correct within-scene spectral coverage or simulation realism.",
            "Zero random entries for deterministic sources exclude training/library uncertainty, not all possible randomness.",
        ],
    }
    write(directory/"budget_report.json", report)
    write(str(model_directory/"nominal_models.joblib")+".json", {
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
    global SETTINGS, CONFIG_PATH, TAXONOMY_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=SETTINGS)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--taxonomy", type=Path, default=TAXONOMY_PATH)
    parser.add_argument("--day", type=int)
    parser.add_argument("--components", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initialization", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--block", type=int)
    parser.add_argument("--form", choices=("regularized", "quadratic"), default="regularized",
                        help="ADM angular form for the adm stage")
    parser.add_argument(
        "stage",
        choices=("cache", "candidate", "candidate-init", "candidate-sweep", "candidate-combine", "train",
                 "prepare", "assignment", "diagnostics", "spatial", "calculate",
                 "preprocessing", "score", "adm", "assignment-day", "assignment-combine",
                 "spatial-part", "spatial-combine", "stability-sample", "diagnostics-component",
                 "diagnostics-combine", "candidates"),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    SETTINGS, CONFIG_PATH = args.settings.resolve(), args.config.resolve()
    TAXONOMY_PATH = args.taxonomy.resolve()
    settings, config = options()
    if args.stage == "prepare":
        prepare(settings, config)
    elif args.stage == "calculate":
        calculate(settings)
    else:
        import nominal_streaming
        nominal_streaming.run(args.stage, settings, config, sys.modules[__name__],
                              args.day, args.components, args.seed, args.initialization,
                              workers=args.workers, block=args.block, form=args.form)

if __name__ == "__main__":
    main()
