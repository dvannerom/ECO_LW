"""Execute isolated, bounded ABI proxy and ECO/Sunny sensitivity stages."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import yaml
from netCDF4 import Dataset
from scipy.integrate import simpson
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from broadband import load_cubic_coefficients
from eco_spectral_response import channel_response_matrix, load_channel_scenarios
from radiometry import radiance_to_brightness_temperature
from sensitivity import (
    abi_broadband, angular_basis, block_mean, fit_gmm, fit_scene_adm, paired_summary,
    planck_derivative, read_settings, retrieve_sunny, summary, switch_labels,
    tile_features,
)
from uncertainty import build_provenance, check_provenance

CHANNELS = (0, 3, 4, 6, 7, 8)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as target:
        json.dump(value, target, indent=2, allow_nan=False)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def provenance(args, inputs):
    return build_provenance(
        root=ROOT,
        code_paths=[
            Path(__file__).resolve(), ROOT / "src/sensitivity.py",
            ROOT / "src/scene_features.py", ROOT / "src/adm.py",
            ROOT / "src/adm_fitting.py", ROOT / "src/broadband.py",
            ROOT / "src/radiometry.py", ROOT / "src/eco_spectral_response.py",
            ROOT / "src/uncertainty.py",
        ],
        configuration_paths=[
            args.settings, args.config, ROOT / "config/error_budget.yaml",
            ROOT / "config/uncertainty_experiments.yaml",
            ROOT / "config/eco_channel_scenarios.yaml",
        ],
        input_paths={str(index): Path(path) for index, path in enumerate(inputs)},
    )


def stage_metadata(args, inputs, **details):
    write_json(str(args.output) + ".json", {
        "provenance": provenance(args, inputs), **details,
    })


def load_verified(path):
    with Path(str(path) + ".json").open() as source:
        metadata = json.load(source)
    freshness = check_provenance(metadata["provenance"], ROOT)
    if freshness["status"] != "current":
        raise ValueError(f"Stale sensitivity input {path}: {freshness['mismatches']}")
    return metadata


def prepare_abi(args, settings, config):
    options = settings["abi"]
    size, halo, block = options["tile_size"], options["halo"], options["block_size"]
    source_path = Path(args.inputs[0])
    records = {key: [] for key in ("features", "coarse_features", "radiance",
                                  "coarse_radiance", "angles", "coarse_angles",
                                  "valid_blocks", "origins")}
    with Dataset(source_path) as source:
        height, width = source.variables["lat_interp_grid"].shape
        planck = np.stack([
            np.asarray(source.variables[f"planck_{satellite}"][:])[list(CHANNELS)]
            for satellite in ("G16", "G18")
        ])
        candidates = [
            (y, x)
            for y in range(((halo + size - 1) // size) * size, height - size - halo, size)
            for x in range(((halo + size - 1) // size) * size, width - size - halo, size)
        ]
        rng = np.random.default_rng(settings["seed"])
        order = rng.permutation(len(candidates))[:options["max_candidate_tiles"]]

        def read(name, y, x, channels=False):
            variable = source.variables[name]
            selection = (slice(y - halo, y + size + halo),
                         slice(x - halo, x + size + halo))
            if channels:
                selection += (list(CHANNELS),)
            return np.asarray(np.ma.filled(variable[selection], np.nan), dtype=np.float32)

        for candidate in order:
            y, x = candidates[candidate]
            angles = np.stack([read(f"lza_{satellite}_interp_corr", y, x)
                               for satellite in ("G16", "G18")], axis=-1)
            central_angles = angles[halo:-halo, halo:-halo]
            angular_valid = np.all(
                np.isfinite(central_angles) & (central_angles >= 0)
                & (central_angles <= options["maximum_vza_deg"]), axis=-1
            )
            if angular_valid.mean() < options["minimum_valid_fraction"]:
                continue
            bt = np.stack([read(f"BT_{satellite}_interp", y, x, True)
                           for satellite in ("G16", "G18")], axis=2)
            radiance = np.stack([read(f"rad_{satellite}_interp_corr", y, x, True)
                                 for satellite in ("G16", "G18")], axis=2)
            uncorrected = np.stack([read(f"rad_{satellite}_interp", y, x, True)
                                   for satellite in ("G16", "G18")], axis=2)
            features, feature_valid = tile_features(bt)
            coarse_uncorrected = block_mean(uncorrected, block)
            coarse_bt = np.empty_like(coarse_uncorrected)
            for satellite in range(2):
                for channel in range(6):
                    values = coarse_uncorrected[:, :, satellite, channel]
                    coarse_bt[:, :, satellite, channel] = radiance_to_brightness_temperature(
                        np.where(values > 0, values, np.nan), planck[satellite, channel]
                    )
            coarse_features, coarse_valid = tile_features(coarse_bt)
            central = (slice(halo, -halo), slice(halo, -halo))
            coarse_halo = halo // block
            coarse_central = (slice(coarse_halo, -coarse_halo),) * 2
            native_valid = (
                feature_valid[central] & angular_valid
                & np.all(np.isfinite(radiance[central]) & (radiance[central] > 0)
                         & (radiance[central] < 1000), axis=(2, 3))
            )
            valid_blocks = (block_mean(native_valid.astype(float), block) == 1)
            valid_blocks &= coarse_valid[coarse_central]
            if valid_blocks.mean() < options["minimum_valid_fraction"]:
                continue
            records["features"].append(features[central])
            records["coarse_features"].append(coarse_features[coarse_central])
            records["radiance"].append(radiance[central])
            records["coarse_radiance"].append(block_mean(radiance, block)[coarse_central])
            records["angles"].append(central_angles)
            records["coarse_angles"].append(block_mean(angles, block)[coarse_central])
            records["valid_blocks"].append(valid_blocks)
            records["origins"].append((y, x))
            print(f"{source_path.name}: accepted tile {len(records['origins'])} at {(y, x)}",
                  flush=True)
            if len(records["origins"]) == options["tiles_per_day"]:
                break
    if len(records["origins"]) != options["tiles_per_day"]:
        raise ValueError(f"Insufficient eligible ABI tiles in {source_path}")
    save_npz(args.output, **{key: np.asarray(value) for key, value in records.items()},
             planck=planck)
    stage_metadata(
        args, [source_path], day=args.day, eligible_blocks=int(np.sum(records["valid_blocks"])),
        scope=settings["scope"], selection="Seeded random non-overlapping tiles; eligibility-screened",
    )


def abi_records(paths, block):
    """Concatenate only cached bounded tiles, not full disk arrays."""
    records = {key: [] for key in ("features", "radiance", "angles")}
    for path in paths:
        load_verified(path)
        with np.load(path) as data:
            valid = np.repeat(np.repeat(data["valid_blocks"], block, axis=1), block, axis=2)
            for key in records:
                records[key].append(data[key][valid])
    return {key: np.concatenate(value) for key, value in records.items()}


def train_abi_gmm(args, settings):
    records = abi_records(args.inputs, settings["abi"]["block_size"])
    features = records["features"]
    count = min(settings["abi"]["training_points"], len(features))
    selected = np.random.default_rng(settings["seed"]).choice(len(features), count, replace=False)
    model = fit_gmm(features[selected], args.components, settings["seed"],
                    settings["abi"]["gmm_initializations"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    stage_metadata(args, args.inputs, training_records=count, components=args.components)


def train_abi_adm(args, settings):
    load_verified(args.model)
    model = joblib.load(args.model)
    records = abi_records(args.inputs, settings["abi"]["block_size"])
    labels = model.predict(records["features"])
    library = []
    counts = []
    for component in range(args.components):
        selected = labels == component
        counts.append(int(selected.sum()))
        library.append([
            fit_scene_adm(records["angles"][selected],
                          records["radiance"][selected, :, channel], args.form)
            for channel in range(6)
        ])
    save_npz(args.output, library=np.asarray(library))
    profile = 1 + np.einsum(
        "ap,scp->sca", angular_basis(np.linspace(0, 90, 901), args.form), np.asarray(library)
    )
    stage_metadata(args, [*args.inputs, args.model], form=args.form, scene_counts=counts,
                   positivity_boundary_scene_channels=int(np.sum(profile.min(axis=-1) < 1e-4)),
                   normalization="Positive 0-90 degree profiles; 0.1-degree constraint grid")


def run_abi(args, settings, config):
    allowed = {"baseline", "spatial", "assignment", "adm"} | {
        f"components_{count}" for count in settings["abi"]["components"]
    }
    if args.variant not in allowed:
        raise ValueError(f"Unsupported ABI sensitivity variant {args.variant}")
    load_verified(args.model)
    load_verified(args.library)
    model = joblib.load(args.model)
    with np.load(args.library) as data:
        library = data["library"]
    block = settings["abi"]["block_size"]
    coefficients = load_cubic_coefficients(config["narrowband_to_broadband_coeffs_file"])
    outputs, ids = [], []
    selected_total = eligible_total = 0
    for file_index, path in enumerate(args.inputs):
        metadata = load_verified(path)
        with np.load(path) as data:
            coarse = args.variant == "spatial"
            prefix = "coarse_" if coarse else ""
            valid_blocks = data["valid_blocks"]
            valid = valid_blocks if coarse else np.repeat(
                np.repeat(valid_blocks, block, axis=1), block, axis=2
            )
            features = data[prefix + "features"][valid]
            probabilities = model.predict_proba(features)
            fraction = settings["abi"]["second_choice_fraction"] if args.variant == "assignment" else 0
            labels, selected = switch_labels(
                probabilities, fraction,
                np.random.SeedSequence([settings["seed"], args.realization, file_index]),
            )
            selected_total += selected.size
            eligible_total += labels.size
            flux = abi_broadband(
                data[prefix + "radiance"][valid], data[prefix + "angles"][valid],
                data["planck"], labels, library, args.form, coefficients,
            )
            if not np.all(np.isfinite(flux)) or np.any((flux < 50) | (flux > 1000)):
                raise ValueError("ABI variant exceeds production flux validity range; inspect rather than drop pairs")
            if coarse:
                coarse_flux = flux
            else:
                field = np.zeros((*valid.shape, 2))
                field[valid] = flux
                coarse_flux = np.stack([block_mean(tile, block) for tile in field])[valid_blocks]
            outputs.append(coarse_flux)
            for tile, row, column in np.argwhere(valid_blocks):
                origin = data["origins"][tile]
                ids.append(f"{metadata['day']}:{origin[0] + row * block}:{origin[1] + column * block}")
    save_npz(args.output, flux=np.concatenate(outputs), record_ids=np.asarray(ids))
    stage_metadata(
        args, [*args.inputs, args.model, args.library,
               config["narrowband_to_broadband_coeffs_file"]],
        variant=args.variant, form=args.form, realization=args.realization,
        selected_pixels=selected_total, eligible_pixels=eligible_total,
        aggregation="Equal-area 5x5 complete native footprints; satellites retained separately",
    )


def prepare_sunny(args, settings, config):
    with Path(config["eco_channel_scenarios_file"]).open() as source:
        catalog = yaml.safe_load(source)
    scenario = load_channel_scenarios(config["eco_channel_scenarios_file"])[settings["sunny"]["scenario"]]
    angles = np.asarray(catalog["retrieval_geometry"]["available_view_angles_deg"])
    requested = np.asarray(catalog["retrieval_geometry"]["retrieval_view_angles_deg"])
    indices = np.array([np.flatnonzero(angles == angle).item() for angle in requested])
    if np.any(requested > 70) or len(np.unique(requested)) != requested.size:
        raise ValueError("Invalid ECO retrieval geometry")
    with (ROOT / "config/error_budget.yaml").open() as source:
        budget = yaml.safe_load(source)
    noise_definition = budget["requirements"]["tir_radiometric_sensitivity"]
    if scenario["channel_names"] != list(noise_definition["channels"]):
        raise ValueError("NEdT channel order differs from selected ECO channels")
    nedt = np.array([value[settings["sunny"]["noise_requirement"]]
                     for value in noise_definition["channels"].values()])
    radiances, true_fluxes, references, groups, regimes, sigmas = [], [], [], [], [], []
    files = []
    for regime, directory in (("clear_sky", "radiance_lw_cs"), ("cloudy", "radiance_lw_cl")):
        for path in sorted((Path(config["sunny_dir"]) / directory).glob("sunny_lw_*")):
            data = np.loadtxt(path, usecols=range(2 + angles.size))
            if np.any(~np.isfinite(data)) or np.any(data[:, 1:] < 0) or np.any(np.diff(data[:, 0]) <= 0):
                raise ValueError(f"Invalid Sunny spectrum {path}")
            wavelength = data[:, 0]
            response = channel_response_matrix(wavelength, scenario, catalog["default_edge_slope_per_um"])
            radiances.append(simpson(
                data[:, 2 + indices, None] * response[:, None, :], x=wavelength, axis=0
            ))
            true_fluxes.append(simpson(data[:, 1, None] * response, x=wavelength, axis=0))
            references.append(simpson(data[:, 1], x=wavelength))
            sigmas.append(simpson(
                planck_derivative(wavelength, noise_definition["reference_temperature_k"])[:, None]
                * response, x=wavelength, axis=0
            ) * nedt)
            groups.append(path.name.rsplit("_", 1)[-1])
            regimes.append(regime)
            files.append(path)
    groups = np.asarray(groups)
    unique, counts = np.unique(groups, return_counts=True)
    if not len(files) or np.any(counts != 2):
        raise ValueError("Sunny library needs paired clear/cloud indices")
    regime_array = np.asarray(regimes)
    for group in unique:
        if set(regime_array[groups == group]) != {"clear_sky", "cloudy"}:
            raise ValueError(f"Invalid clear/cloud grouping {group}")
    save_npz(args.output, radiance=np.asarray(radiances), true_flux=np.asarray(true_fluxes),
             reference=np.asarray(references), noise_sd=np.asarray(sigmas),
             groups=groups, regimes=regime_array, angles=requested)
    stage_metadata(args, files, scene_count=len(files), group_count=len(unique),
                   scenario=settings["sunny"]["scenario"], geometry=requested.tolist(),
                   reference_domain_um=[float(wavelength[0]), float(wavelength[-1])],
                   noise_nedt_k=nedt.tolist(), noise_correlation="assumed independent views/channels")


def fit_sunny_models(args, settings):
    load_verified(args.inputs[0])
    with np.load(args.inputs[0]) as data:
        flux, reference, groups, regimes = (
            data[key] for key in ("true_flux", "reference", "groups", "regimes")
        )
    models = []
    options = settings["sunny"]
    for fold, (train, test) in enumerate(
        GroupKFold(options["folds"]).split(flux, reference, groups), start=1
    ):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Sunny fold leakage")
        fitted = {}
        for regime in np.unique(regimes):
            indices = train[regimes[train] == regime]
            scaler = StandardScaler().fit(flux[indices])
            poly = PolynomialFeatures(options["degree"], include_bias=False)
            design = poly.fit_transform(scaler.transform(flux[indices]))
            target = reference[indices]
            model = LinearRegression().fit(design, target)
            if args.variant == "robust":
                for iteration in range(options["robust_max_iterations"]):
                    previous = model.predict(design)
                    residual = target - previous
                    scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-8)
                    weight = np.minimum(1, options["robust_huber_cutoff"] * scale
                                        / np.maximum(np.abs(residual), 1e-12))
                    model = LinearRegression().fit(design, target, sample_weight=weight)
                    if np.max(np.abs(model.predict(design) - previous)) < options["robust_tolerance"]:
                        break
                else:
                    raise RuntimeError(f"Robust N2BC did not converge in fold {fold}/{regime}")
            elif args.variant != "baseline":
                raise ValueError("Unsupported Sunny training variant")
            fitted[str(regime)] = (scaler, poly, model)
        models.append({"fold": fold, "test": test, "models": fitted})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, args.output)
    stage_metadata(args, args.inputs, variant=args.variant, folds=options["folds"],
                   degree=options["degree"], grouping="shared clear/cloud scene index")


def sunny_predict(flux, regimes, models):
    prediction = np.full(len(flux), np.nan)
    for fold in models:
        for regime, (scaler, poly, model) in fold["models"].items():
            selected = fold["test"][regimes[fold["test"]] == regime]
            prediction[selected] = model.predict(poly.transform(scaler.transform(flux[selected])))
    if np.any(~np.isfinite(prediction)):
        raise RuntimeError("Sunny CV did not predict every record")
    return prediction


def run_sunny(args, settings):
    if args.variant not in {"baseline", "spectral", "adm", "robust", "noise", "joint"}:
        raise ValueError(f"Unsupported ECO/Sunny sensitivity variant {args.variant}")
    load_verified(args.inputs[0])
    load_verified(args.model)
    models = joblib.load(args.model)
    diagnostics = {}
    with np.load(args.inputs[0]) as data:
        radiance = data["radiance"]
        if args.variant in ("noise", "joint"):
            rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], args.realization]))
            radiance = radiance + rng.normal(size=radiance.shape) * data["noise_sd"][:, None, :]
        flux = data["true_flux"] if args.variant == "spectral" else retrieve_sunny(
            radiance, data["angles"], args.form, diagnostics
        )
        prediction = sunny_predict(flux, data["regimes"], models)
        if np.any(prediction <= 0):
            raise ValueError("Sunny N2BC produced non-positive flux")
        ids = np.char.add(np.char.add(data["groups"], ":"), data["regimes"])
        save_npz(args.output, flux=prediction, reference=data["reference"], record_ids=ids,
                 regimes=data["regimes"])
    stage_metadata(args, [*args.inputs, args.model], variant=args.variant,
                   form=args.form, realization=args.realization, **diagnostics)


def compare(args):
    if len(args.inputs) != 2:
        raise ValueError("Comparison needs exactly one variant and one baseline")
    variant_metadata = load_verified(args.inputs[0])
    load_verified(args.inputs[1])
    with np.load(args.inputs[0]) as variant, np.load(args.inputs[1]) as baseline:
        if not np.array_equal(variant["record_ids"], baseline["record_ids"]):
            raise ValueError("Sensitivity records are not paired in the same order")
        reference = baseline["reference"] if "reference" in baseline else None
        result = paired_summary(variant["flux"], baseline["flux"], reference)
        result["variant_metadata"] = {key: value for key, value in variant_metadata.items()
                                      if key != "provenance"}
        result["provenance"] = provenance(args, args.inputs)
        if reference is not None:
            result["truth_error"] = {
                "w_m2": summary(variant["flux"] - reference),
                "percent": summary((variant["flux"] - reference) / reference * 100),
            }
    write_json(args.output, result)


def assemble(args, settings):
    directory = Path(settings["output_dir"])
    results = {}
    for path in args.inputs:
        with Path(path).open() as source:
            record = json.load(source)
        freshness = check_provenance(record["provenance"], ROOT)
        if freshness["status"] != "current":
            raise ValueError(f"Stale comparison {path}")
        results[Path(path).stem] = record
    with np.load(directory / "sunny/outputs/baseline_0.npz") as baseline, np.load(
        directory / "sunny/outputs/spectral_0.npz"
    ) as spectral:
        reference = baseline["reference"]
        spectral_error = (spectral["flux"] - reference) / reference * 100
        angular_increment = (baseline["flux"] - spectral["flux"]) / reference * 100
        covariance = np.cov(np.stack((spectral_error, angular_increment)), ddof=1)
        total_error = (baseline["flux"] - reference) / reference * 100
        if not np.isclose(covariance.sum(), np.var(total_error, ddof=1)):
            raise RuntimeError("Paired covariance closure failed")
    report = {
        "schema_version": 1, "scope": settings["scope"], "settings": settings,
        "results": results,
        "covariance_percent_squared": covariance.tolist(),
        "baseline_error_percent": summary(total_error),
        "blocked_terms": {
            "scene_identification_ECO": "ABI texture-GMM cannot be applied to ECO Sunny scenes without transfer.",
            "library_representativity": "No validated ECO population weights available.",
            "library_physical_realism": "No independent truth or validated metadata screening criteria.",
            "mission_total": "ABI proxy sensitivities cannot be added to Sunny ECO residuals.",
        },
        "provenance": provenance(args, [*args.inputs, *args.paired_fluxes]),
    }
    fluxes = {}
    for path in args.paired_fluxes:
        load_verified(path)
        with np.load(path) as data:
            fluxes[Path(path).stem] = {
                key: data[key] for key in ("flux", "record_ids", "reference")
            }
    baseline = fluxes["baseline_0"]
    for record in fluxes.values():
        if (not np.array_equal(record["record_ids"], baseline["record_ids"])
                or not np.array_equal(record["reference"], baseline["reference"])):
            raise ValueError("Joint interaction inputs do not share the same cases/truth")
    report["joint_interactions"] = []
    for realization in range(settings["realizations"]):
        interaction = (
            fluxes[f"joint_{realization}"]["flux"]
            - fluxes[f"noise_{realization}"]["flux"]
            - fluxes["adm_0"]["flux"] - fluxes["robust_0"]["flux"]
            + 2 * baseline["flux"]
        )
        report["joint_interactions"].append({
            "realization": realization, "w_m2": summary(interaction),
            "percent": summary(interaction / baseline["reference"] * 100),
        })
    write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(
        "prepare-abi", "train-gmm", "fit-abi-adm", "run-abi",
        "prepare-sunny", "fit-sunny", "run-sunny", "compare", "assemble",
    ))
    parser.add_argument("--settings", type=Path, default=ROOT / "config/sensitivity_run.yaml")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--inputs", nargs="+", default=[])
    parser.add_argument("--paired-fluxes", nargs="+", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--day", type=int)
    parser.add_argument("--components", type=int, default=7)
    parser.add_argument("--form", choices=("regularized", "quadratic"), default="regularized")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--realization", type=int, default=0)
    args = parser.parse_args()
    settings = read_settings(args.settings)
    with args.config.open() as source:
        config = yaml.safe_load(source)
    if settings["sunny"]["training_mode"] != "clear_cloud_stratified":
        raise ValueError("First draft supports clear/cloud-stratified N2BC only")
    if config["resolution_km"] != 2 or config["preprocess_step"] != 1:
        raise ValueError("First-draft ABI assessment requires native 2 km step1 inputs")
    stages = {
        "prepare-abi": lambda: prepare_abi(args, settings, config),
        "train-gmm": lambda: train_abi_gmm(args, settings),
        "fit-abi-adm": lambda: train_abi_adm(args, settings),
        "run-abi": lambda: run_abi(args, settings, config),
        "prepare-sunny": lambda: prepare_sunny(args, settings, config),
        "fit-sunny": lambda: fit_sunny_models(args, settings),
        "run-sunny": lambda: run_sunny(args, settings),
        "compare": lambda: compare(args),
        "assemble": lambda: assemble(args, settings),
    }
    stages[args.stage]()
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
