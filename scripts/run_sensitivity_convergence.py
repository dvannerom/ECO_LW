"""Isolated ABI convergence ladders and regime-resolved paired diagnostics."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import yaml

from broadband import load_cubic_coefficients
from run_uncertainty_sensitivity import prepare_abi
from sensitivity import (
    abi_broadband, block_mean, fit_gmm, fit_scene_adm, read_settings,
    summary, switch_labels,
)
from sensitivity_diagnostics import (
    cluster_interval, dominant_scene, fit_diagnostics, stability_check,
)
from uncertainty import build_provenance, check_provenance


def configuration(path):
    with Path(path).open() as source:
        study = yaml.safe_load(source)
    if study["schema_version"] != 1:
        raise ValueError("Unsupported convergence schema")
    base = read_settings(study["base_settings"])
    training, evaluation = study["training_days"], study["evaluation_days"]
    if (not training or len(set(evaluation)) < 2 or set(training) & set(evaluation)
            or len(set(training)) != len(training) or len(set(evaluation)) != len(evaluation)):
        raise ValueError("Use unique, disjoint training/evaluation days and multiple held-out days")
    for key in ("tile_counts", "gmm_point_counts"):
        values = study[key]
        if len(values) < 2 or sorted(set(values)) != values or min(values) < 1:
            raise ValueError(f"{key} must be a strictly increasing positive ladder")
    for key in ("sampling_seeds", "initialization_seeds"):
        if len(set(study[key])) != len(study[key]) or len(study[key]) < 2:
            raise ValueError(f"{key} needs multiple distinct seeds")
    for key in ("fixed_gmm_points", "adm_training_points", "gmm_initializations",
                "bootstrap_replicates", "max_candidate_tiles", "minimum_regime_footprints"):
        if not isinstance(study[key], int) or study[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if study["adm_training_points"] < max(study["gmm_point_counts"] + [study["fixed_gmm_points"]]):
        raise ValueError("Training pool must contain the largest GMM point sample")
    if not 0 < study["confidence_level"] < 1:
        raise ValueError("Confidence level must be in (0,1)")
    if min(study["stability_relative_tolerance"],
           study["stability_absolute_tolerance_w_m2"]) < 0:
        raise ValueError("Stability tolerances cannot be negative")
    base.update(training_days=training, evaluation_days=evaluation, scope=study["scope"])
    base["abi"].update(tiles_per_day=max(study["tile_counts"]),
                       max_candidate_tiles=study["max_candidate_tiles"])
    return study, base


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as target:
        json.dump(value, target, indent=2, allow_nan=False)


def provenance(args, inputs):
    study, _ = configuration(args.study)
    return build_provenance(
        root=ROOT,
        code_paths=[
            Path(__file__).resolve(), ROOT / "scripts/run_uncertainty_sensitivity.py",
            *[ROOT / f"src/{name}.py" for name in (
                "sensitivity", "sensitivity_diagnostics", "uncertainty", "scene_features",
                "adm", "adm_fitting", "radiometry", "broadband",
            )],
        ],
        configuration_paths=[args.study, Path(study["base_settings"]), args.config],
        input_paths={str(index): Path(path) for index, path in enumerate(inputs)},
    )


def verify(metadata):
    check = check_provenance(metadata["provenance"], ROOT)
    if check["status"] != "current":
        raise ValueError(f"Stale convergence input: {check['mismatches']}")


def load_json(path):
    with Path(path).open() as source:
        return json.load(source)


def open_cache(path):
    path = Path(path)
    index = load_json(path / "index.json")
    verify(index)
    arrays = {name: np.load(path / f"{name}.npy", mmap_mode="r")
              for name in index["arrays"]}
    return index, arrays


def prepare(args, study, base, config):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base["seed"] = args.sampling_seed
    scratch = output / "prepared.npz"
    prepare_abi(
        SimpleNamespace(
            inputs=args.inputs, output=scratch, day=args.day,
            settings=args.study, config=args.config,
        ), base, config,
    )
    old_metadata = load_json(str(scratch) + ".json")
    with np.load(scratch) as data:
        names = data.files
        for name in names:
            np.save(output / f"{name}.npy", data[name])
    write_json(output / "index.json", {
        "arrays": names, "day": args.day, "sampling_seed": args.sampling_seed,
        "tile_count": base["abi"]["tiles_per_day"],
        "eligible_blocks": old_metadata["eligible_blocks"],
        "provenance": provenance(args, args.inputs),
        "scope": study["scope"],
    })
    scratch.unlink()
    Path(str(scratch) + ".json").unlink()


def collect_training(paths, tiles, capacity, block, seed):
    """Random-priority reservoir bounded by capacity, with consistent ordering."""
    rng = np.random.default_rng(seed)
    selected = {name: [] for name in ("features", "radiance", "angles")}
    keys = np.empty(0)
    seen = 0
    for path in paths:
        index, data = open_cache(path)
        if tiles > index["tile_count"]:
            raise ValueError("Requested training prefix exceeds prepared pool")
        for tile in range(tiles):
            valid = np.repeat(np.repeat(data["valid_blocks"][tile], block, axis=0), block, axis=1)
            count = int(valid.sum())
            new_keys = rng.random(count)
            arrays = {name: data[name][tile][valid] for name in selected}
            keys = np.concatenate((keys, new_keys))
            for name in selected:
                selected[name] = (np.concatenate((selected[name], arrays[name]))
                                  if seen else arrays[name])
            seen += count
            if keys.size > capacity:
                take = np.argpartition(keys, capacity - 1)[:capacity]
                keys = keys[take]
                for name in selected:
                    selected[name] = selected[name][take]
    if len(keys) < capacity:
        raise ValueError(f"Training pool has {len(keys)} records, fewer than requested {capacity}")
    order = np.argsort(keys)
    return {name: value[order] for name, value in selected.items()}


def train(args, study, base):
    records = collect_training(
        args.inputs, args.tiles, study["adm_training_points"],
        base["abi"]["block_size"], study["training_sampling_seed"],
    )
    model = fit_gmm(
        records["features"][:args.points], args.components, args.initialization_seed,
        study["gmm_initializations"],
    )
    labels = model.predict(records["features"])
    libraries, diagnostics = {}, {}
    for form in ("regularized", "quadratic"):
        library, form_diagnostics = [], []
        for component in range(args.components):
            selected = labels == component
            fitted, channels = [], []
            for channel in range(6):
                angles = records["angles"][selected]
                radiance = records["radiance"][selected, :, channel]
                try:
                    parameters = fit_scene_adm(angles, radiance, form)
                except (ValueError, RuntimeError) as error:
                    raise RuntimeError(
                        f"ADM training failed for {args.components} components, "
                        f"scene {component}, channel {channel}, {form}: {error}"
                    ) from error
                fitted.append(parameters)
                channels.append(fit_diagnostics(angles, radiance, parameters, form))
            library.append(fitted)
            form_diagnostics.append(channels)
        libraries[form] = np.asarray(library)
        diagnostics[form] = form_diagnostics
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "libraries": libraries, "diagnostics": diagnostics}, output)
    means = model[0].inverse_transform(model[1].inverse_transform(model[-1].means_))
    write_json(str(output) + ".json", {
        "components": args.components, "training_tiles_per_day": args.tiles,
        "gmm_points": args.points, "adm_points": len(labels),
        "initialization_seed": args.initialization_seed,
        "scene_counts": np.bincount(labels, minlength=args.components).tolist(),
        "scene_mean_features": means.tolist(), "fit_diagnostics": diagnostics,
        "provenance": provenance(args, args.inputs),
    })


def footprint_flux(data, tile, valid, labels, library, form, coefficients, block):
    values = abi_broadband(
        data["radiance"][tile][valid], data["angles"][tile][valid], data["planck"],
        labels, library, form, coefficients,
    )
    if np.any((values < 50) | (values > 1000)):
        raise ValueError("ABI flux outside production validity range; do not silently drop pairs")
    field = np.zeros((*valid.shape, 2))
    field[valid] = values
    return block_mean(field, block)[data["valid_blocks"][tile]]


def evaluate(args, study, base, config):
    models = []
    for path in args.models:
        metadata = load_json(str(path) + ".json")
        verify(metadata)
        models.append(joblib.load(path))
    if len(models) != 2:
        raise ValueError("Pair the baseline 7-component model with the 6-component model")
    baseline_model, alternate_model = models
    components = base["abi"]["baseline_components"]
    if (baseline_model["model"][-1].n_components != components
            or alternate_model["model"][-1].n_components != 6):
        raise ValueError("Convergence comparison requires 7 vs 6 components")
    block = base["abi"]["block_size"]
    coefficients_path = config["narrowband_to_broadband_coeffs_file"]
    coefficients = load_cubic_coefficients(coefficients_path)
    fields = {name: [] for name in (
        "baseline", "spatial", "adm", "components", "assignment", "scene", "scene_fraction",
        "day", "tile_id", "tile_rank", "angles", "heterogeneity",
        "boundary_fraction", "condition", "record_ids",
    )}
    diagnostics = baseline_model["diagnostics"]
    boundary = np.array([
        any(channel["positivity_boundary"] for channel in scene)
        for scene in diagnostics["regularized"]
    ])
    condition = np.array([
        max(channel["normalized_jacobian_condition"] for channel in scene)
        for scene in diagnostics["quadratic"]
    ])
    for path in args.inputs:
        index, data = open_cache(path)
        for tile in range(index["tile_count"]):
            blocks = data["valid_blocks"][tile]
            valid = np.repeat(np.repeat(blocks, block, axis=0), block, axis=1)
            features = data["features"][tile][valid]
            labels = baseline_model["model"].predict(features)
            alternate_labels = alternate_model["model"].predict(features)
            assignment_seed = np.random.SeedSequence(
                [study["training_sampling_seed"], index["day"], tile]
            ).generate_state(1)[0]
            assigned, _ = switch_labels(
                baseline_model["model"].predict_proba(features),
                base["abi"]["second_choice_fraction"], int(assignment_seed),
            )
            baseline = footprint_flux(
                data, tile, valid, labels, baseline_model["libraries"]["regularized"],
                "regularized", coefficients, block,
            )
            adm = footprint_flux(
                data, tile, valid, labels, baseline_model["libraries"]["quadratic"],
                "quadratic", coefficients, block,
            )
            components_flux = footprint_flux(
                data, tile, valid, alternate_labels, alternate_model["libraries"]["regularized"],
                "regularized", coefficients, block,
            )
            assignment = footprint_flux(
                data, tile, valid, assigned, baseline_model["libraries"]["regularized"],
                "regularized", coefficients, block,
            )
            coarse_labels = baseline_model["model"].predict(data["coarse_features"][tile][blocks])
            spatial = abi_broadband(
                data["coarse_radiance"][tile][blocks], data["coarse_angles"][tile][blocks],
                data["planck"], coarse_labels, baseline_model["libraries"]["regularized"],
                "regularized", coefficients,
            )
            if np.any((spatial < 50) | (spatial > 1000)):
                raise ValueError("Coarse flux exceeds production validity range")
            label_field = np.full(valid.shape, -1)
            label_field[valid] = labels
            scene, fraction = dominant_scene(label_field, valid, components, block)
            boundary_field = np.zeros(valid.shape)
            boundary_field[valid] = boundary[labels]
            condition_field = np.zeros(valid.shape)
            condition_field[valid] = condition[labels]
            bt = data["features"][tile, :, :, 3]
            mean = block_mean(bt, block)
            variance = np.maximum(block_mean(bt.astype(float)**2, block) - mean.astype(float)**2, 0)
            count = int(blocks.sum())
            origin = data["origins"][tile]
            tile_id = f"{index['day']}:{origin[0]}:{origin[1]}"
            ids = [f"{index['day']}:{origin[0]+row*block}:{origin[1]+col*block}"
                   for row, col in np.argwhere(blocks)]
            values = {
                "baseline": baseline, "spatial": spatial, "adm": adm,
                "components": components_flux, "scene": scene[blocks],
                "assignment": assignment,
                "scene_fraction": fraction[blocks],
                "day": np.full(count, index["day"]),
                "tile_id": np.full(count, tile_id), "tile_rank": np.full(count, tile),
                "angles": data["coarse_angles"][tile][blocks],
                "heterogeneity": np.sqrt(variance)[blocks],
                "boundary_fraction": block_mean(boundary_field, block)[blocks],
                "condition": block_mean(condition_field, block)[blocks],
                "record_ids": np.asarray(ids),
            }
            for name in fields:
                fields[name].append(values[name])
        print(f"Evaluated held-out day {index['day']}, sampling seed {args.sampling_seed}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.concatenate(value) for key, value in fields.items()}
    if len(np.unique(arrays["record_ids"])) != len(arrays["record_ids"]):
        raise ValueError("Duplicate footprints in evaluation pool")
    np.savez_compressed(output, **arrays)
    write_json(str(output) + ".json", {
        "sampling_seed": args.sampling_seed, "training_tiles": args.tiles,
        "gmm_points": args.points, "initialization_seed": args.initialization_seed,
        "components": components, "scope": study["scope"],
        "provenance": provenance(args, [*args.inputs, *args.models,
                                       *[str(path)+".json" for path in args.models], coefficients_path]),
    })


def binned_summaries(data, error, study):
    bins = {
        "scene": data["scene"],
        "day": data["day"],
        "tile": data["tile_id"],
        "vza_g16": np.digitize(data["angles"][:, 0], [20, 40, 50, 60]),
        "vza_g18": np.digitize(data["angles"][:, 1], [20, 40, 50, 60]),
        "angular_separation": np.digitize(
            np.abs(data["angles"][:, 0] - data["angles"][:, 1]), [5, 10, 20, 30]
        ),
        "heterogeneity_K": np.digitize(data["heterogeneity"], [1, 3, 5, 10]),
        "positivity_boundary": (data["boundary_fraction"] > 0).astype(int),
        "normalized_condition": np.digitize(data["condition"], [10, 30, 100, 300]),
        "scene_purity": np.digitize(data["scene_fraction"], [.5, .75, .9]),
    }
    output = {}
    for name, ids in bins.items():
        rows = []
        for label in np.unique(ids):
            selected = ids == label
            count = int(selected.sum())
            if count < study["minimum_regime_footprints"] and name != "tile":
                rows.append({"label": str(label), "footprints": count,
                             "status": "insufficient_footprints"})
                continue
            row = {"label": str(label), "footprints": count, "status": "quantified",
                   "pooled": summary(error[selected]),
                   "G16": summary(error[selected, 0]), "G18": summary(error[selected, 1]),
                   "share_of_squared_error": float(
                       np.square(error[selected]).sum() / max(np.square(error).sum(), 1e-30)
                   )}
            rows.append(row)
        output[name] = rows
    return output


def summarize(args, study):
    runs = []
    for path in args.inputs:
        metadata = load_json(str(path)+".json")
        verify(metadata)
        with np.load(path) as source:
            data = {key: source[key] for key in source.files}
        sizes = study["tile_counts"] if (
            metadata["training_tiles"] == max(study["tile_counts"])
            and metadata["gmm_points"] == study["fixed_gmm_points"]
            and metadata["initialization_seed"] == study["initialization_seeds"][0]
        ) else [max(study["tile_counts"])]
        for tiles in sizes:
            selected = data["tile_rank"] < tiles
            results = {}
            for variant in ("spatial", "adm", "components", "assignment"):
                error = data[variant][selected] - data["baseline"][selected]
                results[variant] = {
                    "w_m2": cluster_interval(
                        error, data["day"][selected], data["tile_id"][selected],
                        study["bootstrap_replicates"], study["confidence_level"],
                        study["training_sampling_seed"],
                    ),
                    "percent": summary(error / data["baseline"][selected] * 100),
                    "G16": summary(error[:, 0]), "G18": summary(error[:, 1]),
                }
            runs.append({
                "sampling_seed": metadata["sampling_seed"], "evaluation_tiles": tiles,
                "training_tiles": metadata["training_tiles"],
                "gmm_points": metadata["gmm_points"],
                "initialization_seed": metadata["initialization_seed"],
                "footprints": int(selected.sum()), "results": results,
            })
    # Only this fixed-model, largest evaluation pool is used for regime comparison.
    diagnostic_path = next(
        path for path in args.inputs
        if all(load_json(str(path)+".json")[key] == value for key, value in (
            ("sampling_seed", study["sampling_seeds"][0]),
            ("training_tiles", max(study["tile_counts"])),
            ("gmm_points", study["fixed_gmm_points"]),
            ("initialization_seed", study["initialization_seeds"][0]),
        ))
    )
    with np.load(diagnostic_path) as source:
        data = {key: source[key] for key in source.files}
    diagnostics = {
        variant: binned_summaries(data, data[variant] - data["baseline"], study)
        for variant in ("spatial", "adm", "components", "assignment")
    }
    stability = []
    for seed in study["sampling_seeds"]:
        records = sorted([
            run for run in runs if run["sampling_seed"] == seed
            and run["training_tiles"] == max(study["tile_counts"])
            and run["initialization_seed"] == study["initialization_seeds"][0]
            and run["gmm_points"] == study["fixed_gmm_points"]
        ], key=lambda run: run["evaluation_tiles"])
        for earlier, later in zip(records, records[1:]):
            for variant in ("spatial", "adm", "components", "assignment"):
                stability.append({
                    "axis": "evaluation_sampling", "sampling_seed": seed, "variant": variant,
                    "from_tiles": earlier["evaluation_tiles"], "to_tiles": later["evaluation_tiles"],
                    **stability_check(
                        earlier["results"][variant]["w_m2"]["estimate"]["rmse"],
                        later["results"][variant]["w_m2"]["estimate"]["rmse"],
                        study["stability_relative_tolerance"],
                        study["stability_absolute_tolerance_w_m2"],
                    ),
                })
    model_changes = []
    reference_metadata = load_json(str(diagnostic_path) + ".json")
    for path in args.inputs:
        metadata = load_json(str(path) + ".json")
        if metadata["sampling_seed"] != reference_metadata["sampling_seed"]:
            continue
        with np.load(path) as source:
            if not np.array_equal(source["record_ids"], data["record_ids"]):
                raise ValueError("Training comparisons must use exactly the fixed evaluation IDs")
            error = source["baseline"] - data["baseline"]
        model_changes.append({
            "training_tiles": metadata["training_tiles"], "gmm_points": metadata["gmm_points"],
            "initialization_seed": metadata["initialization_seed"],
            "delta_to_fixed_baseline_w_m2": cluster_interval(
                error, data["day"], data["tile_id"], study["bootstrap_replicates"],
                study["confidence_level"], study["training_sampling_seed"],
            ),
        })
    for axis, key in (("training_coverage", "training_tiles"), ("gmm_points", "gmm_points")):
        records = sorted([
            run for run in runs
            if run["sampling_seed"] == study["sampling_seeds"][0]
            and run["evaluation_tiles"] == max(study["tile_counts"])
            and run["initialization_seed"] == study["initialization_seeds"][0]
            and (run["gmm_points"] == study["fixed_gmm_points"] if key == "training_tiles"
                 else run["training_tiles"] == max(study["tile_counts"]))
        ], key=lambda run: run[key])
        for earlier, later in zip(records, records[1:]):
            for variant in ("spatial", "adm", "components", "assignment"):
                stability.append({
                    "axis": axis, "variant": variant,
                    "from": earlier[key], "to": later[key],
                    **stability_check(
                        earlier["results"][variant]["w_m2"]["estimate"]["rmse"],
                        later["results"][variant]["w_m2"]["estimate"]["rmse"],
                        study["stability_relative_tolerance"],
                        study["stability_absolute_tolerance_w_m2"],
                    ),
                })
    write_json(args.output, {
        "schema_version": 1, "settings": study, "runs": runs,
        "regime_diagnostics": diagnostics, "stability_checks": stability,
        "model_changes_on_fixed_evaluation": model_changes,
        "bin_definitions": {
            "vza_g16/vza_g18": "degrees: [0,20),[20,40),[40,50),[50,60),[60,70]",
            "angular_separation": "degrees: [0,5),[5,10),[10,20),[20,30),[30,70]",
            "heterogeneity_K": "native mean-view C14 BT footprint SD: [0,1),[1,3),[3,5),[5,10),[10,infinity)",
            "normalized_condition": "[1,10),[10,30),[30,100),[100,300),[300,infinity)",
            "positivity_boundary": "0: no affected native pixels; 1: any affected native pixels",
            "scene": "Modal native baseline GMM component; no physical truth label; ties use lowest ID",
            "scene_purity": "[0,.5),[.5,.75),[.75,.9),[.9,1]",
        },
        "limitations": [
            "ABI proxy sensitivities, not ECO error allocations or absolute accuracy.",
            "Nested sample sizes share observations; interval overlap is not a convergence test.",
            "Four-day bootstrap intervals are exploratory, conditional on selected days/eligible tiles.",
            "Same geographic tiles can recur across days: temporal holdout, not spatial independence.",
            "Training coverage, GMM point count, initialization and evaluation sampling vary separately.",
            "The two satellite estimates are kept together in tile/day bootstrap resampling.",
            "Scene IDs are local to a fitted GMM; do not equate IDs across runs.",
            "Assignment is a single seeded 10% second-choice stress test, not misclassification probability.",
        ],
        "provenance": provenance(args, [*args.inputs, *[str(path)+".json" for path in args.inputs]]),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate", "summarize"))
    parser.add_argument("--study", type=Path, default=ROOT / "config/sensitivity_convergence.yaml")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--inputs", nargs="+", default=[])
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--day", type=int)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--initialization-seed", type=int, default=42)
    parser.add_argument("--tiles", type=int, default=192)
    parser.add_argument("--points", type=int, default=20000)
    parser.add_argument("--components", type=int, default=7)
    args = parser.parse_args()
    study, base = configuration(args.study)
    with args.config.open() as source:
        config = yaml.safe_load(source)
    if config["resolution_km"] != 2 or config["preprocess_step"] != 1:
        raise ValueError("Convergence study requires native 2 km step1 products")
    if base["abi"]["baseline_components"] != 7:
        raise ValueError("Convergence study expects baseline 7 vs alternate 6 components")
    stages = {
        "prepare": lambda: prepare(args, study, base, config),
        "train": lambda: train(args, study, base),
        "evaluate": lambda: evaluate(args, study, base, config),
        "summarize": lambda: summarize(args, study),
    }
    stages[args.stage]()
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
