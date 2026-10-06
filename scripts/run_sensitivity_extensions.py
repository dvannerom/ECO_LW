"""Run isolated ECO geometry and nested ABI assignment diagnostics."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import yaml

from broadband import load_cubic_coefficients
from geometry_sensitivity import nested_switch_order, retrieve_geometry
from run_sensitivity_convergence import open_cache, verify, binned_summaries, configuration
from run_uncertainty_sensitivity import load_verified, sunny_predict
from sensitivity import abi_broadband, block_mean, summary
from sensitivity_diagnostics import cluster_interval
from uncertainty import build_provenance

SETTINGS = ROOT / "config/sensitivity_extensions.yaml"


def settings():
    with SETTINGS.open() as source:
        options = yaml.safe_load(source)
    fractions = options["assignment_fractions"]
    if (options["schema_version"] != 1 or sorted(set(fractions)) != fractions
            or fractions[0] != 0 or fractions[-1] > 1):
        raise ValueError("Invalid extension configuration")
    return options


def provenance(inputs):
    return build_provenance(
        root=ROOT,
        code_paths=[Path(__file__), ROOT / "src/geometry_sensitivity.py",
                    ROOT / "scripts/run_sensitivity_convergence.py",
                    ROOT / "scripts/run_uncertainty_sensitivity.py",
                    *[ROOT / f"src/{name}.py" for name in (
                        "sensitivity", "sensitivity_diagnostics", "uncertainty",
                        "broadband", "radiometry", "adm", "scene_features", "adm_fitting",
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


def geometry(args, options):
    load_verified(args.cache)
    load_verified(args.model)
    models = joblib.load(args.model)
    with np.load(args.cache) as source:
        data = {key: source[key] for key in source.files}
    dense, _ = retrieve_geometry(data["radiance"], data["angles"], "regularized")
    baseline = sunny_predict(dense, data["regimes"], models)
    spectral = sunny_predict(data["true_flux"], data["regimes"], models)
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
                prediction = sunny_predict(flux, data["regimes"], models)
                if np.any(prediction <= 0):
                    raise ValueError("Geometry broadband prediction is non-positive")
                if noise_seed is None:
                    clean_prediction = prediction
                error = prediction - data["reference"]
                rows.append({
                    "geometry": name, "angles": angles, "form": form, "status": "quantified",
                    "noise_seed": noise_seed, "diagnostics": diagnostics,
                    "truth_error_w_m2": summary(error),
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
        "scenario": load_verified(args.cache)["scenario"],
        "spectral_only_truth_error_w_m2": summary(spectral-data["reference"]),
        "limitations": [
            "Idealized angle sets, not orbit-derived sampling; maximum 15 distinct Sunny views.",
            "Equal-weight fits; no single-view fallback, scene priors or angular weighting.",
            "Minimum-view fits have no residual degrees of freedom; identifiability is not accuracy.",
            "Gaussian channel/view noise independent at requirement reference temperature.",
            "Fixed grouped held-out N2BC models; clear/cloud regime is known, not classified.",
            "Noisy repetitions measure conditional variability, not mission confidence intervals.",
            "Sunny library realism and reference spectral-domain mismatch remain unresolved.",
        ],
    }, [args.cache, str(args.cache)+".json", args.model, str(args.model)+".json"])


def assignment(args, options):
    study, base = configuration(ROOT / "config/sensitivity_convergence.yaml")
    metadata = json.loads(Path(str(args.model)+".json").read_text())
    verify(metadata)
    fitted = joblib.load(args.model)
    coefficients = load_cubic_coefficients(args.coefficients)
    library = fitted["libraries"]["regularized"]
    block = base["abi"]["block_size"]
    verify(json.loads(Path(str(args.reference)+".json").read_text()))
    reference = np.load(args.reference)
    rows = []
    for seed in options["assignment_seeds"]:
        errors = [[] for _ in options["assignment_fractions"]]
        ids = []
        switches = np.zeros(len(errors), dtype=np.int64)
        total_pixels = 0
        footprint_offset = 0
        for path in args.caches:
            index, data = open_cache(path)
            for tile in range(index["tile_count"]):
                blocks = data["valid_blocks"][tile]
                valid = np.repeat(np.repeat(blocks, block, axis=0), block, axis=1)
                posterior = fitted["model"].predict_proba(data["features"][tile][valid])
                rng_seed = np.random.SeedSequence([seed, index["day"], tile]).generate_state(1)[0]
                first, second, order = nested_switch_order(posterior, int(rng_seed))
                radiance, angles = data["radiance"][tile][valid], data["angles"][tile][valid]
                flux = abi_broadband(radiance, angles, data["planck"], first, library,
                                     "regularized", coefficients)
                alternate = abi_broadband(radiance, angles, data["planck"], second, library,
                                          "regularized", coefficients)
                if np.any((flux < 50) | (flux > 1000) | (alternate < 50) | (alternate > 1000)):
                    raise ValueError("Assignment flux outside production validity range")
                total_pixels += len(first)
                field = np.zeros((*valid.shape, 2))
                flat_valid = np.flatnonzero(valid)
                field.reshape(-1, 2)[flat_valid] = flux
                expected = reference["baseline"][footprint_offset:footprint_offset+int(blocks.sum())]
                if not np.allclose(block_mean(field, block)[blocks], expected, rtol=1e-10, atol=1e-8):
                    raise ValueError("Recomputed baseline flux differs from fixed evaluation reference")
                footprint_offset += int(blocks.sum())
                for fraction_index, fraction in enumerate(options["assignment_fractions"]):
                    count = int(np.floor(fraction * len(first)))
                    selected = order[:count]
                    field.fill(0)
                    field.reshape(-1, 2)[flat_valid[selected]] = alternate[selected]-flux[selected]
                    errors[fraction_index].append(block_mean(field, block)[blocks])
                    switches[fraction_index] += count
                origin = data["origins"][tile]
                ids.extend(f"{index['day']}:{origin[0]+y*block}:{origin[1]+x*block}"
                           for y, x in np.argwhere(blocks))
            print(f"Assignment seed {seed}, day {index['day']} completed", flush=True)
        if not np.array_equal(ids, reference["record_ids"]):
            raise ValueError("Assignment diagnostics must match fixed evaluation IDs")
        for fraction_index, fraction in enumerate(options["assignment_fractions"]):
            error = np.concatenate(errors[fraction_index])
            rows.append({
                "seed": seed, "fraction": fraction, "switched_pixels": int(switches[fraction_index]),
                "eligible_native_pixels": total_pixels,
                "w_m2": cluster_interval(error, reference["day"], reference["tile_id"],
                    study["bootstrap_replicates"], study["confidence_level"], seed),
                "percent": summary(error / reference["baseline"] * 100),
                "G16": summary(error[:, 0]), "G18": summary(error[:, 1]),
                "regimes": binned_summaries(reference, error, study),
            })
    reference.close()
    save(args.output, {
        "settings": options, "rows": rows,
        "limitations": [
            "Nested random permutation prefixes per tile; floor(fraction*eligible pixels) switched.",
            "Fixed GMM/ADM and evaluation population; native top-two component IDs retained.",
            "Sensitivity stress, not a calibrated misclassification probability or ECO scene transfer.",
            "Three assignment seeds; intervals conditional on four held-out days and selected tiles.",
            "Legacy 10% experiment uses a different random selector; not bitwise identical.",
        ],
    }, [*args.caches, args.model, str(args.model)+".json", args.reference,
        str(args.reference)+".json", args.coefficients, ROOT/"config/sensitivity_convergence.yaml"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("geometry", "assignment"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--model", required=True)
    parser.add_argument("--caches", nargs="+")
    parser.add_argument("--reference")
    parser.add_argument("--coefficients")
    args = parser.parse_args()
    options = settings()
    (geometry if args.stage == "geometry" else assignment)(args, options)


if __name__ == "__main__":
    main()
