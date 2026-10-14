"""Production-aligned sensitivity study: one model choice changed at a time.

Baselines are the production configurations that ``snakemake all`` runs:

* ABI: the written daily products (pooled held-in ``adm_k{K}``, regularized,
  VZA <= max, float32 BT storage, production N2B form). ``abi-day`` recomputes
  this baseline on every native pixel of one day, fails unless it closes on the
  written broadband product, and pairs it with each variant.
* Sunny/ECO: the nominal grouped held-out estimator of run_nominal_budget
  (pooled scene shapes, weighted quadratic N2BC, ECO goal channels). ``sunny``
  recomputes it, fails unless it closes on the nominal residuals, and pairs it
  with each variant.

Spatial, radiometric-noise and assignment terms are not repeated here: their
single source is the nominal budget report.

Stages (memory):
  abi-day   one row slab of a few (n, 6) float64 arrays per satellite and
            variant (n = chunk_rows x columns), plus per-day 10 km cell maps
            (float64, cells ~ pixels/25 per variant and satellite).
  sunny     the in-RAM Sunny library (files x views x 6), as in calculate.
  assemble  day JSONs plus one (variant, satellite) monthly map at a time.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import numpy as np
import yaml
from netCDF4 import Dataset
from sklearn.model_selection import GroupKFold

import run_nominal_budget as nominal
from abi_product import (
    SATELLITES, grid_shape, load_adm_library, planck_coefficients, read_satellite, row_slabs,
    scene_labels,
)
from broadband import load_regression_coefficients
from geometry_sensitivity import retrieve_geometry
from nominal_budget import error_classes, fit_pooled_shape, library_flux, population_weights
from product_paths import scene_id_path
from product_sensitivity import (
    GroupMoments, cell_sums, fit_n2bc_huber, n2b_flux, production_flux, slab_spectral_features,
    vza_bins,
)
from sensitivity import angular_integral
from uncertainty import build_provenance, check_provenance

SETTINGS = ROOT / "config/sensitivity_run.yaml"
ABI_VARIANTS = ("components_minus_1", "components_plus_1", "adm_daily", "adm_quadratic", "n2b_quadratic")
SUNNY_VARIANTS = ("n2bc_stratified", "n2bc_robust", "n2bc_degree3", "adm_per_file", "adm_quadratic",
                  "components_minus_1", "components_plus_1")
CODE = [Path(__file__), ROOT / "src/product_sensitivity.py", ROOT / "src/abi_product.py",
        ROOT / "src/broadband.py", ROOT / "src/sensitivity.py", ROOT / "src/nominal_budget.py",
        ROOT / "src/nominal_streaming.py", ROOT / "src/scene_features.py",
        ROOT / "src/geometry_sensitivity.py", ROOT / "scripts/run_nominal_budget.py"]

# What each variant changes, and what it can and cannot say about ECO.
DESCRIPTIONS = {
    "abi": {
        "components_minus_1": ("scene_identification", "K-1 scene taxonomy (best K-1 candidate) with its pooled ADM library"),
        "components_plus_1": ("scene_identification", "K+1 scene taxonomy (best K+1 candidate) with its pooled ADM library"),
        "adm_daily": ("angular", "Single-day in-sample regularized ADM refit instead of the pooled held-in library"),
        "adm_quadratic": ("angular", "Two-parameter quadratic pooled ADM instead of the one-parameter regularized form"),
        "n2b_quadratic": ("spectral", "Quadratic instead of cubic narrowband-to-broadband regression"),
    },
    "sunny": {
        "n2bc_stratified": ("spectral", "Separate clear/cloud N2BC with known regime instead of one global N2BC"),
        "n2bc_robust": ("spectral", "Huber-weighted instead of least-squares N2BC"),
        "n2bc_degree3": ("spectral", "Cubic instead of quadratic N2BC"),
        "adm_per_file": ("angular", "Per-file multi-view regularized fit instead of the pooled scene library"),
        "adm_quadratic": ("angular", "Pooled quadratic scene shape instead of the regularized form"),
        "components_minus_1": ("scene_identification", "K-1 ABI taxonomy applied to Sunny files"),
        "components_plus_1": ("scene_identification", "K+1 ABI taxonomy applied to Sunny files"),
    },
}
ECO_TRANSFER = {
    "abi": ("ABI two-view geometric/spectral proxy: transfers to ECO only through the matching "
            "Sunny/ECO variant (same model choice with ECO channels and geometry)."),
    "sunny": "ECO goal channels on Sunny simulations: conditional ECO evidence, no ECO product exists.",
}


def load_settings(path=None):
    with Path(path or SETTINGS).open() as source:
        sens = yaml.safe_load(source)
    if sens["schema_version"] != 2:
        raise ValueError("Sensitivity settings must use schema_version 2")
    unknown = (set(sens["abi_variants"]) - set(ABI_VARIANTS)) | (set(sens["sunny_variants"]) - set(SUNNY_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown sensitivity variants: {sorted(unknown)}")
    edges = np.asarray(sens["vza_bin_edges_deg"], dtype=float)
    if edges[0] != 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("VZA bin edges must start at 0 and increase")
    return sens


def available(variants, components, settings):
    """Variants whose K-1/K+1 taxonomy is configured among the GMM candidates."""
    shift = {"components_minus_1": -1, "components_plus_1": 1}
    return [name for name in variants
            if name not in shift or components + shift[name] in settings["components"]]


def provenance(sens_path, inputs):
    return build_provenance(
        root=ROOT, code_paths=CODE,
        configuration_paths=[Path(sens_path), nominal.SETTINGS, nominal.CONFIG_PATH, nominal.TAXONOMY_PATH,
                             ROOT / "config/eco_channel_scenarios.yaml"],
        input_paths={str(index): Path(path) for index, path in enumerate(inputs)},
    )


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as target:
        json.dump(value, target, indent=2, allow_nan=False)


def load_current(path):
    report = json.loads(Path(path).read_text())
    freshness = check_provenance(report["provenance"], ROOT)
    if freshness["status"] != "current":
        raise ValueError(f"Stale sensitivity part {path}: {freshness}")
    return report


# --------------------------------------------------------------------- ABI


def abi_variant_models(settings, components, variants, day):
    """Library/label/N2B configuration per variant; baseline under key None."""
    from nominal_streaming import adm_path, selected_candidate
    models = {}
    for name in variants:
        if name.startswith("components_"):
            count = components + (1 if name.endswith("plus_1") else -1)
            best, model, model_path = selected_candidate(settings, count, nominal)
            path = adm_path(settings, count)
            models[name] = {"components": count, "labeler": model, "adm": path,
                            "inputs": [model_path, path]}
        elif name == "adm_daily":
            path = adm_path(settings, components, day=day)
            models[name] = {"adm": path, "inputs": [path]}
        elif name == "adm_quadratic":
            path = adm_path(settings, components, "quadratic")
            models[name] = {"adm": path, "inputs": [path]}
        else:
            models[name] = {"inputs": []}
    for name, entry in models.items():
        if "adm" in entry:
            nominal.verify(entry["adm"])
            library, form, _ = load_adm_library(entry["adm"], entry.get("components", components))
            entry.update(library=library, form=form, integral=angular_integral(library, form))
    return models


def abi_day(args, sens, settings, config):
    from nominal_streaming import adm_path, spectral_model_path
    day, components = args.day, settings["nominal_components"]
    all_days = settings["training_days"] + settings["validation_days"]
    if day not in all_days:
        raise ValueError(f"Day {day} is not a nominal training or validation day")
    resolution = int(config["resolution_km"])
    block = int(round(sens["cell_km"] / resolution))
    if block * resolution != sens["cell_km"] or sens["chunk_rows"] % block:
        raise ValueError("cell_km must be a multiple of the resolution and chunk_rows of the cell block")
    variants = available(sens["abi_variants"], components, settings)
    baseline_path = adm_path(settings, components)
    nominal.verify(baseline_path)
    library, form, _ = load_adm_library(baseline_path, components)
    integral = angular_integral(library, form)
    production = load_regression_coefficients(args.coefficients, args.n2b_form)
    quadratic = load_regression_coefficients(args.coefficients, "quadratic")
    models = abi_variant_models(settings, components, variants, day)
    edges = sens["vza_bin_edges_deg"]
    n_bins = len(edges)  # last index collects out-of-range angles (none for paired pixels)
    radiance_file = f"data/preprocessed_files/abi_{day}_res{resolution}km_step1.nc"
    scene_file = scene_id_path(day, resolution, components)
    product_file = f"data/broadband_flux/broadband_flux_{day}_res{resolution}km.nc"
    stats = {name: {sat: {"global": GroupMoments(1), "scene": GroupMoments(components),
                          "vza": GroupMoments(n_bins), "baseline": GroupMoments(1),
                          "baseline_scene": GroupMoments(components), "cell": GroupMoments(1),
                          "lost": 0, "gained": 0}
                    for sat in SATELLITES} for name in variants}
    closure = {sat: {"max_abs_w_m2": 0., "nan_mismatch": 0, "valid": 0} for sat in SATELLITES}
    label_check = {"pixels": 0, "mismatch": 0}
    with Dataset(radiance_file) as source, Dataset(scene_file) as scene, Dataset(product_file) as product:
        height, width = grid_shape(source)
        cells = (-(-height // block), -(-width // block))
        maps = {f"{name}_{sat}_{kind}": np.zeros(cells, dtype=np.float64 if kind == "sum" else np.int32)
                for name in variants for sat in SATELLITES for kind in ("sum", "count")}
        planck = {sat: planck_coefficients(source, sat) for sat in SATELLITES}
        model_path = spectral_model_path(settings)
        reference_model = joblib.load(nominal.ROOT / model_path)["model"]
        slabs = list(row_slabs(height, sens["chunk_rows"]))
        check_slab = len(slabs) // 2  # mid-disk rows; edge slabs are mostly off-Earth
        for slab, (start, stop) in enumerate(slabs):
            rows = slice(start, stop)
            labels = scene_labels(scene, start * width, stop * width)
            shifted = {}
            if any(name.startswith("components_") for name in variants) or slab == check_slab:
                features, identified = slab_spectral_features(source, rows)
                if slab == check_slab and identified.any():
                    # The taxonomy relabelling path must reproduce the production scene_id.
                    check = np.full(len(labels), -1)
                    check[identified] = reference_model.predict(features)
                    label_check["pixels"] += len(labels)
                    label_check["mismatch"] += int(np.count_nonzero(check != labels))
                for name in variants:
                    if name.startswith("components_"):
                        relabelled = np.full(len(labels), -1, dtype=np.int64)
                        if identified.any():
                            relabelled[identified] = models[name]["labeler"].predict(features)
                        shifted[name] = relabelled
                del features
            row_cells = slice(start // block, -(-stop // block))
            for sat in SATELLITES:
                radiance, angles = read_satellite(source, sat, rows)
                bt, base = production_flux(radiance, angles, labels, planck[sat], library, form,
                                           integral, production, args.max_vza)
                written = np.ma.filled(product.variables[f"flux_{sat}"][rows, :].astype(np.float64),
                                       np.nan).ravel()
                finite = np.isfinite(base)
                closure[sat]["nan_mismatch"] += int(np.count_nonzero(finite != np.isfinite(written)))
                both = finite & np.isfinite(written)
                closure[sat]["valid"] += int(np.count_nonzero(finite))
                if both.any():
                    closure[sat]["max_abs_w_m2"] = max(closure[sat]["max_abs_w_m2"],
                                                       float(np.max(np.abs(base[both] - written[both]))))
                del written, both
                bins = vza_bins(angles, edges)
                for name in variants:
                    entry = models[name]
                    if name == "n2b_quadratic":
                        flux = n2b_flux(bt, quadratic)
                    else:
                        _, flux = production_flux(
                            radiance, angles, shifted.get(name, labels), planck[sat], entry["library"],
                            entry["form"], entry["integral"], production, args.max_vza)
                    valid = np.isfinite(flux)
                    paired = finite & valid
                    record = stats[name][sat]
                    record["lost"] += int(np.count_nonzero(finite & ~valid))
                    record["gained"] += int(np.count_nonzero(~finite & valid))
                    difference = flux[paired] - base[paired]
                    record["global"].update(np.zeros(len(difference), dtype=np.int64), difference)
                    record["scene"].update(labels[paired], difference)
                    record["vza"].update(bins[paired], difference)
                    record["baseline"].update(np.zeros(len(difference), dtype=np.int64), base[paired])
                    record["baseline_scene"].update(labels[paired], base[paired])
                    sums, counts = cell_sums(
                        np.where(paired, flux - base, 0.).reshape(stop - start, width),
                        paired.reshape(stop - start, width), block)
                    maps[f"{name}_{sat}_sum"][row_cells] += sums
                    maps[f"{name}_{sat}_count"][row_cells] += counts.astype(np.int32)
                    del flux, valid, paired, difference
                del radiance, angles, bt, base, finite, bins
            if slab % 10 == 0:
                print(f"Sensitivity day {day}: rows {stop}/{height}", flush=True)
    for sat, record in closure.items():
        if record["nan_mismatch"] or record["max_abs_w_m2"] > sens["closure_tolerance_w_m2"]:
            raise RuntimeError(f"Day {day} {sat}: baseline does not close on {product_file}: {record}")
    if label_check["mismatch"] > sens["label_mismatch_tolerance"] * max(label_check["pixels"], 1):
        raise RuntimeError(f"Day {day}: feature relabelling differs from production scene_id: {label_check}")
    for name in variants:
        for sat in SATELLITES:
            counts = maps[f"{name}_{sat}_count"]
            occupied = counts > 0
            stats[name][sat]["cell"].update(np.zeros(int(occupied.sum()), dtype=np.int64),
                                            maps[f"{name}_{sat}_sum"][occupied] / counts[occupied])
    np.savez(args.maps, **{key: (value.astype(np.float32) if value.dtype == np.float64
                                 else value.astype(np.uint8 if block * block < 256 else np.uint16))
                           for key, value in maps.items()})
    inputs = [radiance_file, scene_file, product_file, args.coefficients, baseline_path,
              *[path for entry in models.values() for path in entry["inputs"]]]
    write(args.output, {
        "day": day, "split": "held_in" if day in settings["training_days"] else "held_out",
        "components": components, "variants": variants, "cell_block": block,
        "closure": closure, "label_check": label_check,
        "stats": {name: {sat: {key: (value.state() if isinstance(value, GroupMoments) else value)
                               for key, value in record.items()}
                         for sat, record in by_sat.items()}
                  for name, by_sat in stats.items()},
        "provenance": provenance(args.settings, inputs),
    })


# ------------------------------------------------------------------- Sunny


def sunny_predictions(data, labels, population, folds, variant, options):
    """Grouped held-out broadband prediction[n] [W m-2] for one estimator.

    variant None is the nominal estimator of run_nominal_budget.calculate;
    every other variant changes exactly one choice. Returns None if a held-out
    scene lacks a training ADM (unsupported taxonomy).
    """
    n = len(labels)
    k = len(population)
    form = "quadratic" if variant == "adm_quadratic" else "regularized"
    prediction = np.full(n, np.nan)
    per_file = (retrieve_geometry(data["eco_radiance"], data["angles"], "regularized")[0]
                if variant == "adm_per_file" else None)
    for training, test in GroupKFold(folds).split(labels, groups=data["groups"]):
        train_weights, _ = population_weights(labels[training], population)
        if per_file is None:
            library = np.zeros((k, 6, 2 if form == "quadratic" else 1))
            present = np.zeros(k, dtype=bool)
            for scene in np.unique(labels[training]):
                library[scene] = fit_pooled_shape(data["eco_radiance"][training[labels[training] == scene]],
                                                  data["angles"], form)
                present[scene] = True
            if np.any(~present[labels[test]]):
                return None
            flux = library_flux(data["eco_radiance"][test], data["angles"], labels[test], library, form)
        else:
            flux = per_file[test]
        band, reference = data["true_band_flux"][training], data["reference"][training]
        if variant == "n2bc_stratified":
            for regime in np.unique(data["regimes"][test]):
                fit_rows = data["regimes"][training] == regime
                scaler, poly, regression = nominal.fit_n2bc(band[fit_rows], reference[fit_rows],
                                                            train_weights[fit_rows])
                rows = data["regimes"][test] == regime
                prediction[test[rows]] = regression.predict(poly.transform(scaler.transform(flux[rows])))
            continue
        if variant == "n2bc_robust":
            scaler, poly, regression = fit_n2bc_huber(
                band, reference, train_weights, 2, options["robust_huber_cutoff"],
                options["robust_max_iterations"], options["robust_tolerance"])
        else:
            scaler, poly, regression = nominal.fit_n2bc(band, reference, train_weights,
                                                        3 if variant == "n2bc_degree3" else 2)
        prediction[test] = regression.predict(poly.transform(scaler.transform(flux)))
    if np.any(~np.isfinite(prediction)) or np.any(prediction <= 0):
        raise ValueError(f"Invalid Sunny prediction for variant {variant}")
    return prediction


def sunny(args, sens, settings, config):
    from nominal_streaming import read_candidates, selected_candidate, spectral_model_path
    directory = Path(settings["output_dir"])
    model_path = spectral_model_path(settings)
    data_path, residual_path = directory/"sunny.npz", directory/"residuals.npz"
    for path in (model_path, data_path):
        nominal.verify(path)
    budget_path = directory / "budget_report.json"
    if check_provenance(json.loads(budget_path.read_text())["provenance"], ROOT)["status"] != "current":
        raise ValueError("Stale nominal budget report")
    fitted = joblib.load(model_path)
    with np.load(data_path) as source:
        data = {key: source[key] for key in source.files}
    labels, _ = nominal.classify(fitted["model"], data["abi_radiance"], data["planck"])
    weights, support = population_weights(labels, fitted["population"])
    folds = settings["folds"]
    baseline = sunny_predictions(data, labels, fitted["population"], folds, None, sens["sunny"])
    with np.load(residual_path) as residuals:
        nominal_error = residuals["spectral"][:, 0] + residuals["angular"][:, 0]
    closure = float(np.max(np.abs(baseline - data["reference"] - nominal_error)))
    if closure > sens["sunny"]["closure_tolerance_w_m2"]:
        raise RuntimeError(f"Sunny baseline differs from nominal residuals by {closure} W/m2")
    regimes = data["regimes"]

    def classes(values, scale=None):
        values = values if scale is None else values / scale * 100
        row = error_classes(values[:, None], weights)
        row["by_regime"] = {}
        for regime in np.unique(regimes):
            selected = regimes == regime
            mass = weights[selected].sum()
            if mass > 0:
                row["by_regime"][str(regime)] = error_classes(values[selected, None],
                                                              weights[selected] / mass)
        return row

    results = {}
    components = settings["nominal_components"]
    inputs = [model_path, str(model_path) + ".json", data_path, str(data_path) + ".json",
              residual_path, budget_path]
    for variant in available(sens["sunny_variants"], components, settings):
        variant_labels, population = labels, fitted["population"]
        if variant.startswith("components_"):
            count = components + (1 if variant.endswith("plus_1") else -1)
            best, model, path = selected_candidate(settings, count, nominal)
            counts = np.asarray(read_candidates(settings, nominal)["best_by_components"][str(count)]
                                ["heldout_counts"], dtype=float)
            population = counts / counts.sum()
            variant_labels, _ = nominal.classify(model, data["abi_radiance"], data["planck"])
            inputs += [path, str(path) + ".json"]
        prediction = sunny_predictions(data, variant_labels, population, folds, variant, sens["sunny"])
        if prediction is None:
            results[variant] = {"status": "unsupported",
                                "reason": "A held-out Sunny scene lacks training support in this taxonomy"}
            continue
        results[variant] = {
            "status": "quantified",
            "paired_difference_w_m2": classes(prediction - baseline),
            "paired_difference_percent": classes(prediction - baseline, data["reference"]),
            "truth_error_w_m2": classes(prediction - data["reference"]),
            "label_change_fraction": float(weights @ (variant_labels != labels)),
        }
        print(f"Sunny variant {variant} completed", flush=True)
    write(args.output, {
        "scenario": settings["scenario"], "folds": folds, "support": support,
        "closure_max_abs_w_m2": closure,
        "baseline_truth_error_w_m2": classes(baseline - data["reference"]),
        "variants": results,
        "provenance": provenance(args.settings, inputs),
    })


# ---------------------------------------------------------------- assemble


def summarize(states, reference=None):
    moments = GroupMoments(len(states[0]["count"]))
    for state in states:
        moments.merge(state["count"], state["mean"], state["m2"])
    return moments.summary(reference)


def assemble(args, sens, settings, config):
    days = [load_current(path) for path in args.days]
    sunny_part = load_current(args.sunny)
    components = settings["nominal_components"]
    variants = available(sens["abi_variants"], components, settings)
    expected = sorted(settings["training_days"] + settings["validation_days"])
    if sorted(part["day"] for part in days) != expected or any(part["variants"] != variants for part in days):
        raise ValueError("Day parts do not cover the nominal days and configured variants")
    splits = {"held_in": [p for p in days if p["split"] == "held_in"],
              "held_out": [p for p in days if p["split"] == "held_out"], "all": days}
    abi = {}
    map_arrays = {}
    for name in variants:
        group, description = DESCRIPTIONS["abi"][name]
        entry = {"source_group": group, "description": description, "splits": {}, "monthly_10km": {}}
        for split, parts in splits.items():
            entry["splits"][split] = {}
            for sat in SATELLITES:
                records = [part["stats"][name][sat] for part in parts]
                base = summarize([r["baseline"] for r in records])[0]
                base_scene = summarize([r["baseline_scene"] for r in records])
                base_mean = np.array([base["bias_w_m2"]])
                scene_mean = np.array([row.get("bias_w_m2", 0.) for row in base_scene])
                entry["splits"][split][sat] = {
                    "pixel": summarize([r["global"] for r in records], base_mean)[0],
                    "by_scene": summarize([r["scene"] for r in records], scene_mean),
                    "by_vza_bin": summarize([r["vza"] for r in records])[:-1],
                    "daily_10km_cell_mean": summarize([r["cell"] for r in records])[0],
                    "baseline_mean_flux_w_m2": base["bias_w_m2"],
                    "validity_lost": sum(r["lost"] for r in records),
                    "validity_gained": sum(r["gained"] for r in records),
                }
        for sat in SATELLITES:
            total = count = None
            for path in args.maps:
                with np.load(path) as maps:
                    day_sum = maps[f"{name}_{sat}_sum"].astype(np.float64)
                    day_count = maps[f"{name}_{sat}_count"].astype(np.int32)
                total = day_sum if total is None else total + day_sum
                count = day_count if count is None else count + day_count
            monthly = np.full(total.shape, np.nan, dtype=np.float32)
            occupied = count > 0
            monthly[occupied] = total[occupied] / count[occupied]
            moments = GroupMoments(1)
            moments.update(np.zeros(int(occupied.sum()), dtype=np.int64), monthly[occupied])
            entry["monthly_10km"][sat] = {**moments.summary()[0],
                                          "p95_abs_w_m2": float(np.percentile(np.abs(monthly[occupied]), 95))}
            map_arrays[f"{name}_{sat}"] = monthly
            del total, count
        abi[name] = entry
    np.savez_compressed(args.monthly_maps, **map_arrays)
    sunny_results = {}
    for name, result in sunny_part["variants"].items():
        group, description = DESCRIPTIONS["sunny"][name]
        sunny_results[name] = {"source_group": group, "description": description, **result}
    report = {
        "schema_version": 2,
        "settings": sens,
        "nominal": {
            "components": components, "training_days": settings["training_days"],
            "validation_days": settings["validation_days"], "scenario": settings["scenario"],
            "settings_file": str(nominal.SETTINGS.relative_to(ROOT)),
        },
        "production": {"adm": f"pooled held-in regularized adm_k{components}",
                       "n2b_form": args.n2b_form, "max_vza_deg": args.max_vza,
                       "bt_storage": "float32", "product_resolution_km": int(config["resolution_km"]),
                       "cell_km": sens["cell_km"]},
        "abi": {"closure": {str(part["day"]): part["closure"] for part in days},
                "label_check": {str(part["day"]): part["label_check"] for part in days},
                "eco_transfer": ECO_TRANSFER["abi"], "variants": abi},
        "sunny": {"scenario": sunny_part["scenario"], "folds": sunny_part["folds"],
                  "support": sunny_part["support"], "closure_max_abs_w_m2": sunny_part["closure_max_abs_w_m2"],
                  "baseline_truth_error_w_m2": sunny_part["baseline_truth_error_w_m2"],
                  "eco_transfer": ECO_TRANSFER["sunny"], "variants": sunny_results},
        "single_source_terms": {
            "spatial": "nominal budget report (ABI 2-to-10 km footprint proxy)",
            "radiometric": "nominal budget report (ECO NEdT on Sunny)",
            "assignment": "nominal budget report (ABI-noise scene transitions)",
        },
        "scope": sens["scope"],
        "limitations": [
            "One-at-a-time model-choice sensitivities, not uncertainties: no variant is claimed as truth.",
            "Sensitivities are not additive; their combination needs a covariance model.",
            "ABI held-in days reuse ADM training data; held-out days are reported separately.",
            "Monthly 10 km maps average paired differences over all configured days; validity changes are counted separately.",
            "ABI variants use ABI channels/two-view geometry; ECO relevance is through the Sunny variants.",
            "Sunny n2bc_stratified uses the known clear/cloud regime, not a classified one.",
        ],
        "provenance": provenance(args.settings, [*args.days, *args.maps, args.sunny]),
    }
    write(args.output, report)


def main():
    global SETTINGS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("abi-day", "sunny", "assemble"))
    parser.add_argument("--settings", type=Path, default=SETTINGS)
    parser.add_argument("--nominal-settings", type=Path, default=nominal.SETTINGS)
    parser.add_argument("--taxonomy", type=Path, default=nominal.TAXONOMY_PATH)
    parser.add_argument("--day", type=int)
    parser.add_argument("--coefficients", default="data/models/narrowband_to_broadband_coeffs.json")
    parser.add_argument("--n2b-form", default="cubic", choices=("linear", "quadratic", "cubic"))
    parser.add_argument("--max-vza", type=float, default=70.)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maps", nargs="+")
    parser.add_argument("--days", nargs="+")
    parser.add_argument("--sunny")
    parser.add_argument("--monthly-maps")
    args = parser.parse_args()
    SETTINGS = args.settings.resolve()
    args.settings = SETTINGS
    nominal.SETTINGS = args.nominal_settings.resolve()
    nominal.TAXONOMY_PATH = args.taxonomy.resolve()
    sens = load_settings(SETTINGS)
    settings, config = nominal.options()
    if sorted(int(day) for day in config["days"]) != sorted(settings["training_days"] + settings["validation_days"]):
        raise ValueError("config.yaml days must equal the nominal training plus validation days")
    if args.stage == "abi-day":
        if args.day is None or not args.maps or len(args.maps) != 1:
            parser.error("abi-day needs --day and one --maps path")
        args.maps = args.maps[0]
        abi_day(args, sens, settings, config)
    elif args.stage == "sunny":
        sunny(args, sens, settings, config)
    else:
        if not args.days or not args.maps or not args.sunny or not args.monthly_maps:
            parser.error("assemble needs --days, --maps, --sunny and --monthly-maps")
        assemble(args, sens, settings, config)


if __name__ == "__main__":
    main()
