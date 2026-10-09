"""Exhaustive ABI nominal assessment; memory is bounded by one row chunk.

Feature caches contain every eligible native pixel, not selected tiles.
Spatial passes partition the grid separately for each block size, so no
complete footprint is lost at a processing boundary.
"""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from netCDF4 import Dataset
from scipy.optimize import minimize

from broadband import load_cubic_coefficients
from nominal_budget import lognormal_radiance
from scene_features import SPECTRAL_CHANNEL_INDICES, spectral_features_from_bt
from sensitivity import angular_basis, abi_broadband
from uncertainty import check_provenance
from gmm_stability import (
    cluster_stability_metrics as bootstrap_cluster_stability_metrics,
    fit_candidate as fit_stability_candidate,
    transform_predict as stability_transform_predict,
)


def validate_settings(settings, config):
    days = settings["training_days"] + settings["validation_days"]
    if set(days) != set(config["days"]):
        raise ValueError("Full ABI mode must partition every configured day")
    for key in ("chunk_rows", "batch_points", "gmm_initializations", "gmm_max_iter"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key, default in (("gmm_workers", 1), ("gmm_chunk_points", 65536),
                         ("netcdf_chunk_cache_mb", 128)):
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key, default in (("stability_max_points", 250000),
                         ("stability_repeats", 3), ("stability_seed", 42)):
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < settings["gmm_tol"] < 1:
        raise ValueError("gmm_tol must be in (0,1)")
    blocks = settings["spatial"]["block_sizes"]
    if (not blocks or blocks[0] != 1 or sorted(set(blocks)) != blocks
            or any(not isinstance(block, int) or block < 1 for block in blocks)
            or settings["spatial"]["nominal_block_size"] not in blocks
            or settings["spatial"]["nominal_block_size"] *
            settings["spatial"]["native_resolution_km"] != settings["product_resolution_km"]):
        raise ValueError("Spatial blocks must include identity and nominal product resolution")
    if settings["nominal_components"] not in settings["components"] or not settings["seeds"]:
        raise ValueError("Include nominal_components and at least one seed")
    if settings.get("maximum_vza_deg") != 70:
        raise ValueError("Nominal full ABI eligibility requires maximum_vza_deg: 70")
    if settings["spatial"]["native_resolution_km"] != 2:
        raise ValueError("Full ABI inputs must be native 2 km preprocessed files")


def source_path(day):
    return Path(f"data/preprocessed_files/abi_{day}_res2km_step1.nc")


def cache_path(settings, day):
    return Path(settings["cache_root"]) / str(day)


def configure_chunk_cache(source, settings):
    """Bound each compressed variable's cache without changing slab geometry."""
    size = settings.get("netcdf_chunk_cache_mb", 128) * 1024**2
    for satellite in ("G16", "G18"):
        for name in (f"rad_{satellite}_interp", f"rad_{satellite}_interp_corr",
                     f"lza_{satellite}_interp_corr"):
            variable = source.variables[name]
            if variable.chunking() != "contiguous":
                _, slots, preemption = variable.get_var_chunk_cache()
                variable.set_var_chunk_cache(size, slots, preemption)


def observations(settings, day, block=1):
    """Yield grid-aligned slabs: raw/corrected[y,x,2,6], angles[y,x,2].

All native pixels are visited for block=1, including partial edge chunks.
For block>1 only the incomplete *disk-grid* bottom/right footprint is omitted.
"""
    with Dataset(source_path(day)) as source:
        configure_chunk_cache(source, settings)
        height, width = source.variables["lat_interp_grid"].shape
        planck = np.stack([np.asarray(source.variables[f"planck_{sat}"][:])[
            list(SPECTRAL_CHANNEL_INDICES)] for sat in ("G16", "G18")])
        rows = max(block, settings["chunk_rows"] // block * block)
        height, width = height // block * block, width // block * block
        for y in range(0, height, rows):
            region = (slice(y, min(y + rows, height)), slice(0, width))

            def read(name, channels=False):
                selection = region + ((list(SPECTRAL_CHANNEL_INDICES),) if channels else ())
                return np.asarray(np.ma.filled(source.variables[name][selection], np.nan),
                                  dtype=np.float64)

            raw = np.stack([read(f"rad_{sat}_interp", True) for sat in ("G16", "G18")], axis=2)
            corrected = np.stack([read(f"rad_{sat}_interp_corr", True)
                                  for sat in ("G16", "G18")], axis=2)
            angles = np.stack([read(f"lza_{sat}_interp_corr") for sat in ("G16", "G18")], axis=2)
            valid = (np.all(np.isfinite(raw) & (raw > 0), axis=(2, 3))
                     & np.all(np.isfinite(corrected) & (corrected > 0) & (corrected < 1000),
                              axis=(2, 3))
                     & np.all(np.isfinite(angles) & (angles >= 0)
                              & (angles <= settings["maximum_vza_deg"]), axis=2))
            # Invalid radiances do not enter the Planck logarithm.
            fk1, fk2, bc1, bc2 = np.moveaxis(planck, -1, 0)
            safe = np.where(valid[:, :, None, None], raw, 1.)
            bt = (fk2 / np.log1p(fk1 / safe) - bc1) / bc2
            valid &= np.all(np.isfinite(bt) & (bt > 0) & (bt < 1000), axis=(2, 3))
            features = spectral_features_from_bt(bt.mean(axis=2)).astype(np.float32)
            yield (y, 0), raw, corrected, angles, planck, valid, features


def prepare_cache(settings, day, runner):
    if day not in settings["training_days"] + settings["validation_days"]:
        raise ValueError("Cache day is not configured")
    output = cache_path(settings, day)
    output.mkdir(parents=True, exist_ok=True)
    count = sum(int(chunk[5].sum()) for chunk in observations(settings, day))
    if count < 100:
        raise ValueError(f"Day {day} has fewer than 100 eligible ABI pixels")
    features = np.lib.format.open_memmap(output / "features.npy", mode="w+",
                                        dtype=np.float32, shape=(count, 10))
    offset = 0
    for _, _, _, _, planck, valid, values in observations(settings, day):
        n = int(valid.sum())
        features[offset:offset+n] = values[valid]
        offset += n
    if offset != count:
        raise RuntimeError("ABI eligibility changed while preparing features")
    features.flush()
    del features
    runner.write(output / "index.json", {
        "day": day, "pixels": count, "planck": planck.tolist(),
        "provenance": runner.provenance([source_path(day)]),
        "scope": "Every eligible native two-satellite overlap pixel; no texture or tile screening",
    })


def open_features(settings, day):
    path = cache_path(settings, day)
    index = json.loads((path / "index.json").read_text())
    freshness = check_provenance(index["provenance"], Path(__file__).resolve().parents[1])
    if freshness["status"] != "current":
        raise ValueError(f"Stale full ABI feature cache {path}: {freshness}")
    features = np.load(path / "features.npy", mmap_mode="r")
    if features.shape != (index["pixels"], 10):
        raise ValueError(f"Invalid feature cache shape for day {day}")
    return index, features


def feature_batches(settings, days):
    for day in days:
        _, features = open_features(settings, day)
        for start in range(0, len(features), settings["batch_points"]):
            yield features[start:start + settings["batch_points"]]


def prepare_stability_sample(settings, runner):
    """Prepare a bounded, reproducible production-style train/test feature sample."""
    days = settings["training_days"]
    maximum = settings.get("stability_max_points", 250000)
    per_day = max(2, maximum // len(days))
    seed = settings.get("stability_seed", 42)
    child_seeds = np.random.SeedSequence(seed).spawn(len(days))
    parts, counts = [], {}
    for day, child_seed in zip(days, child_seeds):
        _, features = open_features(settings, day)
        rng = np.random.default_rng(child_seed)
        size = min(per_day, len(features))
        selected = rng.choice(len(features), size=size, replace=False)
        parts.append(np.asarray(features[selected]))
        counts[str(day)] = size
    sample = np.concatenate(parts, axis=0)
    if len(sample) < 2:
        raise ValueError("At least two sampled training features are required for stability")
    order = np.random.default_rng(seed).permutation(len(sample))
    split = max(1, int(0.8 * len(order)))
    if split == len(order):
        raise ValueError("Stability sample must contain at least one held-out feature")
    path = Path(settings["output_dir"]) / "stability_sample.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_features=sample[order[:split]],
             test_features=sample[order[split:]])
    runner.write(str(path) + ".json", {
        "seed": seed, "maximum_points": maximum, "training_days": days,
        "sampled_points_by_day": counts, "train_points": split,
        "test_points": len(order) - split,
        "scope": "Per-day capped sample, randomly split 80/20 for production-style bootstrap ARI",
        "provenance": runner.provenance([
            *[cache_path(settings, day) for day in days],
        ]),
    })


def prepare_training(settings, runner):
    """Cache shared full-population preprocessing and float64 PCA scores.

    For N held-in pixels and d retained dimensions, disk storage is 8*N*d
    bytes. RAM is bounded by batch_points; all training rows contribute.
    """
    from streaming_gmm import prepare_streaming_preprocessing

    directory = Path(settings["output_dir"]) / "training_pca"
    directory.mkdir(parents=True, exist_ok=True)
    preprocessor = prepare_streaming_preprocessing(
        lambda: feature_batches(settings, settings["training_days"]),
        chunk_rows=settings.get("gmm_chunk_points", 65536),
    )
    count, scaler, pca = preprocessor
    joblib.dump(preprocessor, directory / "preprocessing.joblib")
    total = 0
    for day in settings["training_days"]:
        index, features = open_features(settings, day)
        values = np.lib.format.open_memmap(
            directory / f"{day}.npy", mode="w+", dtype=np.float64,
            shape=(len(features), pca.n_components_),
        )
        for start in range(0, len(features), settings["batch_points"]):
            chunk = np.asarray(features[start:start + settings["batch_points"]],
                               dtype=np.float64)
            values[start:start + len(chunk)] = pca.transform(scaler.transform(chunk))
        values.flush()
        del values
        total += index["pixels"]
    if total != count:
        raise RuntimeError("Full training population changed during PCA preparation")
    runner.write(directory / "index.json", {
        "pixels": count, "dimensions": pca.n_components_,
        "training_days": settings["training_days"],
        "provenance": runner.provenance([
            cache_path(settings, day) for day in settings["training_days"]
        ]),
    })


def read_record(path, runner):
    record = json.loads(Path(path).read_text())
    freshness = check_provenance(
        record["provenance"], getattr(runner, "ROOT", Path(__file__).resolve().parents[1])
    )
    if freshness["status"] != "current":
        raise ValueError(f"Stale nominal stage record {path}: {freshness}")
    return record


def contingency_ari(matrix):
    """Exact adjusted Rand index from a contingency matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    choose = lambda values: (values * (values - 1) / 2).sum()
    pairs = choose(matrix.sum())
    if pairs == 0:
        return 1.
    a, b = choose(matrix.sum(axis=1)), choose(matrix.sum(axis=0))
    expected, maximum = a * b / pairs, (a + b) / 2
    return 1. if maximum == expected else float((choose(matrix) - expected) / (maximum - expected))


def training_options(settings, workers):
    options = {"workers": workers, "chunk_rows": settings.get("gmm_chunk_points", 65536)}
    directory = Path(settings["output_dir"]) / "training_pca"
    # Direct Python callers may fit without a prepared cache; the DAG requires it.
    if directory.exists():
        record = json.loads((directory / "index.json").read_text())
        freshness = check_provenance(record["provenance"], Path(__file__).resolve().parents[1])
        if freshness["status"] != "current":
            raise ValueError(f"Stale training PCA cache {directory}: {freshness}")
        if record["training_days"] != settings["training_days"]:
            raise ValueError("PCA cache training days do not match the configured split")
        preprocessor = joblib.load(directory / "preprocessing.joblib")
        if (preprocessor[0] != record["pixels"]
                or preprocessor[2].n_components_ != record["dimensions"]):
            raise ValueError("PCA cache preprocessing dimensions or population do not match")

        def transformed_batches():
            total = 0
            for day in settings["training_days"]:
                values = np.load(directory / f"{day}.npy", mmap_mode="r")
                index, _ = open_features(settings, day)
                if (values.dtype != np.float64
                        or values.shape != (index["pixels"], record["dimensions"])):
                    raise ValueError(f"Invalid transformed feature cache for day {day}")
                total += len(values)
                for start in range(0, len(values), settings["batch_points"]):
                    yield values[start:start + settings["batch_points"]]
            if total != record["pixels"]:
                raise ValueError("PCA cache population does not match preprocessing")

        options.update(preprocessor=preprocessor, transformed_batches=transformed_batches)
    return options


def score_and_labels(model, features):
    """Evaluate Gaussian densities once for both likelihood and hard labels."""
    transformed = model[1].transform(model[0].transform(features))
    scores, log_responsibilities = model[-1]._estimate_log_prob_resp(transformed)
    return scores, log_responsibilities.argmax(axis=1)


def score_model(settings, model, days):
    k = model[-1].n_components
    counts = np.zeros(k, dtype=np.int64)
    likelihoods, by_day = [], {}
    for day in days:
        total, n = 0., 0
        daily = np.zeros(k, dtype=np.int64)
        for features in feature_batches(settings, [day]):
            scores, labels = score_and_labels(model, features)
            total += float(scores.sum())
            n += len(features)
            daily += np.bincount(labels, minlength=k)
        by_day[str(day)] = daily.tolist()
        counts += daily
        likelihoods.append(total / n)
    sizes = [sum(by_day[str(day)]) for day in days]
    return counts, by_day, likelihoods, float(np.average(likelihoods, weights=sizes))


def score_candidate(settings, components, seed, runner):
    if components not in settings["components"] or seed not in settings["seeds"]:
        raise ValueError("GMM candidate is not configured")
    path = Path(settings["output_dir"]) / f"candidate_k{components}_s{seed}.joblib"
    runner.verify(path)
    counts, by_day, daily, mean = score_model(
        settings, joblib.load(path), settings["validation_days"]
    )
    runner.write(Path(settings["output_dir"]) / "candidate_scores" /
                 f"k{components}_s{seed}.json", {
        "components": components, "seed": seed, "heldout_mean_log_likelihood": mean,
        "heldout_day_log_likelihood": daily, "heldout_counts": counts.tolist(),
        "minimum_occupancy": float(counts.min() / counts.sum()),
        "population_by_day": by_day,
        "provenance": runner.provenance([
            path, str(path) + ".json",
            *[cache_path(settings, day) for day in settings["validation_days"]],
        ]),
    })


def fit_candidate(settings, components, seed, runner, workers=1):
    from streaming_gmm import fit_streaming_gmm
    if components not in settings["components"] or seed not in settings["seeds"]:
        raise ValueError("GMM candidate is not configured")
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    def report_progress(initialization, iteration, objective, converged):
        print(
            f"GMM k={components}, seed={seed}, initialization "
            f"{initialization + 1}/{settings['gmm_initializations']}, "
            f"iteration={iteration}/{settings['gmm_max_iter']}, "
            f"mean_log_likelihood={objective:.6f}, converged={converged}",
            flush=True,
        )

    model = fit_streaming_gmm(
        lambda: feature_batches(settings, settings["training_days"]),
        components, seed, settings["gmm_initializations"],
        settings["gmm_max_iter"], settings["gmm_tol"],
        progress_callback=report_progress, **training_options(settings, workers))
    path = output / f"candidate_k{components}_s{seed}.joblib"
    joblib.dump(model, path)
    runner.write(str(path)+".json", {
        "provenance": runner.provenance([cache_path(settings, day)
                                        for day in settings["training_days"]] +
                                       ([output / "training_pca"]
                                        if (output / "training_pca").exists() else [])),
        "components": components, "seed": seed,
        "scope": "Every eligible held-in pixel in scaler/PCA and each EM iteration",
    })


def candidate_initialization_path(settings, components, seed, initialization):
    return (Path(settings["output_dir"]) / "candidate_initializations"
            / f"candidate_k{components}_s{seed}_i{initialization}.joblib")


def fit_candidate_initialization(settings, components, seed, initialization, runner, workers=1):
    from streaming_gmm import fit_streaming_gmm

    if components not in settings["components"] or seed not in settings["seeds"]:
        raise ValueError("GMM candidate is not configured")
    if (isinstance(initialization, bool) or not isinstance(initialization, int)
            or not 0 <= initialization < settings["gmm_initializations"]):
        raise ValueError("Initialization is not configured")

    output = candidate_initialization_path(settings, components, seed, initialization)
    output.parent.mkdir(parents=True, exist_ok=True)

    def report_progress(start, iteration, objective, converged):
        print(
            f"GMM k={components}, seed={seed}, initialization "
            f"{start + 1}/{settings['gmm_initializations']}, "
            f"iteration={iteration}/{settings['gmm_max_iter']}, "
            f"mean_log_likelihood={objective:.6f}, converged={converged}",
            flush=True,
        )

    print(
        f"Starting GMM k={components}, seed={seed}, "
        f"initialization {initialization + 1}/{settings['gmm_initializations']}",
        flush=True,
    )
    try:
        model = fit_streaming_gmm(
            lambda: feature_batches(settings, settings["training_days"]),
            components, seed, settings["gmm_initializations"],
            settings["gmm_max_iter"], settings["gmm_tol"],
            initialization_index=initialization,
            progress_callback=report_progress,
            **training_options(settings, workers),
        )
    except RuntimeError as error:
        record = {
            "status": "failed", "error": str(error), "components": components,
            "seed": seed, "initialization": initialization,
        }
        print(
            f"GMM k={components}, seed={seed}, initialization "
            f"{initialization + 1} failed: {error}",
            flush=True,
        )
    else:
        record = {
            "status": "converged", "model": model, "components": components,
            "seed": seed, "initialization": initialization,
        }
        print(
            f"Finished GMM k={components}, seed={seed}, "
            f"initialization {initialization + 1}/{settings['gmm_initializations']}",
            flush=True,
        )
    joblib.dump(record, output)
    runner.write(str(output)+".json", {
        "provenance": runner.provenance([
            *[cache_path(settings, day) for day in settings["training_days"]],
            *([Path(settings["output_dir"]) / "training_pca"]
              if (Path(settings["output_dir"]) / "training_pca").exists() else []),
        ]),
        "components": components, "seed": seed, "initialization": initialization,
        "status": record["status"],
        "scope": "Every eligible held-in pixel in scaler/PCA and each EM iteration",
    })


def combine_candidate_initializations(settings, components, seed, runner):
    if components not in settings["components"] or seed not in settings["seeds"]:
        raise ValueError("GMM candidate is not configured")

    scores = np.full(settings["gmm_initializations"], np.nan)
    failures = []
    best_model, best_score = None, -np.inf
    initialization_paths = [
        candidate_initialization_path(settings, components, seed, index)
        for index in range(settings["gmm_initializations"])
    ]
    for initialization, path in enumerate(initialization_paths):
        runner.verify(path)
        record = joblib.load(path)
        if (record["components"] != components or record["seed"] != seed
                or record["initialization"] != initialization):
            raise ValueError(f"Initialization metadata does not match {path}")
        if record["status"] == "failed":
            failures.append(
                f"Streaming GMM start {initialization + 1} failed: {record['error']}"
            )
            continue
        if record["status"] != "converged":
            raise ValueError(f"Unknown initialization status in {path}")
        model = record["model"]
        score = float(model[-1].lower_bound_)
        if not np.isfinite(score):
            raise ValueError(f"Non-finite initialization score in {path}")
        scores[initialization] = score
        if score > best_score:
            best_model, best_score = model, score

    if best_model is None:
        raise RuntimeError(
            f"No streaming GMM initialization converged for k={components}, seed={seed}. "
            + "; ".join(failures)
        )
    best_model[-1].n_init = settings["gmm_initializations"]
    best_model[-1].streaming_initialization_scores_ = scores
    best_model[-1].streaming_initialization_failures_ = tuple(failures)

    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"candidate_k{components}_s{seed}.joblib"
    joblib.dump(best_model, path)
    runner.write(str(path)+".json", {
        "provenance": runner.provenance([
            *[cache_path(settings, day) for day in settings["training_days"]],
            *initialization_paths,
            *[Path(str(source)+".json") for source in initialization_paths],
        ]),
        "components": components, "seed": seed,
        "scope": "Every eligible held-in pixel in scaler/PCA and each EM iteration",
    })


def train(settings, runner):
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    rows, selected = [], {}
    for count in settings["components"]:
        best = None
        for seed in settings["seeds"]:
            path = output / f"candidate_k{count}_s{seed}.joblib"
            runner.verify(path)
            model = joblib.load(path)
            score_path = output / "candidate_scores" / f"k{count}_s{seed}.json"
            if not score_path.exists():
                score_candidate(settings, count, seed, runner)
            score = read_record(score_path, runner)
            if score["components"] != count or score["seed"] != seed:
                raise ValueError(f"Candidate score metadata does not match {score_path}")
            counts = np.asarray(score["heldout_counts"], dtype=np.int64)
            by_day, mean = score["population_by_day"], score["heldout_mean_log_likelihood"]
            row = {key: score[key] for key in (
                "components", "seed", "heldout_mean_log_likelihood",
                "heldout_day_log_likelihood", "heldout_counts", "minimum_occupancy",
            )}
            rows.append(row)
            if np.all(counts >= 100) and (best is None or mean > best[0]):
                best = mean, seed, model, counts, by_day
            print(f"Full ABI spectral GMM k={count}, seed={seed}, likelihood={mean:.6f}", flush=True)
        if best is None:
            raise ValueError(f"Component count {count} has no adequately occupied candidate")
        selected[count] = best
    _, seed, model, counts, by_day = selected[settings["nominal_components"]]
    path = output / "spectral_gmm.joblib"
    joblib.dump({"model": model, "population": counts / counts.sum()}, path)
    # Exact percentile using a writable disk-backed partition, not a RAM copy.
    n = int(counts.sum())
    scores_path = output / "heldout_density.npy"
    scores = np.lib.format.open_memmap(scores_path, mode="w+", dtype=np.float64, shape=(n,))
    offset = 0
    for features in feature_batches(settings, settings["validation_days"]):
        scores[offset:offset+len(features)] = model.score_samples(features)
        offset += len(features)
    position = .01 * (n - 1)
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    scores.partition((lower, upper))
    p01 = float(scores[lower] + (position - lower) * (scores[upper] - scores[lower]))
    del scores
    scores_path.unlink()
    metadata = {
        "provenance": runner.provenance([
            *[cache_path(settings, day) for day in settings["training_days"] + settings["validation_days"]],
            *[output / f"candidate_k{count}_s{seed}.joblib"
              for count in settings["components"] for seed in settings["seeds"]],
            *[output / "candidate_scores" / f"k{count}_s{seed}.json"
              for count in settings["components"] for seed in settings["seeds"]]]),
        "components": settings["nominal_components"], "seed": seed,
        "selection": "Manual component count; best full held-out likelihood initialization",
        "candidates": rows, "heldout_population_counts": counts.tolist(),
        "population_by_day": by_day, "heldout_log_density_p01": p01,
        "population_scope": "Every eligible native overlap pixel on all configured held-out days",
        "fitting_scope": "All held-in pixels in every scaler/PCA/EM pass; no fitting reservoir",
        "diagnostic_reliability": (
            "Full-data model initialization agreement is distinct from the "
            "production-style bootstrap ARI in component diagnostics."
        ),
    }
    runner.write(str(path) + ".json", metadata)
    runner.write(Path(settings["diagnostic_dir"]) / "gmm_selection.json", metadata)


def paired_records(settings, days):
    for day in days:
        for _, raw, corrected, angles, planck, valid, features in observations(settings, day):
            if valid.any():
                yield day, raw[valid], corrected[valid], angles[valid], planck, features[valid]


def fit_library(settings, model, directory, cache_name="adm_fit_records.npy"):
    """Fit full-population paired-ratio least squares with a positive hemisphere.

The disk-backed ratio cache uses 36 bytes per training pixel. All six
channels/scenes are optimized together; each objective/gradient scans it once.
"""
    k = model[-1].n_components
    n = sum(open_features(settings, day)[0]["pixels"] for day in settings["training_days"])
    path = Path(directory) / cache_name
    records = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, 9))
    counts = np.zeros(k, dtype=np.int64)
    angle_sum, angle_square = np.zeros(k), np.zeros(k)
    offset = 0
    for _, _, corrected, angles, _, features in paired_records(settings, settings["training_days"]):
        labels = model.predict(features)
        basis = angular_basis(angles, "regularized")[..., 0]
        size = len(labels)
        records[offset:offset+size, :2] = basis
        records[offset:offset+size, 2:8] = corrected[:, 0] / corrected[:, 1]
        records[offset:offset+size, 8] = labels
        offset += size
        counts += np.bincount(labels, minlength=k)
        difference = angles[:, 0] - angles[:, 1]
        angle_sum += np.bincount(labels, weights=difference, minlength=k)
        angle_square += np.bincount(labels, weights=difference**2, minlength=k)
    if offset != n:
        raise RuntimeError("Feature cache and ADM training eligibility differ")
    if np.any(counts < 100) or np.any(angle_square/counts - (angle_sum/counts)**2 < .01):
        raise ValueError("Full ABI scene ADM lacks 100 pairs or angular diversity")
    grid = angular_basis(np.linspace(0, 90, 901), "regularized")[:, 0]
    lower = np.max((1e-6 - 1) / grid[grid > 0])
    upper = np.min((1e-6 - 1) / grid[grid < 0])

    def objective(flat):
        parameters = flat.reshape(k, 6)
        value, gradient = 0., np.zeros((k, 6))
        for start in range(0, n, settings["batch_points"]):
            batch = np.asarray(records[start:start+settings["batch_points"]], dtype=np.float64)
            labels = batch[:, 8].astype(np.int64)
            b = parameters[labels]
            numerator, denominator = 1 + batch[:, :1] * b, 1 + batch[:, 1:2] * b
            residual = numerator / denominator - batch[:, 2:8]
            derivative = (batch[:, :1] - batch[:, 1:2]) / denominator**2
            value += float(np.sum(residual**2 / counts[labels, None]) / 2)
            contributions = residual * derivative / counts[labels, None]
            for channel in range(6):
                gradient[:, channel] += np.bincount(
                    labels, weights=contributions[:, channel], minlength=k
                )
        return value, gradient.ravel()

    result = minimize(objective, np.zeros(k * 6), jac=True, method="L-BFGS-B",
                      bounds=[(lower, upper)] * (k * 6),
                      options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 300})
    if not result.success:
        raise RuntimeError(f"Full ABI scene ADM fit failed: {result.message}")
    library = result.x.reshape(k, 6, 1)
    del records
    path.unlink()
    return library


def adm_path(settings, components):
    return Path(settings["output_dir"]) / "component_adms" / f"adm_k{components}.joblib"


def selected_candidate(settings, components, runner):
    selection = read_record(Path(settings["diagnostic_dir"]) / "gmm_selection.json", runner)
    candidates = [row for row in selection["candidates"]
                  if row["components"] == components and min(row["heldout_counts"]) >= 100]
    if not candidates:
        raise ValueError(f"Component count {components} has no adequately occupied candidate")
    best = max(candidates, key=lambda row: row["heldout_mean_log_likelihood"])
    path = Path(settings["output_dir"]) / f"candidate_k{components}_s{best['seed']}.joblib"
    runner.verify(path)
    return best, joblib.load(path), path


def prepare_adm(settings, components, runner):
    if components not in settings["components"]:
        raise ValueError("GMM component count is not configured")
    selection_path = Path(settings["diagnostic_dir"]) / "gmm_selection.json"
    if selection_path.exists():
        best, model, model_path = selected_candidate(settings, components, runner)
        selection_inputs = [selection_path]
    elif components == settings["nominal_components"]:
        model_path = Path(settings["output_dir"]) / "spectral_gmm.joblib"
        runner.verify(model_path)
        model = joblib.load(model_path)["model"]
        best, selection_inputs = {"seed": None}, []
    else:
        raise ValueError("Component ADM fitting requires GMM selection")
    path = adm_path(settings, components)
    path.parent.mkdir(parents=True, exist_ok=True)
    library = fit_library(settings, model, path.parent, f"fit_k{components}.npy")
    joblib.dump(library, path)
    runner.write(str(path) + ".json", {
        "components": components, "seed": best["seed"],
        "provenance": runner.provenance([
            model_path, str(model_path) + ".json",
            *selection_inputs,
            *[source_path(day) for day in settings["training_days"]],
        ]),
    })


def load_adm(settings, components, runner):
    path = adm_path(settings, components)
    if not path.exists():
        prepare_adm(settings, components, runner)
    runner.verify(path)
    metadata = json.loads(Path(str(path) + ".json").read_text())
    if (Path(settings["diagnostic_dir"]) / "gmm_selection.json").exists():
        best, _, _ = selected_candidate(settings, components, runner)
    else:
        best = {"seed": None}
    if metadata["components"] != components or metadata["seed"] != best["seed"]:
        raise ValueError(f"ADM library does not match the selected candidate {path}")
    library = joblib.load(path)
    if library.shape != (components, 6, 1) or not np.isfinite(library).all():
        raise ValueError(f"Invalid ADM library {path}")
    return library


class Moments:
    """Merge population moments without retaining footprint residual arrays."""

    def __init__(self):
        self.count, self.mean, self.m2 = 0, 0., 0.

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("Non-finite full-data residuals")
        n = values.size
        if not n:
            return
        mean = float(values.mean())
        delta, total = mean - self.mean, self.count + n
        self.m2 += float(np.square(values - mean).sum()) + delta**2 * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    def report(self):
        if not self.count:
            raise ValueError("No eligible full-data residuals")
        variance = self.m2 / self.count
        rmse = float(np.sqrt(variance + self.mean**2))
        return {"bias_w_m2": self.mean, "scene_sd_w_m2": float(np.sqrt(variance)),
                "random_sd_w_m2": 0., "ensemble_rmse_w_m2": rmse, "class_rmse_w_m2": rmse,
                "raw_scene_variance": variance, "mc_scene_variance_correction": 0.,
                "negative_corrected_scene_variance": False, "realizations": 1}

    def merge(self, count, mean, m2):
        if (not isinstance(count, int) or isinstance(count, bool) or count < 1
                or not np.isfinite(mean) or not np.isfinite(m2) or m2 < 0):
            raise ValueError("Invalid residual sufficient statistics")
        delta, total = mean - self.mean, self.count + count
        self.m2 += m2 + delta**2 * self.count * count / total
        self.mean += delta * count / total
        self.count = total


def spatial(settings, config, runner):
    for day in settings["validation_days"]:
        for block in settings["spatial"]["block_sizes"]:
            spatial_part(settings, config, day, block, runner)
        print(f"Full spatial ABI day {day} completed", flush=True)
    combine_spatial(settings, config, runner)


def spatial_part(settings, config, day, block, runner):
    if day not in settings["validation_days"] or block not in settings["spatial"]["block_sizes"]:
        raise ValueError("Spatial day/block is not configured")
    output = Path(settings["output_dir"])
    model_path = output / "spectral_gmm.joblib"
    runner.verify(model_path)
    model = joblib.load(model_path)["model"]
    library = load_adm(settings, settings["nominal_components"], runner)
    coefficients = load_cubic_coefficients(config["narrowband_to_broadband_coeffs_file"])
    daily = Moments()
    for origin, raw, corrected, angles, planck, valid, _ in observations(settings, day, block):
        complete = valid.reshape(valid.shape[0]//block, block,
                                 valid.shape[1]//block, block).all(axis=(1, 3))
        if complete.any():
            daily.update(runner.spatial_tile(
                model, raw, corrected, angles, planck, library, coefficients, valid, block, origin
            ))
    daily.report()
    runner.write(output / "spatial_parts" / f"day{day}_b{block}.json", {
        "day": day, "block_size": block, "count": daily.count,
        "mean": daily.mean, "m2": daily.m2,
        "provenance": runner.provenance([
            model_path, str(model_path) + ".json",
            adm_path(settings, settings["nominal_components"]),
            str(adm_path(settings, settings["nominal_components"])) + ".json",
            source_path(day), config["narrowband_to_broadband_coeffs_file"],
        ]),
    })


def combine_spatial(settings, config, runner):
    output = Path(settings["output_dir"])
    blocks = settings["spatial"]["block_sizes"]
    totals, by_day, sources = {block: Moments() for block in blocks}, {}, []
    for day in settings["validation_days"]:
        by_day[str(day)] = {}
        for block in blocks:
            path = output / "spatial_parts" / f"day{day}_b{block}.json"
            record = read_record(path, runner)
            if record["day"] != day or record["block_size"] != block:
                raise ValueError(f"Spatial metadata does not match {path}")
            daily = Moments()
            daily.merge(record["count"], record["mean"], record["m2"])
            totals[block].merge(daily.count, daily.mean, daily.m2)
            by_day[str(day)][str(block)] = {"footprints": daily.count, **daily.report()}
            sources.append(path)
    rows = [{"block_size": block,
             "resolution_km": block * settings["spatial"]["native_resolution_km"],
             "footprints": totals[block].count, **totals[block].report()} for block in blocks]
    if rows[0]["ensemble_rmse_w_m2"] > 1e-9:
        raise RuntimeError("Full-data identity spatial route does not close")
    np.savez(output / "spatial_residuals.npz", block_sizes=blocks,
             count=[totals[b].count for b in blocks],
             mean=[totals[b].mean for b in blocks], m2=[totals[b].m2 for b in blocks])
    library = load_adm(settings, settings["nominal_components"], runner)
    joblib.dump(library, output / "abi_spatial_adm.joblib")
    runner.write(output / "spatial_report.json", {
        "provenance": runner.provenance([
            *sources, output / "spectral_gmm.joblib",
            output / "spectral_gmm.joblib.json",
            adm_path(settings, settings["nominal_components"]),
            str(adm_path(settings, settings["nominal_components"])) + ".json",
            config["narrowband_to_broadband_coeffs_file"],
            *[source_path(day) for day in settings["training_days"] +
              settings["validation_days"]],
        ]),
        "rows": rows, "by_day": by_day,
        "nominal": next(row for row in rows
                        if row["block_size"] == settings["spatial"]["nominal_block_size"]),
        "residual_storage": "Sufficient statistics count/mean/M2; not per-footprint residuals",
        "assumptions": [
            "Every complete eligible footprint, aligned to the full grid; no chunk-edge exclusions.",
            "All held-in eligible pixel pairs fit fixed positive scene ADMs.",
            "Native spectral eligibility only; no inherited texture or tile screening.",
            "ABI coarse-minus-mean-fine processing difference, not absolute ECO spatial truth.",
            "ABI-to-ECO transfer and independence from Sunny-source errors are assumed.",
        ],
    })


def assignment_day(settings, day, runner):
    if day not in settings["validation_days"]:
        raise ValueError("Assignment day is not configured")
    output = Path(settings["output_dir"])
    path = output / "spectral_gmm.joblib"
    runner.verify(path)
    fitted = joblib.load(path)
    model, population = fitted["model"], fitted["population"]
    k, repetitions = len(population), settings["noise_realizations"]
    daily, baseline = np.zeros((k, k), dtype=np.int64), np.zeros(k, dtype=np.int64)
    # Preserve the persistent day/realization stream and native pixel order.
    rngs = [np.random.default_rng(np.random.SeedSequence([settings["noise_seed"], day, r, 3]))
            for r in range(repetitions)]
    for _, raw, _, _, planck, features in paired_records(settings, [day]):
        labels = model.predict(features)
        baseline += np.bincount(labels, minlength=k)
        fk1, fk2, bc1, bc2 = np.moveaxis(planck, -1, 0)
        argument = fk2 / (bc1 + bc2 * 255.)
        sd = settings["nedt_k_at_255"] * fk1 * np.exp(argument) * fk2 * bc2 / (
            np.expm1(argument)**2 * (bc1 + bc2*255.)**2)
        for rng in rngs:
            perturbed = lognormal_radiance(raw, sd, rng.normal(size=raw.shape))
            noisy = model.predict(runner.abi_noise_features(perturbed, planck, 0))
            daily += np.bincount(labels*k + noisy, minlength=k*k).reshape(k, k)
    runner.write(output / "assignment_parts" / f"day{day}.json", {
        "day": day, "transition_counts": daily.tolist(), "baseline_counts": baseline.tolist(),
        "noise_realizations": repetitions,
        "provenance": runner.provenance([path, str(path)+".json", source_path(day)]),
    })


def combine_assignment(settings, runner):
    output = Path(settings["output_dir"])
    path = output / "spectral_gmm.joblib"
    runner.verify(path)
    population = joblib.load(path)["population"]
    k = len(population)
    counts, baseline = np.zeros((k, k), dtype=np.int64), np.zeros(k, dtype=np.int64)
    by_day, sources = {}, []
    for day in settings["validation_days"]:
        part = output / "assignment_parts" / f"day{day}.json"
        record = read_record(part, runner)
        daily = np.asarray(record["transition_counts"], dtype=np.int64)
        daily_baseline = np.asarray(record["baseline_counts"], dtype=np.int64)
        if (record["day"] != day or record["noise_realizations"] != settings["noise_realizations"]
                or daily.shape != (k, k) or daily_baseline.shape != (k,)
                or np.any(daily < 0) or np.any(daily_baseline < 0)
                or not np.array_equal(daily.sum(axis=1),
                                      daily_baseline * settings["noise_realizations"])):
            raise ValueError(f"Invalid assignment record {part}")
        by_day[str(day)] = daily.tolist()
        counts += daily
        baseline += daily_baseline
        sources.append(part)
    if np.any(counts.sum(axis=1) == 0):
        raise ValueError("Full held-out ABI population lacks a nominal scene")
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    runner.write(output / "abi_assignment.json", {
        "provenance": runner.provenance([path, str(path)+".json",
                                        *sources,
                                        *[source_path(day) for day in settings["validation_days"]]]),
        "transition_counts": counts.tolist(), "transition_probabilities": probabilities.tolist(),
        "weighted_change_probability": float(population @ (1 - np.diag(probabilities))),
        "sample_change_probability": float(1 - np.trace(counts) / counts.sum()),
        "baseline_sample_counts": baseline.tolist(), "sample_pixels": int(baseline.sum()),
        "by_day_transition_counts": by_day, "noise_realizations": settings["noise_realizations"],
        "nedt_k_at_255": settings["nedt_k_at_255"],
        "noise_model": "mean_preserving_lognormal",
        "assumptions": [
            "Every eligible held-out native pixel; no reservoir or geographic sampling.",
            "Observed ABI radiance treated as truth; fixed clean-trained GMM.",
            "Independent mean-preserving lognormal satellite/channel/pixel radiance.",
            "Radiance SD matches NEdT times the ABI Planck derivative at 255 K; asymmetric noise is a modelling assumption.",
            "Scene-conditional transitions transferred to ECO/Sunny independently of ECO-channel noise.",
        ],
    })


def assignment(settings, runner):
    for day in settings["validation_days"]:
        assignment_day(settings, day, runner)
    combine_assignment(settings, runner)


def diagnostics(settings, config, runner, components=None):
    """Full-population physical scores and production-style bootstrap ARI."""
    if components is None:
        for count in settings["components"]:
            diagnostics(settings, config, runner, count)
        combine_diagnostics(settings, runner)
        return
    if components not in settings["components"]:
        raise ValueError("GMM component count is not configured")
    output = Path(settings["output_dir"])
    stability_path = output / "stability_sample.npz"
    runner.verify(stability_path)
    stability_metadata = read_record(str(stability_path) + ".json", runner)
    if stability_metadata["training_days"] != settings["training_days"]:
        raise ValueError("Bootstrap stability sample has a different training-day split")
    with np.load(stability_path) as stability_sample:
        stability_train = stability_sample["train_features"]
        stability_test = stability_sample["test_features"]
    if (stability_train.ndim != 2 or stability_train.shape[1] != 10
            or stability_test.ndim != 2 or stability_test.shape[1] != 10):
        raise ValueError("Bootstrap stability sample must contain 10-feature matrices")
    coefficients = load_cubic_coefficients(config["narrowband_to_broadband_coeffs_file"])
    for k in (components,):
        best, model, model_path = selected_candidate(settings, k, runner)
        library = load_adm(settings, k, runner)
        stability_seed = settings.get("stability_seed", 42)
        (stability_scaler, stability_pca, stability_model,
         stability_train_transformed) = fit_stability_candidate(
            stability_train, k, True, .98, stability_seed
        )
        stability_test_transformed, stability_baseline_labels = stability_transform_predict(
            stability_test, stability_scaler, stability_pca, stability_model
        )
        heldout_loglik = float(stability_model.score_samples(stability_test_transformed).mean())
        responsibilities = stability_model.predict_proba(stability_train_transformed)
        train_entropy = -float(np.sum(
            responsibilities * np.log(np.maximum(responsibilities, 1e-12))
        ))
        diagnostic_icl = float(
            stability_model.bic(stability_train_transformed) + 2 * train_entropy
        )
        ari_min, ari_std = bootstrap_cluster_stability_metrics(
            stability_train, stability_test, stability_baseline_labels,
            k, True, .98, stability_seed,
            settings.get("stability_repeats", 3),
        )
        ratio_square, scene_counts = np.zeros((k, 6)), np.zeros(k, dtype=np.int64)
        bins = np.zeros((k, 9), dtype=np.int64)
        raw_flux, corrected_flux = Moments(), Moments()
        for _, _, corrected, angles, planck, features in paired_records(settings, settings["validation_days"]):
            _, log_probabilities = model[-1]._estimate_log_prob_resp(
                model[1].transform(model[0].transform(features))
            )
            labels = log_probabilities.argmax(axis=1)
            scene_counts += np.bincount(labels, minlength=k)
            basis = angular_basis(angles, "regularized")[..., 0]
            b = library[labels, :, 0]
            residual = (1 + basis[:, :1]*b)/(1 + basis[:, 1:2]*b) - corrected[:, 0]/corrected[:, 1]
            for channel in range(6):
                ratio_square[:, channel] += np.bincount(
                    labels, weights=residual[:, channel]**2, minlength=k
                )
            for satellite in range(2):
                angular_bin = np.minimum((angles[:, satellite]/10).astype(int), 8)
                bins += np.bincount(labels*9 + angular_bin, minlength=k*9).reshape(k, 9)
            uncorrected = abi_broadband(corrected, angles, planck, labels,
                                       np.zeros_like(library), "regularized", coefficients)
            retrieved = abi_broadband(corrected, angles, planck, labels,
                                     library, "regularized", coefficients)
            raw_flux.update(uncorrected[:, 0]-uncorrected[:, 1])
            corrected_flux.update(retrieved[:, 0]-retrieved[:, 1])
        n = int(scene_counts.sum())
        rmse = float(np.sqrt(ratio_square.sum() / (n*6)))
        worst_channel = float(np.sqrt(ratio_square.sum(axis=0) / n).max())
        worst_scene = float(np.sqrt(ratio_square.sum(axis=1) / (scene_counts*6)).max())
        spread, raw_spread = corrected_flux.report()["scene_sd_w_m2"], raw_flux.report()["scene_sd_w_m2"]
        gain = 1 - spread/raw_spread if raw_spread > 0 else 0.
        coverage = float((bins > 0).mean(axis=1).min())
        row = {
            "n_components": k, "heldout_loglik": heldout_loglik,
            "icl": diagnostic_icl,
            "ari_min": ari_min, "ari_std": ari_std,
            "adm_ratio_rmse": rmse, "worst_channel_adm_rmse": worst_channel,
            "worst_scene_adm_rmse": worst_scene, "minimum_angular_coverage": coverage,
            "exact_adm_ratio_rmse": rmse, "exact_worst_channel_adm_rmse": worst_channel,
            "exact_worst_scene_adm_rmse": worst_scene, "exact_corrected_flux_std": spread,
            "exact_raw_flux_std": raw_spread, "exact_flux_improvement": gain, "shortlisted": True,
            "criterion_adm_plateau": False,
            "criterion_occupancy": bool(scene_counts.min()/n >= .005),
            "criterion_stability": bool(np.isfinite(ari_min) and ari_min >= .8),
            "criterion_angular_coverage": coverage >= .7,
            "criterion_adm_coverage": True, "criterion_exact_adm_coverage": True,
            "criterion_exact_improvement": gain >= 0,
        }
        row["all_criteria"] = all(value for key, value in row.items() if key.startswith("criterion_"))
        runner.write(Path(settings["diagnostic_dir"]) / "component_parts" / f"k{k}.json", {
            "components": k, "row": row,
            "provenance": runner.provenance([
                model_path, str(model_path) + ".json", adm_path(settings, k),
                str(adm_path(settings, k)) + ".json",
                stability_path, str(stability_path) + ".json",
                *[source_path(day) for day in settings["validation_days"]],
                config["narrowband_to_broadband_coeffs_file"],
            ]),
        })


def combine_diagnostics(settings, runner):
    rows, sources, previous = [], [], None
    for k in settings["components"]:
        path = Path(settings["diagnostic_dir"]) / "component_parts" / f"k{k}.json"
        record = read_record(path, runner)
        if record["components"] != k or record["row"]["n_components"] != k:
            raise ValueError(f"Diagnostic metadata does not match {path}")
        row = record["row"]
        rmse = row["adm_ratio_rmse"]
        row["criterion_adm_plateau"] = (
            previous is not None and previous > 0 and (previous-rmse)/previous < .01
        )
        row["all_criteria"] = all(
            value for key, value in row.items() if key.startswith("criterion_")
        )
        for key in ("ari_min", "ari_std"):
            if row[key] is None:
                row[key] = np.nan
        rows.append(row)
        sources.append(path)
        previous = rmse
    path = Path(settings["diagnostic_dir"]) / "gmm_component_diagnostics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    runner.write(str(path)+".json", {
        "provenance": runner.provenance([
            *sources,
            Path(settings["output_dir"]) / "spectral_gmm.joblib",
            *[source_path(day) for day in settings["training_days"] +
              settings["validation_days"]],
            *[adm_path(settings, k) for k in settings["components"]],
        ]),
        "reliability": (
            "Production-style bootstrap refits on a per-day capped diagnostic sample; "
            "ARI is the minimum agreement across repeats on its fixed 20% held-out split."
        ),
        "scope": "Every eligible held-in/held-out pixel; exact cubic ABI scoring with per-day Planck",
        "notes": ["Positive constrained paired-ratio scene ADMs, same as full spatial stage.",
                  "Nine original 10-degree angular bins retained; eligibility is <=70 degrees."],
    })


def run(stage, settings, config, runner, day=None, components=None, seed=None,
        initialization=None, workers=1, block=None):
    if stage == "cache":
        if day is None:
            raise ValueError("cache requires --day")
        prepare_cache(settings, day, runner)
    elif stage == "train":
        train(settings, runner)
    elif stage == "candidate":
        if components is None or seed is None:
            raise ValueError("candidate requires --components and --seed")
        fit_candidate(settings, components, seed, runner, workers)
    elif stage == "candidate-init":
        if components is None or seed is None or initialization is None:
            raise ValueError("candidate-init requires --components, --seed and --initialization")
        fit_candidate_initialization(settings, components, seed, initialization, runner, workers)
    elif stage == "candidate-combine":
        if components is None or seed is None:
            raise ValueError("candidate-combine requires --components and --seed")
        combine_candidate_initializations(settings, components, seed, runner)
    elif stage == "assignment":
        assignment(settings, runner)
    elif stage == "spatial":
        spatial(settings, config, runner)
    elif stage == "diagnostics":
        diagnostics(settings, config, runner)
    elif stage == "preprocessing":
        prepare_training(settings, runner)
    elif stage == "stability-sample":
        prepare_stability_sample(settings, runner)
    elif stage == "score":
        if components is None or seed is None:
            raise ValueError("score requires --components and --seed")
        score_candidate(settings, components, seed, runner)
    elif stage == "adm":
        if components is None:
            raise ValueError("adm requires --components")
        prepare_adm(settings, components, runner)
    elif stage == "assignment-day":
        if day is None:
            raise ValueError("assignment-day requires --day")
        assignment_day(settings, day, runner)
    elif stage == "assignment-combine":
        combine_assignment(settings, runner)
    elif stage == "spatial-part":
        if day is None or block is None:
            raise ValueError("spatial-part requires --day and --block")
        spatial_part(settings, config, day, block, runner)
    elif stage == "spatial-combine":
        combine_spatial(settings, config, runner)
    elif stage == "diagnostics-component":
        if components is None:
            raise ValueError("diagnostics-component requires --components")
        diagnostics(settings, config, runner, components)
    elif stage == "diagnostics-combine":
        combine_diagnostics(settings, runner)
    else:
        raise ValueError(f"Unknown full nominal stage {stage}")
