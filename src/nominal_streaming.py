"""Exhaustive ABI nominal assessment; memory is bounded by one row chunk.

Feature caches contain every eligible native pixel, not selected tiles.
Spatial passes partition the grid separately for each block size, so no
complete footprint is lost at a processing boundary.
"""

import csv
from concurrent.futures.process import BrokenProcessPool
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
from netCDF4 import Dataset
from scipy.optimize import minimize

from broadband import load_cubic_coefficients
from nominal_budget import lognormal_radiance
from nominal_provenance import (
    candidate_identity, check_stage_provenance as check_nominal_stage_provenance,
    initialization_identity, sweep_identity,
)
from scene_features import SPECTRAL_CHANNEL_INDICES, spectral_features_from_bt
from sensitivity import angular_basis, angular_integral, abi_broadband
from uncertainty import check_provenance, sha256_file
from gmm_stability import (
    cluster_stability_details,
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
    if "gmm_workers" in settings:
        raise ValueError("Remove gmm_workers: full GMM scheduling now uses Snakemake's CPU allocation")
    for key, default in (("gmm_chunk_points", 65536),
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
    if "seeds" in settings:
        raise ValueError("Remove seeds: the component sweep now uses one gmm_seed")
    seed = settings["gmm_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("gmm_seed must be an integer in [0, 2**32)")
    # The reviewed K comes from config/scene_taxonomy.yaml (None until reviewed).
    if (settings.get("nominal_components") is not None
            and settings["nominal_components"] not in settings["components"]):
        raise ValueError("The selected scene-taxonomy K is not among the diagnosed components")
    if settings.get("maximum_vza_deg") != 70:
        raise ValueError("Nominal full ABI eligibility requires maximum_vza_deg: 70")
    if settings["spatial"]["native_resolution_km"] != 2:
        raise ValueError("Full ABI inputs must be native 2 km preprocessed files")


def source_path(day):
    return Path(f"data/preprocessed_files/abi_{day}_res2km_step1.nc")


def cache_path(settings, day):
    return Path(settings["cache_root"]) / str(day)


def spectral_model_path(settings):
    """Return the shared classifier path, defaulting to the legacy layout for callers."""
    return Path(settings.get(
        "model_path", Path(settings["output_dir"]) / "spectral_gmm.joblib"
    ))


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
        "provenance": runner.provenance([source_path(day)], stage="features",
                                        identity={"day": day}),
        "scope": "Every eligible native two-satellite overlap pixel; no texture or tile screening",
    })


def open_features(settings, day):
    path = cache_path(settings, day)
    index = json.loads((path / "index.json").read_text())
    freshness = check_nominal_stage_provenance(
        index["provenance"], "features", settings, {"day": day},
        root=Path(__file__).resolve().parents[1])
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
        ], stage="pca"),
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


def training_options(settings, workers, *, mapped=False):
    options = {"workers": workers, "chunk_rows": settings.get("gmm_chunk_points", 65536)}
    directory = Path(settings["output_dir"]) / "training_pca"
    # Direct Python callers may fit without a prepared cache; the DAG requires it.
    if directory.exists():
        record = json.loads((directory / "index.json").read_text())
        freshness = check_nominal_stage_provenance(
            record["provenance"], "pca", settings, root=Path(__file__).resolve().parents[1])
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

        if mapped:
            from streaming_gmm import MappedTrainingData
            source = MappedTrainingData(
                [directory / f"{day}.npy" for day in settings["training_days"]],
                settings["batch_points"], record["dimensions"],
            )
            if source.rows != record["pixels"]:
                raise ValueError("Mapped PCA cache population does not match preprocessing")
            options.update(preprocessor=preprocessor, mapped_data=source)
        else:
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
    if components not in settings["components"] or seed != settings["gmm_seed"]:
        raise ValueError("GMM candidate is not configured")
    path = Path(settings["output_dir"]) / f"candidate_k{components}_s{seed}.joblib"
    runner.verify(path, stage="candidate", identity=candidate_identity(settings, components, seed),
                  settings=settings)
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
    if components not in settings["components"] or seed != settings["gmm_seed"]:
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
        "provenance": runner.provenance(
            _initialization_inputs(settings), stage="candidate",
            identity=candidate_identity(settings, components, seed)),
        "components": components, "seed": seed,
        "scope": "Every eligible held-in pixel in scaler/PCA and each EM iteration",
    })


def candidate_initialization_path(settings, components, seed, initialization):
    return (Path(settings["output_dir"]) / "candidate_initializations"
            / f"candidate_k{components}_s{seed}_i{initialization}.joblib")


def _initialization_inputs(settings):
    return [
        *[cache_path(settings, day) for day in settings["training_days"]],
        *([Path(settings["output_dir"]) / "training_pca"]
          if (Path(settings["output_dir"]) / "training_pca").exists() else []),
    ]


def _atomic_output(path, operation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                            delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        operation(temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def fit_candidate_initialization(settings, components, seed, initialization, runner, workers=1,
                                 *, statistics_scheduler=None, provenance=None,
                                 event_callback=None):
    from streaming_gmm import fit_streaming_gmm

    if components not in settings["components"] or seed != settings["gmm_seed"]:
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
            statistics_scheduler=statistics_scheduler,
            event_callback=event_callback,
            **training_options(settings, workers, mapped=True),
        )
    except BrokenProcessPool:
        raise
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
    _atomic_output(output, lambda path: joblib.dump(record, path))
    metadata = {
        "provenance": (runner.provenance(
            _initialization_inputs(settings), stage="gmm",
            identity=initialization_identity(components, seed, initialization))
                       if provenance is None else provenance),
        "model_sha256": sha256_file(output),
        "components": components, "seed": seed, "initialization": initialization,
        "status": record["status"],
        "scope": "Every eligible held-in pixel in scaler/PCA and each EM iteration",
    }
    _atomic_output(str(output) + ".json", lambda path: runner.write(path, metadata))
    return record["status"]


def fit_initialization_sweep(settings, runner, cpus):
    """Fit expensive candidates first, automatically sharing the CPU allocation.

    Per-initialization files are durable restart records, not Snakemake outputs:
    rerunning an incomplete sweep must not delete already completed fits.
    The final manifest is published only after every configured start completes.
    """
    from gmm_scheduler import SweepProgress, run_gmm_jobs

    jobs = [
        (components, settings["gmm_seed"], initialization)
        for components in sorted(settings["components"], reverse=True)
        for initialization in range(settings["gmm_initializations"])
    ]
    pending, paths = [], []
    provenance_by_job = {}
    for components, seed, initialization in jobs:
        job = (components, seed, initialization)
        provenance = runner.provenance(
            _initialization_inputs(settings), stage="gmm",
            identity=initialization_identity(components, seed, initialization))
        provenance_by_job[job] = provenance
        path = candidate_initialization_path(settings, components, seed, initialization)
        paths.extend((path, Path(str(path) + ".json")))
        metadata_path = Path(str(path) + ".json")
        reusable = False
        if path.is_file() and metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text())
            except json.JSONDecodeError as error:
                print(f"Rebuilding incomplete GMM metadata {metadata_path}: {error}", flush=True)
            else:
                reusable = isinstance(metadata, dict) and (
                    metadata.get("provenance") == provenance
                    and metadata.get("model_sha256") == sha256_file(path)
                    and (metadata.get("components"), metadata.get("seed"),
                         metadata.get("initialization")) == (components, seed, initialization)
                    and metadata.get("status") in ("converged", "failed")
                )
        if reusable:
            print(f"Reusing completed GMM initialization {path}", flush=True)
        else:
            print(f"Scheduling new, stale or incomplete GMM initialization {path}", flush=True)
            pending.append((components, seed, initialization))

    def operation(job, scheduler):
        progress.emit({"event": "fit_started", "job": job})
        status = fit_candidate_initialization(
            settings, *job, runner, statistics_scheduler=scheduler,
            provenance=provenance_by_job[job],
            event_callback=lambda event: progress.emit({**event, "job": job}),
        )
        progress.emit({"event": "fit_completed" if status == "converged" else "fit_failed",
                       "job": job, "status": status})

    print(f"GMM sweep: {len(pending)} pending, {cpus} allocated CPUs; "
          "one coordinator CPU where possible, up to 32 compute workers per fit", flush=True)
    with SweepProgress(Path(settings["output_dir"]) / "sweep_logs",
                       pending, reused=len(jobs) - len(pending)) as progress:
        run_gmm_jobs(pending, operation, cpus)
        manifest = Path(settings["output_dir"]) / "initialization_sweep.json"
        _atomic_output(manifest, lambda path: runner.write(path, {
            "provenance": runner.provenance(
                [*_initialization_inputs(settings), *paths], stage="sweep",
                identity=sweep_identity(settings)),
            "initializations": [str(path) for path in paths[::2]],
            "allocated_cpus": cpus,
            "scope": "All configured starts; every training pixel in every EM pass",
        }))


def combine_candidate_initializations(settings, components, seed, runner):
    if components not in settings["components"] or seed != settings["gmm_seed"]:
        raise ValueError("GMM candidate is not configured")
    manifest = Path(settings["output_dir"]) / "initialization_sweep.json"
    if manifest.exists():
        freshness = check_nominal_stage_provenance(
            json.loads(manifest.read_text())["provenance"], "sweep", settings,
            sweep_identity(settings), root=Path(__file__).resolve().parents[1])
        if freshness["status"] != "current":
            raise ValueError(f"Stale GMM sweep {manifest}: {freshness}")

    scores = np.full(settings["gmm_initializations"], np.nan)
    failures = []
    best_model, best_score = None, -np.inf
    initialization_paths = [
        candidate_initialization_path(settings, components, seed, index)
        for index in range(settings["gmm_initializations"])
    ]
    for initialization, path in enumerate(initialization_paths):
        runner.verify(path, stage="gmm",
                      identity=initialization_identity(components, seed, initialization),
                      settings=settings)
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
        ], stage="candidate", identity=candidate_identity(settings, components, seed)),
        "components": components, "seed": seed,
        "scope": "Every eligible held-in pixel in scaler/PCA and each EM iteration",
    })


def candidates_path(settings):
    return Path(settings["diagnostic_dir"]) / "gmm_candidates.json"


def summarize_candidates(settings, runner):
    """K-independent scores for one reproducible candidate per component count."""
    output = Path(settings["output_dir"])
    seed = settings["gmm_seed"]
    rows, best = [], {}
    for count in settings["components"]:
        path = output / f"candidate_k{count}_s{seed}.joblib"
        runner.verify(path, stage="candidate",
                      identity=candidate_identity(settings, count, seed), settings=settings)
        score_path = output / "candidate_scores" / f"k{count}_s{seed}.json"
        if not score_path.exists():
            score_candidate(settings, count, seed, runner)
        score = read_record(score_path, runner)
        if score["components"] != count or score["seed"] != seed:
            raise ValueError(f"Candidate score metadata does not match {score_path}")
        row = {key: score[key] for key in (
            "components", "seed", "heldout_mean_log_likelihood",
            "heldout_day_log_likelihood", "heldout_counts", "minimum_occupancy",
        )}
        row["population_by_day"] = score["population_by_day"]
        if min(row["heldout_counts"]) < 100:
            raise ValueError(f"Component count {count} has no adequately occupied candidate")
        rows.append(row)
        best[str(count)] = row
        print(f"Full ABI spectral GMM k={count}, seed={seed}, "
              f"likelihood={row['heldout_mean_log_likelihood']:.6f}", flush=True)
    runner.write(candidates_path(settings), {
        "provenance": runner.provenance([
            *[output / f"candidate_k{count}_s{seed}.joblib"
              for count in settings["components"]],
            *[output / "candidate_scores" / f"k{count}_s{seed}.json"
              for count in settings["components"]]]),
        "selection": "Fixed gmm_seed, best training-likelihood start per K; K is chosen manually",
        "candidates": [{key: value for key, value in row.items() if key != "population_by_day"}
                       for row in rows],
        "best_by_components": best,
    })


def read_candidates(settings, runner):
    record = read_record(candidates_path(settings), runner)
    if sorted(int(count) for count in record["best_by_components"]) != sorted(settings["components"]):
        raise ValueError("GMM candidate summary does not cover the configured components")
    if (len(record["candidates"]) != len(settings["components"])
            or any(row["seed"] != settings["gmm_seed"]
                   for row in record["candidates"])
            or any(row["seed"] != settings["gmm_seed"]
                   for row in record["best_by_components"].values())):
        raise ValueError("GMM candidate summary must use the single configured gmm_seed")
    return record


def train(settings, runner):
    """Fit nothing new: promote the reviewed K's best candidate to the shared model."""
    from scene_taxonomy import load_scene_taxonomy, verify_selection
    taxonomy_file = Path(settings["scene_taxonomy_file"])
    gate = verify_selection(load_scene_taxonomy(taxonomy_file), runner.ROOT,
                            runner.SETTINGS, taxonomy_file)
    if gate["n_components"] != settings["nominal_components"]:
        raise ValueError("Scene taxonomy changed while loading nominal settings")
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    record = read_candidates(settings, runner)
    best = record["best_by_components"][str(gate["n_components"])]
    seed = best["seed"]
    model_path = output / f"candidate_k{gate['n_components']}_s{seed}.joblib"
    runner.verify(model_path, stage="candidate",
                  identity=candidate_identity(settings, gate["n_components"], seed),
                  settings=settings)
    model = joblib.load(model_path)
    counts = np.asarray(best["heldout_counts"], dtype=np.int64)
    by_day = best["population_by_day"]
    path = spectral_model_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
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
            *[cache_path(settings, day) for day in settings["validation_days"]],
            model_path, str(model_path) + ".json", candidates_path(settings),
            taxonomy_file, Path(gate["diagnostics_csv"])]),
        "components": gate["n_components"], "seed": seed,
        "selection": ("Reviewed manual component count from config/scene_taxonomy.yaml; "
                      "best full held-out likelihood initialization at that K"),
        "scene_taxonomy": gate,
        "candidates": record["candidates"], "heldout_population_counts": counts.tolist(),
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


def fit_library(settings, model, directory, cache_name="adm_fit_records.npy",
                days=None, form="regularized"):
    """Fit full-population paired-ratio least squares with a positive hemisphere.

For each scene s and channel c, minimise over every eligible pixel pair

    sum_i [ (1 + B16_i . p) / (1 + B18_i . p) - L16_i/L18_i ]^2 / n_s

with B the angular basis (P=1 regularized, P=2 quadratic). ``days`` defaults
to the training days. The disk-backed record cache uses (2P+7)*4 bytes per
pixel. All channels/scenes are optimized together; each objective/gradient
scans it once. The regularized form is bounded to a positive hemisphere; the
quadratic form is unbounded and its positivity is checked after the fit.
Returns library[K, 6, P].
"""
    days = settings["training_days"] if days is None else list(days)
    if form not in ("regularized", "quadratic"):
        raise ValueError(f"Unknown ADM form {form}")
    width = 1 if form == "regularized" else 2
    columns = 2 * width + 7
    k = model[-1].n_components
    n = sum(open_features(settings, day)[0]["pixels"] for day in days)
    path = Path(directory) / cache_name
    records = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, columns))
    counts = np.zeros(k, dtype=np.int64)
    angle_sum, angle_square = np.zeros(k), np.zeros(k)
    offset = 0
    for _, _, corrected, angles, _, features in paired_records(settings, days):
        labels = model.predict(features)
        basis = angular_basis(angles, form)  # n, satellite, P
        size = len(labels)
        records[offset:offset+size, :width] = basis[:, 0]
        records[offset:offset+size, width:2*width] = basis[:, 1]
        records[offset:offset+size, 2*width:2*width+6] = corrected[:, 0] / corrected[:, 1]
        records[offset:offset+size, -1] = labels
        offset += size
        counts += np.bincount(labels, minlength=k)
        difference = angles[:, 0] - angles[:, 1]
        angle_sum += np.bincount(labels, weights=difference, minlength=k)
        angle_square += np.bincount(labels, weights=difference**2, minlength=k)
    if offset != n:
        raise RuntimeError("Feature cache and ADM training eligibility differ")
    if np.any(counts < 100) or np.any(angle_square/counts - (angle_sum/counts)**2 < .01):
        raise ValueError("Full ABI scene ADM lacks 100 pairs or angular diversity")
    if form == "regularized":
        grid = angular_basis(np.linspace(0, 90, 901), "regularized")[:, 0]
        bounds = [(np.max((1e-6 - 1) / grid[grid > 0]),
                   np.min((1e-6 - 1) / grid[grid < 0]))] * (k * 6)
    else:
        bounds = None

    def objective(flat):
        parameters = flat.reshape(k, 6, width)
        value, gradient = 0., np.zeros((k, 6, width))
        for start in range(0, n, settings["batch_points"]):
            batch = np.asarray(records[start:start+settings["batch_points"]], dtype=np.float64)
            labels = batch[:, -1].astype(np.int64)
            b = parameters[labels]  # n, channel, P
            first, second = batch[:, None, :width], batch[:, None, width:2*width]
            numerator = 1 + np.sum(first * b, axis=-1)
            denominator = 1 + np.sum(second * b, axis=-1)
            residual = numerator / denominator - batch[:, 2*width:2*width+6]
            weight = residual / counts[labels, None]
            value += float(np.sum(residual * weight) / 2)
            for term in range(width):
                derivative = (first[..., term] * denominator
                              - second[..., term] * numerator) / denominator**2
                contributions = weight * derivative
                for channel in range(6):
                    gradient[:, channel, term] += np.bincount(
                        labels, weights=contributions[:, channel], minlength=k)
        return value, gradient.ravel()

    result = minimize(objective, np.zeros(k * 6 * width), jac=True, method="L-BFGS-B",
                      bounds=bounds, options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 300})
    if not result.success:
        raise RuntimeError(f"Full ABI scene ADM fit failed: {result.message}")
    library = result.x.reshape(k, 6, width)
    angular_integral(library, form)  # raises unless positive over the hemisphere
    del records
    path.unlink()
    return library


def adm_path(settings, components, form="regularized", day=None):
    """Pooled held-in library, its quadratic variant, or a per-day in-sample refit."""
    directory = Path(settings["output_dir"]) / "component_adms"
    if day is not None:
        if form != "regularized":
            raise ValueError("Per-day ADM libraries use the regularized form")
        return directory / "daily" / f"adm_k{components}_day{day}.joblib"
    suffix = "" if form == "regularized" else f"_{form}"
    return directory / f"adm_k{components}{suffix}.joblib"


def selected_candidate(settings, components, runner):
    """Best-seed candidate at K from the K-independent candidate summary."""
    best = read_candidates(settings, runner)["best_by_components"].get(str(components))
    if best is None:
        raise ValueError(f"Component count {components} has no adequately occupied candidate")
    path = Path(settings["output_dir"]) / f"candidate_k{components}_s{best['seed']}.joblib"
    runner.verify(path, stage="candidate",
                  identity=candidate_identity(settings, components, best["seed"]),
                  settings=settings)
    return best, joblib.load(path), path


def prepare_adm(settings, components, runner, form="regularized", day=None):
    if components not in settings["components"]:
        raise ValueError("GMM component count is not configured")
    all_days = settings["training_days"] + settings["validation_days"]
    if day is not None and day not in all_days:
        raise ValueError(f"Day {day} is not configured")
    days = settings["training_days"] if day is None else [day]
    best, model, model_path = selected_candidate(settings, components, runner)
    path = adm_path(settings, components, form, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_name = f"fit_k{components}_{form}" + ("" if day is None else f"_day{day}") + ".npy"
    library = fit_library(settings, model, path.parent, cache_name, days=days, form=form)
    joblib.dump(library, path)
    runner.write(str(path) + ".json", {
        "components": components, "seed": best["seed"], "form": form, "days": days,
        "scope": ("pooled held-in training days" if day is None
                  else "single-day in-sample refit (between-day/out-of-sample sensitivity)"),
        "provenance": runner.provenance([
            model_path, str(model_path) + ".json", candidates_path(settings),
            *[source_path(value) for value in days],
            *[cache_path(settings, value) / "index.json" for value in days],
        ]),
    })


def load_adm(settings, components, runner, form="regularized", day=None):
    path = adm_path(settings, components, form, day)
    if not path.exists():
        prepare_adm(settings, components, runner, form, day)
    runner.verify(path)
    metadata = json.loads(Path(str(path) + ".json").read_text())
    best, _, _ = selected_candidate(settings, components, runner)
    if (metadata["components"] != components or metadata["seed"] != best["seed"]
            or metadata.get("form", "regularized") != form):
        raise ValueError(f"ADM library does not match the selected candidate {path}")
    library = joblib.load(path)
    width = 1 if form == "regularized" else 2
    if library.shape != (components, 6, width) or not np.isfinite(library).all():
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
    model_path = spectral_model_path(settings)
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
            *sources, spectral_model_path(settings),
            Path(str(spectral_model_path(settings)) + ".json"),
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
    path = spectral_model_path(settings)
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
    path = spectral_model_path(settings)
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


def full_training_statistics(settings, model):
    """Evaluate final-model likelihood and entropy in bounded training chunks.

    All eligible training pixels enter ICL; the usual independent-sample
    interpretation of its BIC penalty does not account for spatial dependence.
    """
    options = training_options(settings, workers=1)
    if "transformed_batches" in options:
        _, scaler, pca = options["preprocessor"]
        if (not np.array_equal(scaler.mean_, model[0].mean_)
                or not np.array_equal(scaler.scale_, model[0].scale_)
                or not np.array_equal(pca.components_, model[1].components_)
                or not np.array_equal(pca.mean_, model[1].mean_)):
            raise ValueError("Training PCA cache differs from the candidate preprocessing")
        batches = options["transformed_batches"]()
    else:
        batches = (
            model[1].transform(model[0].transform(features.astype(np.float64, copy=False)))
            for features in feature_batches(settings, settings["training_days"])
        )
    count, log_likelihood, entropy = 0, 0., 0.
    for transformed in batches:
        scores, log_resp = model[-1]._estimate_log_prob_resp(transformed)
        if not np.isfinite(scores).all() or not np.isfinite(log_resp).all():
            raise ValueError("Non-finite training likelihood or responsibilities")
        count += len(transformed)
        log_likelihood += float(scores.sum())
        entropy -= float(np.sum(np.exp(log_resp) * log_resp))
    if count < 2:
        raise ValueError("Full-training ICL requires at least two eligible pixels")
    k, dimensions = model[-1].n_components, model[-1].means_.shape[1]
    parameters = k * (dimensions + dimensions * (dimensions + 1) // 2) + k - 1
    bic = -2 * log_likelihood + parameters * np.log(count)
    return {
        "points": count, "log_likelihood": log_likelihood, "entropy": entropy,
        "parameters": parameters, "bic": float(bic), "icl": float(bic + 2 * entropy),
    }


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
        bootstrap = cluster_stability_details(
            stability_train, stability_test, stability_baseline_labels,
            k, True, .98, stability_seed,
            settings.get("stability_repeats", 3),
        )
        ari_scores = [repeat["ari"] for repeat in bootstrap]
        ari_min, ari_std = float(np.min(ari_scores)), float(np.std(ari_scores))
        training_statistics = full_training_statistics(settings, model)
        ratio_square, scene_counts = np.zeros((k, 6)), np.zeros(k, dtype=np.int64)
        bins = np.zeros((k, 2, 7), dtype=np.int64)
        paired_bins = np.zeros((k, 7, 7), dtype=np.int64)
        ratio_sensitivity_square = np.zeros((k, 6))
        entropy_sum, ambiguous_count = 0., 0
        by_day = {}
        raw_flux, corrected_flux = Moments(), Moments()
        for day, _, corrected, angles, planck, features in paired_records(settings, settings["validation_days"]):
            scores, log_probabilities = model[-1]._estimate_log_prob_resp(
                model[1].transform(model[0].transform(features))
            )
            if not np.isfinite(scores).all() or not np.isfinite(log_probabilities).all():
                raise ValueError(f"K={k}, day={day}: non-finite held-out GMM statistics")
            labels = log_probabilities.argmax(axis=1)
            scene_counts += np.bincount(labels, minlength=k)
            entropy_sum -= float(np.sum(np.exp(log_probabilities) * log_probabilities))
            ambiguous_count += int(np.count_nonzero(log_probabilities.max(axis=1) < np.log(.8)))
            if day not in by_day:
                by_day[day] = {"raw": Moments(), "corrected": Moments(),
                               "counts": np.zeros(k, dtype=np.int64)}
            by_day[day]["counts"] += np.bincount(labels, minlength=k)
            basis = angular_basis(angles, "regularized")[..., 0]
            b = library[labels, :, 0]
            residual = (1 + basis[:, :1]*b)/(1 + basis[:, 1:2]*b) - corrected[:, 0]/corrected[:, 1]
            sensitivity = (basis[:, :1] - basis[:, 1:2]) / (1 + basis[:, 1:2]*b)**2
            for channel in range(6):
                ratio_square[:, channel] += np.bincount(
                    labels, weights=residual[:, channel]**2, minlength=k
                )
                ratio_sensitivity_square[:, channel] += np.bincount(
                    labels, weights=sensitivity[:, channel]**2, minlength=k,
                )
            angular_bins = np.minimum((angles / 10).astype(int), 6)
            for satellite in range(2):
                bins[:, satellite] += np.bincount(
                    labels*7 + angular_bins[:, satellite], minlength=k*7,
                ).reshape(k, 7)
            paired_bins += np.bincount(
                labels*49 + angular_bins[:, 0]*7 + angular_bins[:, 1], minlength=k*49,
            ).reshape(k, 7, 7)
            uncorrected = abi_broadband(corrected, angles, planck, labels,
                                       np.zeros_like(library), "regularized", coefficients)
            retrieved = abi_broadband(corrected, angles, planck, labels,
                                     library, "regularized", coefficients)
            raw_difference = uncorrected[:, 0]-uncorrected[:, 1]
            corrected_difference = retrieved[:, 0]-retrieved[:, 1]
            raw_flux.update(raw_difference)
            corrected_flux.update(corrected_difference)
            by_day[day]["raw"].update(raw_difference)
            by_day[day]["corrected"].update(corrected_difference)
        n = int(scene_counts.sum())
        if np.any(scene_counts == 0):
            raise ValueError(f"K={k}: held-out ADM diagnostics contain an unoccupied scene")
        rmse = float(np.sqrt(ratio_square.sum() / (n*6)))
        worst_channel = float(np.sqrt(ratio_square.sum(axis=0) / n).max())
        worst_scene = float(np.sqrt(ratio_square.sum(axis=1) / (scene_counts*6)).max())
        spread, raw_spread = corrected_flux.report()["scene_sd_w_m2"], raw_flux.report()["scene_sd_w_m2"]
        gain = 1 - spread/raw_spread if raw_spread > 0 else 0.
        pooled_bins = bins.sum(axis=1)
        coverage = float((pooled_bins > 0).mean(axis=1).min())
        effective_bins = pooled_bins.sum(axis=1)**2 / np.square(pooled_bins.astype(float)).sum(axis=1)
        ratio_sensitivity = np.sqrt(ratio_sensitivity_square / scene_counts[:, None])
        corrected_report = corrected_flux.report()
        details = {
            "training": training_statistics,
            "bootstrap": bootstrap,
            "by_day": {
                str(day): {
                    "points": moments["corrected"].count,
                    "scene_counts": moments["counts"].tolist(),
                    "raw_bias": moments["raw"].report()["bias_w_m2"],
                    "raw_sd": moments["raw"].report()["scene_sd_w_m2"],
                    "raw_rmse": moments["raw"].report()["ensemble_rmse_w_m2"],
                    "bias": moments["corrected"].report()["bias_w_m2"],
                    "sd": moments["corrected"].report()["scene_sd_w_m2"],
                    "rmse": moments["corrected"].report()["ensemble_rmse_w_m2"],
                } for day, moments in by_day.items()
            },
            "angular_bin_counts": bins.tolist(),
            "paired_angular_bin_counts": paired_bins.tolist(),
            "scene_ratio_sensitivity_rms": ratio_sensitivity.tolist(),
        }
        row = {
            "n_components": k, "heldout_loglik": heldout_loglik,
            "icl": diagnostic_icl,
            "full_training_icl": training_statistics["icl"],
            "full_training_bic": training_statistics["bic"],
            "heldout_normalized_entropy": entropy_sum / (n * np.log(k)),
            "heldout_ambiguous_fraction": ambiguous_count / n,
            "minimum_effective_angular_bins": float(effective_bins.min()),
            "minimum_ratio_sensitivity_rms": float(ratio_sensitivity.min()),
            "minimum_day_occupancy": min(
                float(day["counts"].min() / day["corrected"].count)
                for day in by_day.values()
            ),
            "exact_corrected_flux_bias": corrected_report["bias_w_m2"],
            "exact_corrected_flux_rmse": corrected_report["ensemble_rmse_w_m2"],
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
            "components": k, "row": row, "details": details,
            "provenance": runner.provenance([
                model_path, str(model_path) + ".json", adm_path(settings, k),
                str(adm_path(settings, k)) + ".json",
                stability_path, str(stability_path) + ".json",
                *[cache_path(settings, day) for day in settings["training_days"]],
                *([output / "training_pca"] if (output / "training_pca").exists() else []),
                *[source_path(day) for day in settings["validation_days"]],
                config["narrowband_to_broadband_coeffs_file"],
            ]),
        })


def combine_diagnostics(settings, runner):
    rows, sources, previous, component_details = [], [], None, {}
    candidates = read_candidates(settings, runner)
    for k in settings["components"]:
        path = Path(settings["diagnostic_dir"]) / "component_parts" / f"k{k}.json"
        record = read_record(path, runner)
        if record["components"] != k or record["row"]["n_components"] != k:
            raise ValueError(f"Diagnostic metadata does not match {path}")
        row = record["row"]
        if "details" in record:
            component_details[str(k)] = record["details"]
        # Full-population held-out criteria (every eligible held-out pixel).
        best = candidates["best_by_components"][str(k)]
        row["full_heldout_loglik"] = best["heldout_mean_log_likelihood"]
        row["full_minimum_occupancy"] = best["minimum_occupancy"]
        row["gmm_seed"] = best["seed"]
        if "details" in record:
            if len(best["heldout_day_log_likelihood"]) != len(settings["validation_days"]):
                raise ValueError(f"K={k}: held-out likelihood day count differs from settings")
            component_details[str(k)]["heldout_day_log_likelihood"] = dict(zip(
                map(str, settings["validation_days"]), best["heldout_day_log_likelihood"],
            ))
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
    if component_details and len(component_details) != len(rows):
        raise ValueError("Mixed diagnostic schemas: regenerate every component part")
    path = Path(settings["diagnostic_dir"]) / "gmm_component_diagnostics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    runner.write(str(path)+".json", {
        "diagnostic_schema_version": 2 if component_details else 1,
        "component_details": component_details,
        "provenance": runner.provenance([
            *sources,
            candidates_path(settings),
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
                  "Seven 10-degree bins cover 0-70 degrees; exactly 70 belongs to the final bin.",
                  "Full-training ICL uses nominal pixel count; spatial dependence is not corrected.",
                  "Ambiguous assignment means maximum posterior <0.8; not a selection threshold.",
                  "Angular ratio sensitivity is RMS d(L16/L18)/db, not parameter uncertainty.",
                  "Flux diagnostics evaluate ABI with Sunny-calibrated N2B; not absolute accuracy."],
    })


def run(stage, settings, config, runner, day=None, components=None, seed=None,
        initialization=None, workers=1, block=None, form="regularized"):
    if stage == "cache":
        if day is None:
            raise ValueError("cache requires --day")
        prepare_cache(settings, day, runner)
    elif stage == "train":
        train(settings, runner)
    elif stage == "candidates":
        summarize_candidates(settings, runner)
    elif stage == "candidate":
        if components is None or seed is None:
            raise ValueError("candidate requires --components and --seed")
        fit_candidate(settings, components, seed, runner, workers)
    elif stage == "candidate-init":
        if components is None or seed is None or initialization is None:
            raise ValueError("candidate-init requires --components, --seed and --initialization")
        fit_candidate_initialization(settings, components, seed, initialization, runner, workers)
    elif stage == "candidate-sweep":
        fit_initialization_sweep(settings, runner, workers)
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
        prepare_adm(settings, components, runner, form, day)
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
