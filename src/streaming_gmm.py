"""Full-data scaler, PCA and Gaussian-mixture fitting with bounded memory.

Only initialization uses randomly selected rows (one center per component).
Neither preprocessing nor EM is fitted on a sample: all rows contribute to
the merged covariance and to every EM expectation/maximization step. Inputs
must be immutable and replayable for the duration of fitting.
"""

from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
import multiprocessing
from numbers import Integral, Real
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.extmath import stable_cumsum, svd_flip
from threadpoolctl import threadpool_limits


_FEATURES = 10
_CHUNK_ROWS = 4096
_REG_COVAR = 1e-6


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _chunk_rows(chunk_rows):
    return _positive_integer(
        "chunk_rows", _CHUNK_ROWS if chunk_rows is None else chunk_rows,
    )


def _chunks(batches, chunk_rows=None, dimensions=_FEATURES):
    """Validate and convert only bounded slices, including mmap inputs."""
    chunk_rows = _chunk_rows(chunk_rows)
    for batch in batches():
        array = np.asarray(batch)
        if array.ndim != 2 or array.shape[1] != dimensions:
            raise ValueError(f"Each batch must have shape (n, {dimensions}).")
        if array.dtype.kind not in "fiu":
            raise ValueError("Feature arrays must contain real numeric values.")
        for start in range(0, len(array), chunk_rows):
            chunk = np.asarray(array[start:start + chunk_rows], dtype=np.float64)
            if not np.isfinite(chunk).all():
                raise ValueError("Feature arrays must contain only finite values.")
            yield chunk


def _merge_moments(mass, mean, scatter, new_mass, new_mean, new_scatter):
    """Merge centered (possibly weighted) scatter without raw second moments."""
    if new_mass == 0:
        return mass, mean, scatter
    if mass == 0:
        return new_mass, new_mean.copy(), new_scatter.copy()
    total = mass + new_mass
    delta = new_mean - mean
    scatter = scatter + new_scatter + np.outer(delta, delta) * (
        mass * (new_mass / total)
    )
    mean = mean + delta * (new_mass / total)
    return total, mean, scatter


def _stream_moments(batches, chunk_rows=None):
    count = 0
    mean = np.zeros(_FEATURES)
    scatter = np.zeros((_FEATURES, _FEATURES))
    for chunk in _chunks(batches, chunk_rows):
        # Anchor the reduction to avoid summing large absolute offsets.
        new_mean = chunk[0] + np.mean(chunk - chunk[0], axis=0)
        centered = chunk - new_mean
        count, mean, scatter = _merge_moments(
            count, mean, scatter, len(chunk), new_mean, centered.T @ centered
        )
    if count < 2:
        raise ValueError("At least two training rows are required.")
    if not np.isfinite(scatter).all() or not np.isfinite(mean).all():
        raise ValueError("Feature moments overflowed; rescale the input values.")
    return count, mean, scatter


def _preprocessing(count, mean, scatter):
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.var_ = np.maximum(np.diag(scatter) / count, 0.0)
    eps = np.finfo(np.float64).eps
    constant = scaler.var_ <= (
        count * eps * scaler.var_ + (count * mean * eps) ** 2
    )
    scaler.scale_ = np.sqrt(scaler.var_)
    scaler.scale_[constant] = 1.0
    scaler.n_features_in_ = _FEATURES
    scaler.n_samples_seen_ = np.int64(count)

    covariance = scatter / scaler.scale_[:, None] / scaler.scale_[None, :]
    covariance /= count - 1
    values, vectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    # Dense full-SVD PCA exposes min(n_samples, n_features) singular values.
    available = min(count, _FEATURES)
    values = np.maximum(values[::-1][:available], 0.0)
    _, components = svd_flip(None, vectors[:, ::-1].T, u_based_decision=False)
    total_variance = values.sum()
    if not np.isfinite(total_variance) or total_variance <= 0:
        raise ValueError("PCA requires positive, finite total feature variance.")
    ratios = values / total_variance
    retained = int(np.searchsorted(stable_cumsum(ratios), 0.98, side="right") + 1)
    retained = min(retained, available)
    pca = PCA(n_components=0.98, svd_solver="full")
    pca.mean_ = np.zeros(_FEATURES)
    pca.n_features_in_ = _FEATURES
    pca.n_samples_ = count
    pca.n_components_ = retained
    pca.components_ = components[:retained].copy()
    pca.explained_variance_ = values[:retained].copy()
    pca.explained_variance_ratio_ = ratios[:retained].copy()
    pca.singular_values_ = np.sqrt(values[:retained] * (count - 1))
    pca.noise_variance_ = float(values[retained:].mean()) if retained < available else 0.0
    pca._fit_svd_solver = "full"
    return scaler, pca


def prepare_streaming_preprocessing(batches, *, chunk_rows=None):
    """Return reusable ``(count, scaler, pca)`` fitted once on all raw rows.

    The caller may persist this tuple and float64 transformed mmap batches.
    No feature population is retained here. ``chunk_rows=None`` preserves the
    legacy 4096-row bound; 65536 is a useful explicit performance setting.
    """
    if not callable(batches):
        raise TypeError("batches must be a callable returning a fresh iterable.")
    count, mean, scatter = _stream_moments(batches, chunk_rows)
    scaler, pca = _preprocessing(count, mean, scatter)
    return count, scaler, pca


def _validate_preprocessor(preprocessor):
    """Check fitted float64 preprocessing metadata, not cache provenance."""
    if not isinstance(preprocessor, (tuple, list)) or len(preprocessor) != 3:
        raise ValueError("preprocessor must be (count, scaler, pca).")
    count, scaler, pca = preprocessor
    count = _positive_integer("preprocessor count", count)
    if count < 2:
        raise ValueError("At least two training rows are required.")
    if not isinstance(scaler, StandardScaler) or not isinstance(pca, PCA):
        raise ValueError("preprocessor must contain a fitted StandardScaler and PCA.")
    dimensions = getattr(pca, "n_components_", None)
    if (isinstance(dimensions, bool) or not isinstance(dimensions, Integral)
            or not 1 <= dimensions <= min(count, _FEATURES)):
        raise ValueError("preprocessor has invalid retained PCA dimensions.")
    for obj, name, shape in (
        (scaler, "mean_", (_FEATURES,)), (scaler, "var_", (_FEATURES,)),
        (scaler, "scale_", (_FEATURES,)), (pca, "mean_", (_FEATURES,)),
        (pca, "components_", (dimensions, _FEATURES)),
        (pca, "explained_variance_", (dimensions,)),
        (pca, "explained_variance_ratio_", (dimensions,)),
        (pca, "singular_values_", (dimensions,)),
    ):
        value = getattr(obj, name, None)
        if (not isinstance(value, np.ndarray) or value.shape != shape
                or value.dtype != np.dtype(np.float64) or not np.isfinite(value).all()):
            raise ValueError(f"preprocessor {name} must be finite float64 with shape {shape}.")
    if (getattr(scaler, "n_features_in_", None) != _FEATURES
            or getattr(pca, "n_features_in_", None) != _FEATURES
            or np.ndim(getattr(scaler, "n_samples_seen_", None)) != 0
            or getattr(scaler, "n_samples_seen_", None) != count
            or getattr(pca, "n_samples_", None) != count
            or not scaler.with_mean or not scaler.with_std or pca.whiten
            or np.any(scaler.scale_ <= 0) or np.any(scaler.var_ < 0)
            or np.any(pca.explained_variance_ < 0)
            or pca.explained_variance_.sum() <= 0):
        raise ValueError("preprocessor has inconsistent fitted metadata.")
    return count, scaler, pca


def _transformed_chunks(batches, scaler, pca, expected_count, chunk_rows=None):
    count = 0
    for chunk in _chunks(batches, chunk_rows):
        count += len(chunk)
        yield pca.transform(scaler.transform(chunk))
    if count != expected_count:
        raise ValueError(
            f"Batch factory must replay the same rows: expected {expected_count}, got {count}."
        )


def _training_chunks(batches, scaler, pca, count, chunk_rows, transformed_batches):
    if transformed_batches is None:
        yield from _transformed_chunks(batches, scaler, pca, count, chunk_rows)
        return
    seen = 0
    for chunk in _chunks(transformed_batches, chunk_rows, pca.n_components_):
        seen += len(chunk)
        yield chunk
    if seen != count:
        raise ValueError(
            f"Batch factory must replay the same rows: expected {count}, got {seen}."
        )


def _initial_centers(
    batches, scaler, pca, count, components, random_state,
    chunk_rows=None, transformed_batches=None,
):
    # Floyd sampling uses O(components) storage, unlike choice's permutation
    # of all training indices. Selection is independent of input partitioning.
    selected = set()
    for upper in range(count - components, count):
        index = int(random_state.randint(upper + 1))
        selected.add(upper if index in selected else index)
    indices = np.array(sorted(selected), dtype=np.int64)
    centers = np.empty((components, pca.n_components_))
    offset = 0
    for chunk in _training_chunks(
        batches, scaler, pca, count, chunk_rows, transformed_batches,
    ):
        lo = np.searchsorted(indices, offset)
        hi = np.searchsorted(indices, offset + len(chunk))
        centers[lo:hi] = chunk[indices[lo:hi] - offset]
        offset += len(chunk)
    return centers


def _set_parameters(model, weights, means, covariances):
    if not all(np.isfinite(array).all() for array in (weights, means, covariances)):
        raise RuntimeError("Non-finite Gaussian-mixture parameters.")
    model.weights_ = weights
    model.means_ = means
    model.covariances_ = covariances
    try:
        model.precisions_cholesky_ = _compute_precision_cholesky(covariances, "full")
    except ValueError as error:
        raise RuntimeError("Gaussian-mixture covariance factorization failed.") from error
    model.precisions_ = (
        model.precisions_cholesky_
        @ model.precisions_cholesky_.transpose(0, 2, 1)
    )


def _worker_initialize():
    """Prevent nested BLAS oversubscription in explicitly spawned workers."""
    threadpool_limits(limits=1)


def _chunk_statistics(chunk, model):
    """Worker entry point: no population, factory or inherited model globals."""
    components, dimensions = model.means_.shape
    masses = np.zeros(components)
    means = np.zeros((components, dimensions))
    scatters = np.zeros((components, dimensions, dimensions))
    log_prob, log_responsibilities = model._estimate_log_prob_resp(chunk)
    if (not np.isfinite(log_prob).all()
            or np.isnan(log_responsibilities).any()
            or np.isposinf(log_responsibilities).any()):
        raise RuntimeError("Non-finite Gaussian-mixture likelihood.")
    np.exp(log_responsibilities, out=log_responsibilities)
    shifted = chunk - chunk[0]
    for component in range(components):
        weights = log_responsibilities[:, component]
        mass = float(weights.sum())
        if mass == 0:
            continue  # A locally absent component can be globally valid.
        mean = chunk[0] + (weights @ shifted) / mass
        centered = chunk - mean
        masses[component] = mass
        means[component] = mean
        scatters[component] = centered.T @ (centered * weights[:, None])
    if not all(np.isfinite(value).all() for value in (masses, means, scatters)):
        raise RuntimeError("Non-finite Gaussian-mixture sufficient statistics.")
    return masses, means, scatters, float(log_prob.sum())


def _bounded_statistics(chunks, model, executor, workers):
    """At most workers submitted chunks; consume in input order, not finish order.

    Pickling is limited to one owned float64 chunk and O(K*d**2) model metadata
    per task. Copy before submission prevents asynchronous serialization from
    observing a reused producer buffer. Neither factories nor populations are
    sent to processes. Persistent processes are reused for all EM passes.
    """
    if executor is None:
        for chunk in chunks:
            yield _chunk_statistics(chunk, model)
        return
    pending = deque()
    chunks = iter(chunks)
    try:
        for _ in range(workers):
            chunk = next(chunks, None)
            if chunk is None:
                break
            pending.append(executor.submit(_chunk_statistics, chunk.copy(), model))
        while pending:
            yield pending.popleft().result()
            chunk = next(chunks, None)
            if chunk is not None:
                pending.append(executor.submit(_chunk_statistics, chunk.copy(), model))
    finally:
        for future in pending:
            future.cancel()
        # Drain running tasks before a new start can mutate model parameters.
        if pending:
            wait(pending)


def _expectation_statistics(
    batches, scaler, pca, model, count, chunk_rows=None,
    transformed_batches=None, executor=None, workers=1,
):
    """Return weighted counts, means, centered scatters and average log density.

    Responsibilities exist only for one bounded chunk, never for all rows.
    Scatters have shape (components, retained_dimensions, retained_dimensions).
    """
    components, dimensions = model.means_.shape
    masses = np.zeros(components)
    means = np.zeros((components, dimensions))
    scatters = np.zeros((components, dimensions, dimensions))
    log_likelihood = 0.0
    chunks = _training_chunks(
        batches, scaler, pca, count, chunk_rows, transformed_batches,
    )
    for new_masses, new_means, new_scatters, likelihood in _bounded_statistics(
        chunks, model, executor, workers,
    ):
        log_likelihood += likelihood
        for component in range(components):
            masses[component], means[component], scatters[component] = _merge_moments(
                masses[component], means[component], scatters[component],
                new_masses[component], new_means[component], new_scatters[component],
            )
    objective = log_likelihood / count
    if not np.isfinite(objective):
        raise RuntimeError("Non-finite Gaussian-mixture likelihood.")
    _validate_masses(masses, count)
    return masses, means, scatters, objective


def _validate_masses(masses, count):
    if not np.isfinite(masses).all() or np.any(
        masses <= 10 * np.finfo(np.float64).eps * count
    ):
        raise RuntimeError("Gaussian-mixture component collapsed (negligible mass).")


def _maximization(model, masses, means, scatters, count):
    _validate_masses(masses, count)
    covariances = scatters / masses[:, None, None]
    covariances = (covariances + covariances.transpose(0, 2, 1)) * 0.5
    dimensions = means.shape[1]
    covariances[:, np.arange(dimensions), np.arange(dimensions)] += _REG_COVAR
    _set_parameters(model, masses / masses.sum(), means, covariances)


def fit_streaming_gmm(
    batches, components, seed, initializations=2, max_iter=300, tol=1e-3,
    initialization_index=None, progress_callback=None,
    *, chunk_rows=None, workers=1, preprocessor=None, transformed_batches=None,
):
    """Fit a replayable (n, 10) feature stream and return an sklearn Pipeline.

    Features are unit-agnostic; all arithmetic uses float64. ``batches`` is a
    callable returning a fresh iterable of immutable finite feature arrays;
    empty batches are allowed, and arrays may be disk-backed memory maps.
    Internally, even large supplied arrays are processed in bounded slices.
    ``chunk_rows=None`` preserves the legacy 4096-row bound; explicitly choosing
    65536 often reduces overhead. ``workers=1`` runs in the calling process.
    Higher positive values use that many persistent spawn processes, each with
    one BLAS thread; the caller must budget memory/CPUs and guard script entry
    points with ``if __name__ == "__main__"``.

    ``preprocessor=(count, scaler, pca)`` reuses float64 fitted metadata returned
    by :func:`prepare_streaming_preprocessing`, skipping its raw-data pass.
    ``transformed_batches`` optionally supplies a replayable callable of finite
    arrays with shape (n, pca.n_components_), e.g. float64 read-only mmap slices.
    It requires ``preprocessor`` and bypasses *all* raw feature reads/transforms.
    The caller verifies cache provenance and identical row order; this function
    validates dimensions, finite values and row counts but cannot prove contents
    match the raw features. The returned Pipeline still accepts raw 10-D rows.

    StandardScaler uses population variance. PCA retains the smallest number
    of components whose cumulative explained variance strictly exceeds 0.98,
    with sklearn's component signs. Covariance eigendecomposition is exact
    up to floating-point roundoff (degenerate eigenspaces need not have the
    same basis as a dense SVD).

    Independent starts select uniform random training rows ONLY as initial
    centers, with equal weights and the global PCA covariance. Every EM pass
    uses every row, full covariances and sklearn's default 1e-6 regularization.
    Convergence means absolute change in full-data mean log likelihood < tol;
    the returned lower_bound_ is the likelihood of the returned parameters.
    Failed starts emit RuntimeWarning; if none converges, RuntimeError is
    raised. The highest-likelihood converged start is returned.

    For N rows, K components and d <= 10 retained dimensions, each EM pass
    costs O(N K d**2); working memory is O(workers * chunk_rows * (10 + K)
    + workers * K d**2), independent of N, in addition to process-library overhead.
    Only bounded chunks are pickled, with at most workers tasks in flight;
    sufficient statistics are merged deterministically in input order.
    Data must be replayable and unchanged across passes;
    changed row counts are checked, but input contents are not checksummed.
    """
    workers = _positive_integer("workers", workers)
    chunk_rows = _chunk_rows(chunk_rows)
    context = (
        ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_initialize,
        ) if workers > 1 else nullcontext()
    )
    with context as executor:
        return _fit_streaming_gmm(
            batches, components, seed, initializations, max_iter, tol,
            initialization_index, progress_callback, chunk_rows, workers,
            preprocessor, transformed_batches, executor,
        )


def _fit_streaming_gmm(
    batches, components, seed, initializations, max_iter, tol,
    initialization_index, progress_callback, chunk_rows, workers,
    preprocessor, transformed_batches, executor,
):
    if not callable(batches):
        raise TypeError("batches must be a callable returning a fresh iterable.")
    for name, value in (
        ("components", components), ("initializations", initializations),
        ("max_iter", max_iter),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(tol, bool) or not isinstance(tol, Real) or not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a positive finite number.")
    if initialization_index is not None and (
            isinstance(initialization_index, bool)
            or not isinstance(initialization_index, Integral)
            or not 0 <= initialization_index < initializations):
        raise ValueError("initialization_index must be within the configured starts.")
    random_state = check_random_state(seed)
    if transformed_batches is not None:
        if preprocessor is None:
            raise ValueError("transformed_batches requires preprocessor.")
        if not callable(transformed_batches):
            raise TypeError("transformed_batches must be a callable returning a fresh iterable.")
    if preprocessor is None:
        count, scaler, pca = prepare_streaming_preprocessing(batches, chunk_rows=chunk_rows)
    else:
        count, scaler, pca = _validate_preprocessor(preprocessor)
    if components > count:
        raise ValueError("components cannot exceed the number of training rows.")
    seeds = random_state.randint(np.iinfo(np.int32).max, size=initializations)
    global_covariance = np.diag(pca.explained_variance_ * ((count - 1) / count))
    global_covariance.flat[::pca.n_components_ + 1] += _REG_COVAR
    if initialization_index is None:
        start_indices = range(initializations)
    else:
        start_indices = (int(initialization_index),)
    best = None
    scores = np.full(initializations, np.nan)
    failures = []
    for start in start_indices:
        start_seed = seeds[start]
        model = GaussianMixture(
            n_components=components, covariance_type="full", random_state=int(start_seed),
            n_init=1, max_iter=max_iter, tol=tol, reg_covar=_REG_COVAR,
            init_params="random_from_data",
        )
        model.n_features_in_ = pca.n_components_
        centers = _initial_centers(
            batches, scaler, pca, count, components, check_random_state(int(start_seed)),
            chunk_rows, transformed_batches,
        )
        try:
            _set_parameters(
                model, np.full(components, 1.0 / components), centers,
                np.repeat(global_covariance[None, :, :], components, axis=0),
            )
            masses, means, scatters, previous = _expectation_statistics(
                batches, scaler, pca, model, count,
                chunk_rows, transformed_batches, executor, workers,
            )
            model.lower_bounds_ = []
            for iteration in range(1, max_iter + 1):
                _maximization(model, masses, means, scatters, count)
                masses, means, scatters, objective = _expectation_statistics(
                    batches, scaler, pca, model, count,
                    chunk_rows, transformed_batches, executor, workers,
                )
                model.lower_bounds_.append(objective)
                converged = abs(objective - previous) < tol
                if (progress_callback is not None
                        and (iteration % 10 == 0 or converged or iteration == max_iter)):
                    progress_callback(start, iteration, objective, converged)
                if converged:
                    model.converged_ = True
                    model.n_iter_ = iteration
                    model.lower_bound_ = objective
                    break
                previous = objective
            else:
                raise RuntimeError(f"did not converge within {max_iter} EM iterations")
        except BrokenProcessPool:
            raise  # Infrastructure failures are not failed numerical starts.
        except RuntimeError as error:
            message = f"Streaming GMM start {start + 1} failed: {error}"
            failures.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            continue
        scores[start] = model.lower_bound_
        if best is None or model.lower_bound_ > best.lower_bound_:
            best = model
    if best is None:
        raise RuntimeError("No streaming GMM initialization converged. " + "; ".join(failures))
    best.n_init = initializations
    best.streaming_initialization_scores_ = scores
    best.streaming_initialization_failures_ = tuple(failures)
    return Pipeline([("scaler", scaler), ("pca", pca), ("gmm", best)])
