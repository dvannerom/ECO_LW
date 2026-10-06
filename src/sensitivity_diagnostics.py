"""Cluster-aware summaries and fit diagnostics for paired ABI sensitivities."""

import numpy as np

from sensitivity import angular_basis, summary


def fit_diagnostics(angles, radiance, parameters, form):
    """Diagnose pooled ADM[channel] fits using normalized ratio Jacobians."""
    parameters = np.asarray(parameters)
    grid = angular_basis(np.linspace(0, 90, 901), form)
    profile = 1 + grid @ parameters
    basis = angular_basis(angles, form)
    numerator = 1 + basis[:, 0] @ parameters
    denominator = 1 + basis[:, 1] @ parameters
    jacobian = (
        basis[:, 0] * denominator[:, None] - numerator[:, None] * basis[:, 1]
    ) / denominator[:, None]**2
    singular = np.linalg.svd(jacobian, compute_uv=False)
    norms = np.linalg.norm(jacobian, axis=0)
    if np.any(norms == 0) or singular[-1] <= 0:
        raise ValueError("Unidentifiable ADM diagnostic Jacobian")
    normalized = jacobian / norms
    ratio_error = numerator / denominator - radiance[:, 0] / radiance[:, 1]
    return {
        "minimum_profile": float(profile.min()),
        "positivity_boundary": bool(profile.min() < 1e-4),
        "jacobian_condition": float(singular[0] / singular[-1]),
        "normalized_jacobian_condition": float(np.linalg.cond(normalized)),
        "minimum_singular_value_per_sqrt_pair": float(singular[-1] / np.sqrt(len(angles))),
        "ratio_rmse": float(np.sqrt(np.mean(ratio_error**2))),
        "pairs": len(angles),
    }


def sufficient_statistics(values):
    """Keep satellites and within-tile pixels together for resampling."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or values.size == 0:
        raise ValueError("Cluster contains empty or non-finite residuals")
    return np.array([values.size, values.sum(), np.square(values).sum()], dtype=float)


def statistics_from_sums(sums):
    count, total, squared = np.asarray(sums, dtype=float)
    if count < 2:
        raise ValueError("Need two observations for sample SD")
    mean = total / count
    variance = max((squared - total**2 / count) / (count - 1), 0)
    return {"count": int(count), "bias": float(mean), "sd": float(np.sqrt(variance)),
            "rmse": float(np.sqrt(squared / count))}


def cluster_interval(values, days, tile_ids, replicates, confidence, seed):
    """Hierarchical day/tile percentile interval, keeping both satellites paired.

    Work is O(records + replicates*tiles); memory O(tiles + replicates), not
    O(replicates*records). This describes the empirical selected-day population.
    """
    values = np.asarray(values, dtype=float)
    days, tile_ids = np.asarray(days), np.asarray(tile_ids)
    if values.shape[0] != len(days) or len(days) != len(tile_ids):
        raise ValueError("Cluster IDs do not align with paired records")
    unique_days = np.unique(days)
    if unique_days.size < 2 or replicates < 2 or not 0 < confidence < 1:
        raise ValueError("Bootstrap needs multiple days/replicates and valid confidence")
    if not np.all(np.isfinite(values)):
        raise ValueError("Bootstrap residuals must be finite")
    row_values = values.reshape(len(days), -1)
    row_count = row_values.shape[1]
    grouped = []
    for day in unique_days:
        selected = days == day
        _, inverse = np.unique(tile_ids[selected], return_inverse=True)
        rows = row_values[selected]
        grouped.append(np.column_stack((
            np.bincount(inverse) * row_count,
            np.bincount(inverse, weights=rows.sum(axis=1)),
            np.bincount(inverse, weights=np.square(rows).sum(axis=1)),
        )))
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, 3))
    for iteration in range(replicates):
        total = np.zeros(3)
        for day_index in rng.integers(0, len(grouped), len(grouped)):
            tiles = grouped[day_index]
            total += tiles[rng.integers(0, len(tiles), len(tiles))].sum(axis=0)
        metrics = statistics_from_sums(total)
        draws[iteration] = [metrics[key] for key in ("bias", "sd", "rmse")]
    tail = (1 - confidence) / 2
    bounds = np.quantile(draws, [tail, 1 - tail], axis=0)
    return {
        "estimate": summary(values), "confidence_level": confidence,
        "intervals": {key: bounds[:, index].tolist()
                      for index, key in enumerate(("bias", "sd", "rmse"))},
        "day_count": len(unique_days), "tile_count": sum(len(group) for group in grouped),
        "method": "Hierarchical day/tile percentile bootstrap; satellites stay together",
        "caution": "Conditional on eligible sampled tiles and selected days; few-day exploratory interval",
    }


def dominant_scene(labels, valid, components, block):
    """Return modal native component and its pixel fraction per complete footprint."""
    height, width = valid.shape
    counts = np.stack([
        ((labels == component) & valid).reshape(
            height // block, block, width // block, block
        ).sum(axis=(1, 3))
        for component in range(components)
    ], axis=-1)
    total = counts.sum(axis=-1)
    scene = counts.argmax(axis=-1)
    fraction = counts.max(axis=-1) / np.maximum(total, 1)
    return scene, fraction


def stability_check(previous, current, relative_tolerance, absolute_tolerance):
    """Predeclared pragmatic threshold; agreement is not proof of convergence."""
    change = abs(current - previous)
    threshold = max(absolute_tolerance, relative_tolerance * abs(previous))
    return {"absolute_change_w_m2": change, "threshold_w_m2": threshold,
            "within_tolerance": bool(change <= threshold)}
