"""Nested sampling, clustered inference and pooled-shape diagnostic invariants."""

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run_sensitivity_convergence as runner
from sensitivity import angular_basis, summary
from sensitivity_diagnostics import (
    cluster_interval, dominant_scene, fit_diagnostics, stability_check,
    statistics_from_sums, sufficient_statistics,
)


def test_cluster_statistics_close_to_direct_summary():
    values = np.arange(24).reshape(12, 2) * .37 - 2
    sums = sum(sufficient_statistics(chunk) for chunk in np.array_split(values, 3))
    for key, value in summary(values).items():
        assert statistics_from_sums(sums)[key] == pytest.approx(value)


def test_cluster_bootstrap_pairs_satellites_and_is_reproducible():
    # Opposite satellite biases always cancel if each footprint stays paired.
    values = np.tile([3., -3.], (16, 1))
    days = np.repeat([1, 2, 3, 4], 4)
    tiles = np.tile([0, 0, 1, 1], 4)
    result = cluster_interval(values, days, tiles, 100, .95, 42)
    assert result == cluster_interval(values, days, tiles, 100, .95, 42)
    assert result["intervals"]["bias"] == [0, 0]
    assert result["intervals"]["rmse"] == [3, 3]
    assert result["tile_count"] == 8
    with pytest.raises(ValueError):
        cluster_interval(values, np.ones(16), tiles, 100, .95, 42)
    values[0, 0] = np.nan
    with pytest.raises(ValueError):
        cluster_interval(values, days, tiles, 100, .95, 42)


def test_day_cluster_spread_not_pixel_independence():
    days = np.repeat(np.arange(4), 100)
    values = np.repeat([0., 2., 4., 6.], 100)
    result = cluster_interval(values, days, np.zeros(400), 500, .95, 109)
    assert result["intervals"]["bias"][1] - result["intervals"]["bias"][0] > 2


def test_modal_scene_fraction_and_tie_rule():
    labels = np.array([[0, 1, 2, 2], [0, 1, 2, 1]])
    scene, fraction = dominant_scene(labels, np.ones_like(labels, dtype=bool), 3, 2)
    np.testing.assert_array_equal(scene, [[0, 2]])
    np.testing.assert_allclose(fraction, [[.5, .75]])


@pytest.mark.parametrize("form,parameters", [
    ("regularized", [-.3]), ("quadratic", [-.3, .1]),
])
def test_fit_diagnostics_exact_ratio_closure(form, parameters):
    angles = np.column_stack((np.linspace(2, 68, 1000), np.linspace(61, 10, 1000)))
    radiance = 10 * (1 + angular_basis(angles, form) @ np.asarray(parameters))
    diagnostics = fit_diagnostics(angles, radiance, parameters, form)
    assert diagnostics["ratio_rmse"] < 1e-12
    assert diagnostics["minimum_profile"] > 0
    assert diagnostics["normalized_jacobian_condition"] >= 1
    assert not diagnostics["positivity_boundary"]


def test_training_reservoir_bounded_nested_and_alignment(tmp_path, monkeypatch):
    # Fake cache with coordinates encoded in all arrays tests selected-row pairing.
    paths = []
    for day in range(2):
        path = tmp_path / str(day)
        path.mkdir()
        paths.append(path)
        values = np.arange(16).reshape(4, 2, 2) + day * 100
        arrays = {
            "features": values[..., None],
            "radiance": np.broadcast_to(values[..., None, None], (*values.shape, 2, 6)),
            "angles": np.broadcast_to(values[..., None], (*values.shape, 2)),
            "valid_blocks": np.ones((4, 2, 2), dtype=bool),
        }
        for name, value in arrays.items():
            np.save(path / f"{name}.npy", value)
        (path / "index.json").write_text(json.dumps({"tile_count": 4, "arrays": list(arrays)}))
    monkeypatch.setattr(runner, "verify", lambda metadata: None)
    large = runner.collect_training(paths, 4, 12, 1, 42)
    small = runner.collect_training(paths, 4, 6, 1, 42)
    np.testing.assert_array_equal(large["features"][:6], small["features"])
    np.testing.assert_array_equal(large["features"][:, 0], large["radiance"][:, 1, 5])
    np.testing.assert_array_equal(large["features"][:, 0], large["angles"][:, 1])
    changed = runner.collect_training(paths, 4, 12, 1, 73)
    assert not np.array_equal(large["features"], changed["features"])
    prefix = runner.collect_training(paths, 1, 6, 1, 42)["features"][:, 0]
    assert np.all((prefix < 4) | ((prefix >= 100) & (prefix < 104)))
    with pytest.raises(ValueError):
        runner.collect_training(paths, 1, 20, 1, 42)


def test_stability_threshold_is_explicit():
    assert stability_check(2, 2.15, .1, .1)["within_tolerance"]
    assert not stability_check(2, 2.3, .1, .1)["within_tolerance"]
    assert stability_check(.01, .10, .1, .1)["within_tolerance"]


def test_approved_configuration_is_disjoint():
    study, base = runner.configuration(Path("config/sensitivity_convergence.yaml"))
    assert not set(base["training_days"]) & set(base["evaluation_days"])
    assert study["tile_counts"] == [12, 48, 192]
    assert base["abi"]["tiles_per_day"] == 192


def test_plot_smoke(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "convergence_plot", Path("plotting/plot_sensitivity_convergence.py")
    )
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    study, _ = runner.configuration(Path("config/sensitivity_convergence.yaml"))
    metrics = {"w_m2": {"estimate": {"rmse": 1}, "intervals": {"rmse": [.8, 1.2]}}}
    cases = {
        (tiles, 20000, 42, seed) for tiles in study["tile_counts"]
        for seed in study["sampling_seeds"]
    } | {(192, points, 42, 42) for points in study["gmm_point_counts"]}
    cases |= {(192, 20000, seed, 42) for seed in study["initialization_seeds"]}
    runs = [{"training_tiles": tiles, "evaluation_tiles": tiles, "gmm_points": points,
             "initialization_seed": init, "sampling_seed": seed,
             "results": {name: metrics for name in plot.VARIANTS}}
            for tiles, points, init, seed in cases]
    rows = [{"status": "quantified", "label": "0", "pooled": {"rmse": 1}}]
    report = {"settings": study, "runs": runs, "bin_definitions": {},
              "regime_diagnostics": {
                  variant: {regime: rows for regime in (
                      "scene", "vza_g16", "vza_g18", "angular_separation",
                      "heterogeneity_K", "positivity_boundary", "day", "scene_purity",
                  )} for variant in plot.VARIANTS
              }}
    model = {
        "libraries": {"regularized": np.zeros((2, 6, 1)), "quadratic": np.zeros((2, 6, 2))},
        "diagnostics": {form: [[{
            "positivity_boundary": False, "normalized_jacobian_condition": 10,
        } for _ in range(6)] for _ in range(2)] for form in ("regularized", "quadratic")},
    }
    plot.plot_all(report, model, tmp_path)
    for name in ("convergence", "regime_diagnostics", "adm_profiles"):
        assert (tmp_path / f"{name}.png").stat().st_size > 1000


def test_report_axes_fixed_support_and_persistence(tmp_path, monkeypatch):
    study, _ = runner.configuration(Path("config/sensitivity_convergence.yaml"))
    study.update(tile_counts=[1, 2, 4], gmm_point_counts=[10, 20, 30],
                 fixed_gmm_points=10, bootstrap_replicates=20,
                 minimum_regime_footprints=2)
    cases = {(tiles, 10, 42, 42) for tiles in study["tile_counts"]}
    cases |= {(4, points, 42, 42) for points in study["gmm_point_counts"]}
    cases |= {(4, 10, init, 42) for init in study["initialization_seeds"]}
    cases |= {(4, 10, 42, seed) for seed in study["sampling_seeds"]}
    day = np.repeat(np.arange(4), 8)
    rank = np.tile(np.repeat(np.arange(4), 2), 4)
    ids = np.array([f"{d}:{t}:{index}" for index, (d, t) in enumerate(zip(day, rank))])
    baseline = np.full((32, 2), 200.)
    paths = []
    for tiles, points, init, seed in sorted(cases):
        path = tmp_path / f"t{tiles}_p{points}_i{init}_s{seed}.npz"
        metadata = {"training_tiles": tiles, "gmm_points": points,
                    "initialization_seed": init, "sampling_seed": seed}
        shifted = baseline + (tiles - 4) * .1 + (points - 10) * .01 + (init - 42) * .001
        arrays = {
            "baseline": shifted, "day": day, "tile_rank": rank,
            "tile_id": np.array([f"{d}:{t}" for d, t in zip(day, rank)]),
            "record_ids": ids, "scene": np.zeros(32),
            "scene_fraction": np.ones(32), "angles": np.full((32, 2), 50.),
            "heterogeneity": np.ones(32), "boundary_fraction": np.zeros(32),
            "condition": np.full(32, 25.),
            **{variant: shifted + delta for variant, delta in (
                ("spatial", 1), ("adm", 2), ("components", .5), ("assignment", .1),
            )},
        }
        np.savez_compressed(path, **arrays)
        Path(str(path) + ".json").write_text(json.dumps(metadata))
        paths.append(str(path))
    monkeypatch.setattr(runner, "verify", lambda metadata: None)
    monkeypatch.setattr(runner, "provenance", lambda args, inputs: {"test": True})
    args = SimpleNamespace(inputs=paths, output=tmp_path / "report.json")
    runner.summarize(args, study)
    report = json.loads(args.output.read_text())
    assert len(report["runs"]) == 15
    assert len(report["model_changes_on_fixed_evaluation"]) == 7
    assert len(report["stability_checks"]) == 40
    assert report["runs"][0]["results"]["adm"]["w_m2"]["estimate"]["rmse"] == 2
    assert {row["axis"] for row in report["stability_checks"]} == {
        "evaluation_sampling", "training_coverage", "gmm_points",
    }
    # Training variations must not compare equal-shaped but mismatched footprints.
    with np.load(paths[0]) as source:
        corrupted = {key: source[key] for key in source.files}
    corrupted["record_ids"] = ids[::-1]
    np.savez_compressed(paths[0], **corrupted)
    with pytest.raises(ValueError, match="fixed evaluation IDs"):
        runner.summarize(args, study)
