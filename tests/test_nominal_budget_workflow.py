"""Small end-to-end validation of grouped source decomposition and plot output."""

import importlib.util
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

import nominal_processing as runner
from nominal_budget import integrate_domain
from sensitivity import angular_basis, angular_integral


def test_grouped_budget_closure_and_models(tmp_path, monkeypatch):
    n = 40
    angles = np.arange(0, 75, 5)
    labels = np.arange(n) % 2
    rng = np.random.default_rng(42)
    shapes = -.3+.02*rng.normal(size=(n, 6))
    amplitude = rng.uniform(20, 60, (n, 6))
    radiance = amplitude[:, None, :]*(1+angular_basis(angles, "regularized")[:, 0][None, :, None]*shapes[:, None, :])
    trueflux = np.pi*amplitude*angular_integral(shapes[..., None], "regularized")
    reference = trueflux.mean(axis=1)+10
    data = dict(
        eco_radiance=radiance, abi_radiance=np.ones_like(radiance),
        true_band_flux=trueflux, reference=reference, planck=np.ones((6, 4)),
        eco_noise_sd=np.full((n, 6), .002),
        groups=np.repeat(np.arange(n//2).astype(str), 2), regimes=np.where(labels, "cloudy", "clear"),
        angles=angles,
    )
    options = {
        "output_dir": str(tmp_path/"budget"), "diagnostic_dir": str(tmp_path/"diagnostics"),
        "noise_realizations": 4, "noise_seed": 42, "folds": 5,
        "reference_domain_um": [4, 100],
        "product_dir": str(tmp_path/"products"), "scenario": "rfma_goal_6",
        "figure_dir": str(tmp_path/"figures"), "diagnostic_figure_dir": str(tmp_path/"diagnostic_figures"),
        "spatial": {"nominal_block_size": 5, "independent_of_other_sources": True},
    }
    output = Path(options["output_dir"])
    output.mkdir()
    np.savez_compressed(output/"sunny.npz", **data)
    joblib.dump({"model": "test", "population": np.array([.7, .3])}, output/"spectral_gmm.joblib")
    Path(str(output/"spectral_gmm.joblib")+".json").write_text(json.dumps({"heldout_log_density_p01": -10}))
    (output/"abi_assignment.json").write_text(json.dumps({
        "provenance": {}, "transition_probabilities": [[.9, .1], [.2, .8]],
        "weighted_change_probability": .13, "transition_counts": [[90, 10], [20, 80]],
    }))
    spatial_row = {"block_size": 5, "resolution_km": 10, "footprints": 3,
                   **runner.error_classes(np.array([1., 2., 3.])[:, None], np.full(3, 1/3))}
    (output/"spatial_report.json").write_text(json.dumps({
        "provenance": {}, "nominal": spatial_row, "rows": [spatial_row],
        "by_day": {"246": {"5": spatial_row}},
    }))
    monkeypatch.setattr(runner, "verify", lambda path: None)
    monkeypatch.setattr(runner, "provenance", lambda inputs: {"test": True})
    monkeypatch.setattr(runner, "check_provenance", lambda metadata, root: {"status": "current"})
    monkeypatch.setattr(runner, "classify", lambda model, radiance, planck: (labels.copy(), np.ones(n)))
    monkeypatch.setattr(runner, "density", lambda model, radiance, planck: np.zeros(radiance.shape[:2]))
    runner.calculate(options)
    report = json.loads((output/"budget_report.json").read_text())
    assert report["support"]["covered_population"] == 1
    assert report["assignment_transfer"]["weighted_change_probability"] == .13
    assert report["assignment_transfer"]["included_in_total"]
    products = report["products"]["clean_heldout_flux"]
    assert products["narrowband_flux"]["npz"].startswith("narrowband_flux/")
    assert products["broadband_flux"]["npz"].startswith("broadband_flux/")
    assert "simulation/evaluation-only" in report["products"]["scope"]
    with np.load(output/"residuals.npz") as source:
        np.testing.assert_allclose(source["total"],
            source["spectral"]+source["angular"]+source["radiometric"]+source["assignment"], atol=1e-10)
        assert np.any(source["assignment"] != 0)
        covariance = report["covariances_w_m4"]
        moments = {key: np.mean(source[key], axis=1) @ source["weights"]
                   for key in ("spectral", "angular", "radiometric", "assignment", "total")}
        variances = {key: np.mean((source[key]-moments[key])**2, axis=1) @ source["weights"]
                     for key in moments}
        assert np.isclose(variances["total"],
            sum(variances[key] for key in ("spectral", "angular", "radiometric", "assignment"))
            +2*sum(covariance.values()))
    cv = joblib.load(output/"cv_models.joblib")
    seen = []
    narrowband_path = Path(options["product_dir"])/products["narrowband_flux"]["npz"]
    broadband_path = Path(options["product_dir"])/products["broadband_flux"]["npz"]
    with np.load(narrowband_path) as narrowband:
        source_case_folds = narrowband["fold_id"]
        for fold_id, fold in enumerate(cv):
            assert not set(fold["training_groups"]) & set(fold["test_groups"])
            seen.extend(fold["test"])
            assert np.all(source_case_folds[fold["test"]] == fold_id)
            assert not set(narrowband["groups"][fold["test"]]) & set(fold["training_groups"])
        assert narrowband["clean_narrowband_flux_w_m2"].shape == (n, 6)
        np.testing.assert_array_equal(narrowband["case_index"], np.arange(n))
        np.testing.assert_array_equal(narrowband["scene_label"], labels)
        np.testing.assert_array_equal(narrowband["groups"], data["groups"])
        np.testing.assert_array_equal(narrowband["regimes"], data["regimes"])
        np.testing.assert_array_equal(narrowband["reference_truth_w_m2"], reference)
        with np.load(output/"residuals.npz") as residual:
            np.testing.assert_allclose(narrowband["population_weight"], residual["weights"])
            expected_broadband = (
                reference+residual["spectral"][:, 0]+residual["angular"][:, 0]
            )
    np.testing.assert_array_equal(np.sort(seen), np.arange(n))
    with np.load(broadband_path) as broadband:
        np.testing.assert_allclose(
            broadband["predicted_clean_broadband_flux_w_m2"],
            expected_broadband,
        )
        assert np.all(np.isfinite(broadband["predicted_clean_broadband_flux_w_m2"]))
        assert np.all(broadband["predicted_clean_broadband_flux_w_m2"] > 0)
    with np.load(narrowband_path) as narrowband:
        assert np.all(np.isfinite(narrowband["clean_narrowband_flux_w_m2"]))
        assert np.all(narrowband["clean_narrowband_flux_w_m2"] > 0)
    for product_link, expected_type, flux_name, units in (
        (products["narrowband_flux"], "narrowband_flux", "clean_narrowband_flux_w_m2", "W m-2"),
        (products["broadband_flux"], "broadband_flux", "predicted_clean_broadband_flux_w_m2", "W m-2"),
    ):
        metadata = json.loads((Path(options["product_dir"])/product_link["metadata"]).read_text())
        assert metadata["product_type"] == expected_type
        assert metadata["scenario"] == options["scenario"]
        assert metadata["units"][flux_name] == units
        assert metadata["provenance"] == {"test": True}
        assert metadata["array_shapes"][flux_name][0] == n
        assert "simulation/evaluation-only" in metadata["scope"]
    assert np.all(source_case_folds >= 0)
    nominal = joblib.load(output/"nominal_models.joblib")
    assert nominal["present"].all()
    spec = importlib.util.spec_from_file_location("nominal_plot", Path("plotting/plot_nominal_budget.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.plot(report)
    assert not (output/"numerical_budget.csv").exists()
    from budget_reporting import write_nominal_budget_csv
    write_nominal_budget_csv(report, output/"numerical_budget.csv")
    diagnostic_csv = Path(options["diagnostic_dir"])/"gmm_component_diagnostics.csv"
    diagnostic_csv.parent.mkdir(parents=True, exist_ok=True)
    with diagnostic_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("n_components", "ari_min", "ari_std"))
        writer.writeheader()
        writer.writerows([
            {"n_components": k, "ari_min": .9, "ari_std": .02} for k in (4, 6, 8)
        ])
    module.plot_gmm_diagnostics({
        "components": 6,
        "candidates": [{"components": k, "seed": seed, "heldout_mean_log_likelihood": -1/k,
                        "heldout_day_log_likelihood": [-1/k, -.5/k], "minimum_occupancy": .05}
                       for k in (4, 6, 8) for seed in (42, 73)],
    }, diagnostic_csv, options)
    assert (Path(options["figure_dir"])/"eco_uncertainty_numerical_budget.png").stat().st_size > 1000
    assert (Path(options["diagnostic_figure_dir"])/"gmm_stability.png").stat().st_size > 1000
    assert (Path(options["diagnostic_figure_dir"])/"spatial_processing.png").stat().st_size > 1000
    assert (output/"numerical_budget.csv").exists()


def test_scene_transition_distribution():
    labels = np.repeat([0, 1], 5000)
    probabilities = [[.9, .1], [.2, .8]]
    samples = runner.draw_scene_transitions(labels, probabilities, 30, 42)
    assert np.mean(samples[:5000] == 1) == pytest.approx(.1, abs=.005)
    assert np.mean(samples[5000:] == 0) == pytest.approx(.2, abs=.005)
    np.testing.assert_array_equal(samples, runner.draw_scene_transitions(labels, probabilities, 30, 42))
    np.testing.assert_array_equal(runner.draw_scene_transitions(labels, np.eye(2), 3, 42),
                                  np.repeat(labels[:, None], 3, axis=1))
    with pytest.raises(ValueError, match="Invalid conditional"):
        runner.draw_scene_transitions(labels, [[1, -1], [0, 1]], 3, 42)


class ThresholdGMM:
    def predict(self, features):
        return (features[:, 0] > 250).astype(int)


def test_spatial_operator_identity_and_complete_footprints():
    planck = np.broadcast_to([200000., 1300., 0., 1.], (2, 6, 4)).copy()
    temperature = np.array([[240., 260., 245., 245.], [255., 255., 245., 245.]])
    raw = np.broadcast_to((200000./np.expm1(1300./temperature))[:, :, None, None],
                          (2, 4, 2, 6)).copy()
    angles = np.full((2, 4, 2), 55.)
    valid = np.ones((2, 4), dtype=bool)
    library = np.full((2, 6, 1), -.3)
    coefficients = np.zeros(83)
    coefficients[0] = .5
    coefficients = (coefficients, 100.)
    identity = runner.spatial_tile(ThresholdGMM(), raw, raw, angles, planck,
                                   library, coefficients, valid, 1)
    np.testing.assert_allclose(identity, 0, atol=1e-10)
    delta = runner.spatial_tile(ThresholdGMM(), raw, raw, angles, planck,
                               library, coefficients, valid, 2)
    assert len(delta) == 2
    assert abs(delta[0]) > .01
    assert abs(delta[1]) < 1e-9
    valid[0, 2] = False
    filtered = runner.spatial_tile(ThresholdGMM(), raw, raw, angles, planck,
                                  library, coefficients, valid, 2)
    np.testing.assert_allclose(filtered, delta[:1])
    with pytest.raises(ValueError, match="complete aligned"):
        runner.spatial_tile(ThresholdGMM(), raw, raw, angles, planck,
                            library, coefficients, valid, 3)


def test_spatial_tile_crops_edges_and_aligns_to_full_grid():
    planck = np.broadcast_to([200000., 1300., 0., 1.], (2, 6, 4)).copy()
    temperature = np.arange(35).reshape(5, 7)+240.
    raw = np.broadcast_to((200000./np.expm1(1300./temperature))[:, :, None, None],
                          (5, 7, 2, 6)).copy()
    angles = np.full((5, 7, 2), 55.)
    valid = np.ones((5, 7), dtype=bool)
    library = np.full((2, 6, 1), -.3)
    coeff = np.zeros(83)
    coeff[0] = .5
    coefficients = (coeff, 100.)
    result = runner.spatial_tile(ThresholdGMM(), raw, raw, angles, planck,
                                 library, coefficients, valid, 3, origin=(1, 2))
    # Global row 3 / column 3: local start (2,1), complete extent (3,6).
    region = (slice(2, 5), slice(1, 7))
    expected = runner.spatial_tile(ThresholdGMM(), raw[region], raw[region],
                                   angles[region], planck, library, coefficients,
                                   valid[region], 3)
    assert len(result) == 2
    np.testing.assert_allclose(result, expected)
    valid[2, 1] = False
    result = runner.spatial_tile(ThresholdGMM(), raw, raw, angles, planck,
                                 library, coefficients, valid, 3, origin=(1, 2))
    np.testing.assert_allclose(result, expected[1:])


def test_abi_noise_features_match_noise_free_brightness_temperature():
    planck = np.broadcast_to([200000., 1300., 0., 1.], (2, 6, 4)).copy()
    bt = np.where(np.indices((5, 5))[0] < 2, 249.9, 250.1)
    radiance = np.repeat((200000./np.expm1(1300./bt))[..., None], 6, axis=2)
    selected = np.stack([radiance]*2, axis=2)
    features = runner.abi_noise_features(selected.reshape(-1, 2, 6), planck, 0)
    np.testing.assert_allclose(features[:, 0], bt.ravel())


def test_linear_spectral_domain_integration():
    wavelength = np.array([2., 5., 50., 110.])
    value = 3*wavelength+2
    expected = 1.5*(100**2-4**2)+2*(100-4)
    assert np.isclose(integrate_domain(wavelength, value, 4, 100), expected)
