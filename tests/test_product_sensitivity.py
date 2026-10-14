"""Production-aligned sensitivity helpers and the Sunny paired estimator."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures

import abi_product
import nominal_processing as nominal
import run_product_sensitivity as runner
from nominal_budget import fit_pooled_shape, library_flux, population_weights
from product_sensitivity import (
    GroupMoments, cell_sums, fit_n2bc_huber, n2b_flux, production_flux, vza_bins,
)
from sensitivity import angular_integral

PLANCK = np.broadcast_to([200000., 1300., 0.1, 0.999], (6, 4)).copy()


def test_group_moments_match_numpy_and_merge_in_any_order():
    rng = np.random.default_rng(0)
    groups, values = rng.integers(0, 4, 1000), rng.normal(3, 2, 1000)
    first, second = GroupMoments(5), GroupMoments(5)
    first.update(groups[:300], values[:300])
    second.update(groups[300:], values[300:])
    first.merge(second.count, second.mean, second.m2)
    restored = GroupMoments.from_state(first.state())
    rows = restored.summary(np.full(5, 10.))
    for group in range(4):
        selected = values[groups == group]
        assert rows[group]["count"] == len(selected)
        assert rows[group]["bias_w_m2"] == pytest.approx(selected.mean())
        assert rows[group]["sd_w_m2"] == pytest.approx(selected.std())
        assert rows[group]["rmse_percent"] == pytest.approx(np.sqrt(np.mean(selected**2)) * 10)
    assert rows[4] == {"count": 0}
    with pytest.raises(ValueError):
        first.update(np.array([5]), np.array([1.]))
    with pytest.raises(ValueError):
        first.update(np.array([0]), np.array([np.nan]))


def test_cell_sums_partial_edges_and_monthly_identity():
    rng = np.random.default_rng(1)
    values, valid = rng.normal(size=(7, 12)), rng.random((7, 12)) > .3
    sums, counts = cell_sums(values, valid, 5)
    assert sums.shape == counts.shape == (2, 3)
    assert counts[1, 2] == valid[5:, 10:].sum()
    assert sums[1, 2] == pytest.approx(values[5:, 10:][valid[5:, 10:]].sum())
    assert counts.sum() == valid.sum()


def test_vza_bins_out_of_range_bin():
    edges = [0, 10, 20, 70]
    angles = np.array([0., 9.9, 10., 69., 70., 75., -1., np.nan])
    np.testing.assert_array_equal(vza_bins(angles, edges), [0, 0, 1, 2, 2, 3, 3, 3])


def test_production_flux_rounds_bt_like_product_storage():
    rng = np.random.default_rng(2)
    n, k = 300, 3
    library = rng.uniform(-.25, .1, (k, 6, 1))
    labels = rng.integers(-1, k, n)
    angles = rng.uniform(0, 80, n)
    radiance = 200000. / np.expm1(1300. / rng.uniform(220, 300, (n, 6)))
    integral = angular_integral(library, "regularized")
    coefficients = np.zeros(83)
    coefficients[:6] = 1 / 6
    regression = (coefficients, 0., 3)
    bt, flux = production_flux(radiance, angles, labels, PLANCK, library, "regularized",
                               integral, regression, 70.)
    exact_bt, exact = abi_product.satellite_chain(radiance, angles, labels, PLANCK, library,
                                                  "regularized", integral, regression, 70.)
    np.testing.assert_array_equal(np.isnan(flux), np.isnan(exact))
    np.testing.assert_array_equal(bt, exact_bt.astype(np.float32).astype(np.float64))
    np.testing.assert_array_equal(flux, n2b_flux(bt, regression))
    assert np.isnan(flux[(labels < 0) | (angles > 70)]).all()


def test_huber_n2bc_resists_outliers():
    rng = np.random.default_rng(3)
    flux = rng.uniform(1, 10, (400, 6))
    reference = flux.sum(axis=1) + rng.normal(0, .01, 400)
    reference[:10] += 100.
    weights = np.full(400, 1 / 400)
    scaler, poly, regression = fit_n2bc_huber(flux, reference, weights, 2, 1.345, 100, 1e-8)
    prediction = regression.predict(poly.transform(scaler.transform(flux)))
    scaler, poly, regression = nominal.fit_n2bc(flux, reference, weights)
    least_squares = regression.predict(poly.transform(scaler.transform(flux)))
    truth = flux[10:].sum(axis=1)
    huber_error = np.sqrt(np.mean((prediction[10:] - truth)**2))
    assert huber_error < .2 * np.sqrt(np.mean((least_squares[10:] - truth)**2))


def sunny_data(seed=4, files=60, views=5):
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 60, views)
    amplitude = rng.uniform(5, 15, (files, 6))
    shape = 1 - .2 * (1 - np.cos(np.deg2rad(angles)))[None, :, None] * rng.uniform(.5, 1.5, (1, 1, 6))
    radiance = amplitude[:, None, :] * shape * (1 + rng.normal(0, .01, (files, views, 6)))
    band = amplitude * np.pi * rng.uniform(.97, 1.03, (files, 6))
    return {"eco_radiance": radiance, "angles": angles, "true_band_flux": band,
            "reference": band.sum(axis=1) * 1.1 + 5 + rng.standard_t(2, files),
            "groups": np.repeat(np.arange(files // 2), 2), "regimes": rng.integers(0, 2, files)}


def test_sunny_baseline_reproduces_nominal_calculate_clean_path():
    data = sunny_data()
    labels = np.arange(len(data["groups"])) % 2
    population = np.array([.4, .6])
    baseline = runner.sunny_predictions(data, labels, population, 3, None, {})
    expected = np.zeros(len(labels))
    for training, test in GroupKFold(3).split(labels, groups=data["groups"]):
        weights, _ = population_weights(labels[training], population)
        library = np.zeros((2, 6, 1))
        for scene in range(2):
            library[scene] = fit_pooled_shape(data["eco_radiance"][training[labels[training] == scene]],
                                              data["angles"])
        scaler, poly, regression = nominal.fit_n2bc(data["true_band_flux"][training],
                                                    data["reference"][training], weights)
        flux = library_flux(data["eco_radiance"][test], data["angles"], labels[test], library)
        expected[test] = regression.predict(poly.transform(scaler.transform(flux)))
    np.testing.assert_array_equal(baseline, expected)
    options = {"robust_huber_cutoff": 1.345, "robust_max_iterations": 100, "robust_tolerance": 1e-8}
    for variant in ("n2bc_stratified", "n2bc_robust", "n2bc_degree3", "adm_quadratic"):
        prediction = runner.sunny_predictions(data, labels, population, 3, variant, options)
        assert np.all(np.isfinite(prediction)) and not np.array_equal(prediction, baseline)


def test_sunny_unsupported_taxonomy_is_reported_not_raised():
    data = sunny_data()
    labels = np.zeros(len(data["groups"]), dtype=int)
    labels[:2] = 2  # one group: always held out together, never in training
    assert runner.sunny_predictions(data, labels, np.array([.5, .3, .2]), 3, None, {}) is None


def test_available_variants_respect_configured_components():
    settings = {"components": [2, 3, 4]}
    variants = ["components_minus_1", "components_plus_1", "adm_daily"]
    assert runner.available(variants, 4, settings) == ["components_minus_1", "adm_daily"]
    assert runner.available(variants, 2, settings) == ["components_plus_1", "adm_daily"]


class BinLabeler:
    """Picklable stand-in for a GMM pipeline: bins the first feature into K scenes."""

    def __init__(self, components):
        self.components = components

    def predict(self, features):
        edges = np.quantile(np.linspace(-3, 3, 101), np.linspace(0, 1, self.components + 1)[1:-1])
        return np.digitize(np.tanh(features[:, 0] / 50) * 3, edges).astype(np.int64)


def write_day(directory, day, rng, height=23, width=17):
    from netCDF4 import Dataset
    from product_sensitivity import slab_spectral_features
    (directory / "data/preprocessing/abi").mkdir(parents=True, exist_ok=True)
    (directory / "data/abi_proxy/products/scene_id").mkdir(parents=True, exist_ok=True)
    with Dataset(directory / f"data/preprocessing/abi/abi_{day}_res2km_step1.nc", "w") as target:
        for name, size in (("y", height), ("x", width), ("c", 9), ("p", 4)):
            target.createDimension(name, size)
        for name in ("lat_interp_grid", "lon_interp_grid"):
            target.createVariable(name, "f4", ("y", "x"))[:] = rng.uniform(-50, 50, (height, width))
        for sat in abi_product.SATELLITES:
            bt = rng.uniform(220, 300, (height, width, 9))
            radiance = PLANCK[0, 0] / np.expm1(PLANCK[0, 1] / (PLANCK[0, 2] + PLANCK[0, 3] * bt))
            radiance[0, :3, 2] = np.nan
            target.createVariable(f"planck_{sat}", "f8", ("c", "p"))[:] = np.broadcast_to(PLANCK[0], (9, 4))
            target.createVariable(f"rad_{sat}_interp_corr", "f4", ("y", "x", "c"))[:] = radiance
            target.createVariable(f"BT_{sat}_interp", "f4", ("y", "x", "c"))[:] = bt
            target.createVariable(f"lza_{sat}_interp_corr", "f4", ("y", "x"))[:] = rng.uniform(0, 80, (height, width))
    with Dataset(directory / f"data/preprocessing/abi/abi_{day}_res2km_step1.nc") as source:
        features, identified = slab_spectral_features(source, slice(0, height))
    labels = np.full(height * width, -1.)
    labels[identified] = BinLabeler(3).predict(features)
    labels[5] = -1.
    with Dataset(directory / f"data/abi_proxy/products/scene_id/scene_id_{day}_res2km_3comp.nc", "w") as target:
        target.createDimension("n", height * width)
        target.createVariable("labels", "i8", ("n",))[:] = labels.astype(np.int64)
        sorted_labels = labels.copy()
        sorted_labels[labels < 0] = np.nan
        target.createVariable("sorted_labels", "f4", ("n",))[:] = sorted_labels


def test_abi_day_closes_on_production_product_and_assembles_monthly(tmp_path, monkeypatch):
    import json
    import sys
    import joblib
    import narrowband_to_broadband
    import nominal_streaming
    import radiance_to_flux
    from types import SimpleNamespace

    rng = np.random.default_rng(11)
    monkeypatch.chdir(tmp_path)
    days = [245, 246]
    output = tmp_path / "nominal"
    (output / "component_adms/daily").mkdir(parents=True)
    joblib.dump({"model": BinLabeler(3), "population": np.full(3, 1 / 3)}, output / "spectral_gmm.joblib")
    paths = {}
    for count in (2, 3, 4):
        for form, width in (("regularized", 1), ("quadratic", 2)):
            path = nominal_streaming.adm_path({"output_dir": output}, count, form)
            joblib.dump(rng.uniform(-.2, .05, (count, 6, width)), path)
            path.with_name(path.name + ".json").write_text(json.dumps({"form": form}))
            paths[count, form, None] = path
    for day in days:
        path = nominal_streaming.adm_path({"output_dir": output}, 3, day=day)
        joblib.dump(rng.uniform(-.2, .05, (3, 6, 1)), path)
        path.with_name(path.name + ".json").write_text(json.dumps({"form": "regularized"}))
    coefficients = {}
    for form, degree in (("linear", 1), ("quadratic", 2), ("cubic", 3)):
        values = np.zeros(len(PolynomialFeatures(degree).fit(np.zeros((1, 6))).powers_) - 1)
        values[:6] = 1 / 6
        if degree > 1:
            values[6] = 1e-4
        coefficients[form] = {"coefficients": values.tolist(), "intercept": 1.}
    (tmp_path / "coeffs.json").write_text(json.dumps(coefficients))
    for day in days:
        write_day(tmp_path, day, rng)
        for module, argv in ((radiance_to_flux, ["--n-components", "3", "--adm-library",
                                                 str(paths[3, "regularized", None]), "--chunk-rows", "7"]),
                             (narrowband_to_broadband, ["--coefficients-file", "coeffs.json", "--chunk-rows", "6"])):
            monkeypatch.setattr(sys, "argv", ["x", "-d", str(day), *argv])
            module.main()

    monkeypatch.setattr(nominal, "verify", lambda path: None)
    monkeypatch.setattr(nominal_streaming, "selected_candidate",
                        lambda settings, count, runner_: ({}, BinLabeler(count), output / f"k{count}"))
    monkeypatch.setattr(runner, "provenance", lambda *args: {})
    monkeypatch.setattr(runner, "load_current", lambda path: json.loads(Path(path).read_text()))
    settings = {"nominal_components": 3, "components": [2, 3, 4], "training_days": [245],
                "validation_days": [246], "output_dir": str(output), "scenario": "test"}
    variants = list(runner.ABI_VARIANTS)
    sens = {"cell_km": 10, "chunk_rows": 10, "closure_tolerance_w_m2": .01, "label_mismatch_tolerance": 0.,
            "vza_bin_edges_deg": [0, 10, 20, 30, 40, 50, 60, 70], "abi_variants": variants,
            "scope": "test"}
    for day in days:
        args = SimpleNamespace(day=day, coefficients="coeffs.json", n2b_form="cubic", max_vza=70.,
                               settings=None, maps=f"maps_{day}.npz", output=f"day_{day}.json")
        runner.abi_day(args, sens, settings, {"resolution_km": 2})
    with pytest.raises(RuntimeError, match="does not close"):
        runner.abi_day(SimpleNamespace(**{**vars(args), "max_vza": 60., "output": "bad.json"}),
                       sens, settings, {"resolution_km": 2})
    part = json.loads(Path("day_245.json").read_text())
    assert part["split"] == "held_in" and part["variants"] == variants
    assert all(record["nan_mismatch"] == 0 and record["max_abs_w_m2"] <= .01
               for record in part["closure"].values())

    # n2b_quadratic paired mean against an independent recomputation from the product BTs.
    from broadband import load_regression_coefficients
    from netCDF4 import Dataset
    bt = np.stack([np.ma.filled(Dataset(f"data/abi_proxy/products/narrowband_flux/narrowband_flux_245_res2km_C{c}.nc")
                                .variables["BT_G18"][:].astype(np.float64), np.nan) for c in range(6)], -1)
    bt = bt.reshape(-1, 6)
    base = n2b_flux(bt, load_regression_coefficients("coeffs.json", "cubic"))
    quad = n2b_flux(bt, load_regression_coefficients("coeffs.json", "quadratic"))
    paired = np.isfinite(base) & np.isfinite(quad)
    state = part["stats"]["n2b_quadratic"]["G18"]["global"]
    assert state["count"][0] == paired.sum() > 100
    assert state["mean"][0] == pytest.approx(np.mean(quad[paired] - base[paired]), rel=1e-9)
    with np.load("maps_245.npz") as maps:
        assert maps["n2b_quadratic_G18_sum"].shape == (5, 4)
        assert maps["n2b_quadratic_G18_count"].sum() == paired.sum()

    classes = {"bias_w_m2": .1, "scene_sd_w_m2": .2, "class_rmse_w_m2": .3}
    Path("sunny.json").write_text(json.dumps({
        "scenario": "test", "folds": 5, "support": {}, "closure_max_abs_w_m2": 0.,
        "baseline_truth_error_w_m2": classes,
        "variants": {"n2bc_robust": {"status": "quantified", "paired_difference_w_m2": classes,
                                     "paired_difference_percent": classes, "truth_error_w_m2": classes,
                                     "label_change_fraction": 0.},
                     "adm_per_file": {"status": "unsupported", "reason": "test"}}}))
    args = SimpleNamespace(days=["day_245.json", "day_246.json"], maps=["maps_245.npz", "maps_246.npz"],
                           sunny="sunny.json", monthly_maps="monthly.npz", output="report.json",
                           n2b_form="cubic", max_vza=70., settings=None)
    monkeypatch.setattr(nominal, "SETTINGS", runner.ROOT / "config/gmm_component_selection.yaml")
    runner.assemble(args, sens, settings, {"resolution_km": 2})
    report = json.loads(Path("report.json").read_text())
    assert report["schema_version"] == 2
    variant = report["abi"]["variants"]["n2b_quadratic"]
    assert variant["splits"]["held_in"]["G18"]["pixel"]["count"] == paired.sum()
    assert len(variant["splits"]["all"]["G16"]["by_vza_bin"]) == 7
    with np.load("monthly.npz") as monthly, np.load("maps_245.npz") as first, np.load("maps_246.npz") as second:
        total = first["n2b_quadratic_G18_sum"].astype(float) + second["n2b_quadratic_G18_sum"]
        count = first["n2b_quadratic_G18_count"].astype(int) + second["n2b_quadratic_G18_count"]
        np.testing.assert_allclose(monthly["n2b_quadratic_G18"][count > 0], (total / count)[count > 0], rtol=1e-6)

    import importlib.util
    spec = importlib.util.spec_from_file_location("sensitivity_plot", runner.ROOT / "plotting/plot_uncertainty_sensitivity.py")
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)
    plotter.plot_summary(report, tmp_path / "summary.png")
    plotter.plot_maps(report, "monthly.npz", tmp_path / "maps.png")
    assert (tmp_path / "summary.png").stat().st_size > 1000
    assert (tmp_path / "maps.png").stat().st_size > 1000
