"""Physical closure, rank exclusions and nominal-model reuse of the geometry diagnostic."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from geometry_sensitivity import retrieve_geometry
from sensitivity import angular_basis, angular_integral


@pytest.mark.parametrize("form,parameters,angles", [
    ("regularized", [-.3], [0, 70]),
    ("regularized", [-.3], [50, 60]),
    ("quadratic", [-.3, .1], [0, 35, 70]),
    ("quadratic", [-.3, .1], [45, 50, 60]),
])
def test_geometry_without_observed_55_closes(form, parameters, angles):
    parameters = np.array(parameters)
    radiance = (80 * (1 + angular_basis(angles, form) @ parameters))[None, :, None]
    flux, diagnostic = retrieve_geometry(radiance, angles, form)
    np.testing.assert_allclose(flux, 80*np.pi*angular_integral(parameters, form), rtol=1e-8)
    assert diagnostic["constrained_scene_channels"] == 0
    assert diagnostic["degrees_of_freedom"] == 0


def test_geometry_rank_and_validity_fail_explicitly():
    with pytest.raises(ValueError, match="identify"):
        retrieve_geometry(np.ones((1, 2, 1)), [0, 70], "quadratic")
    with pytest.raises(ValueError):
        retrieve_geometry(np.ones((1, 2, 1)), [50, 50], "regularized")
    with pytest.raises(ValueError):
        retrieve_geometry(-np.ones((1, 2, 1)), [0, 70], "regularized")


def test_clustered_geometry_has_less_shape_information():
    _, wide = retrieve_geometry(np.ones((1, 3, 1)), [0, 35, 70], "quadratic")
    _, clustered = retrieve_geometry(np.ones((1, 3, 1)), [45, 50, 60], "quadratic")
    assert clustered["normalized_design_condition"] > wide["normalized_design_condition"]
    assert clustered["unconstrained_flux_noise_gain"] > wide["unconstrained_flux_noise_gain"]


def test_unphysical_fit_is_constrained():
    angles = np.array([0, 35, 70])
    radiance = (80*(1+4*angular_basis(angles, "regularized")[:, 0]))[None, :, None]
    flux, diagnostic = retrieve_geometry(radiance, angles, "regularized")
    assert flux.item() > 0
    assert diagnostic["constrained_scene_channels"] == 1


def test_reported_noise_gain_matches_flux_derivatives():
    angles = [0, 15, 35, 50, 70]
    radiance = np.full((1, len(angles), 1), 80.)
    baseline, diagnostic = retrieve_geometry(radiance, angles, "quadratic")
    derivatives = []
    for view in range(len(angles)):
        perturbed = radiance.copy()
        perturbed[0, view, 0] += 1e-3
        flux, _ = retrieve_geometry(perturbed, angles, "quadratic")
        derivatives.append(((flux-baseline)/1e-3).item())
    assert np.linalg.norm(derivatives) == pytest.approx(
        diagnostic["unconstrained_flux_noise_gain"], rel=1e-7
    )


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nominal_predict_uses_each_fold_global_model_on_its_test_rows():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    runner = load_script("extension_runner", "scripts/run_sensitivity_extensions.py")
    rng = np.random.default_rng(0)
    flux = rng.uniform(50, 100, (12, 3))
    reference = flux.sum(axis=1)
    folds = []
    for test in (np.arange(0, 6), np.arange(6, 12)):
        train = np.setdiff1d(np.arange(12), test)
        scaler = StandardScaler().fit(flux[train])
        poly = PolynomialFeatures(1)
        regression = LinearRegression().fit(poly.fit_transform(scaler.transform(flux[train])), reference[train])
        folds.append({"fold": len(folds), "test": test, "scaler": scaler, "poly": poly, "regression": regression})
    np.testing.assert_allclose(runner.nominal_predict(flux, folds), reference, rtol=1e-10)
    with pytest.raises(RuntimeError, match="every Sunny record"):
        runner.nominal_predict(flux, folds[:1])
    weights = np.full(4, .25)
    assert runner.weighted(np.array([1., -1, 1, -1]), weights) == {"bias": 0., "rmse": 1.}


def test_extension_plot_smoke(tmp_path):
    module = load_script("extension_plot", "plotting/plot_sensitivity_extensions.py")
    geometry = {
        "settings": {"geometry_sets": {"wide_2": [0, 70], "wide_3": [0, 35, 70]}},
        "spectral_only_truth_error_w_m2": {"rmse": 1},
        "rows": [{
            "geometry": name, "form": form, "status": "quantified", "noise_seed": seed,
            "truth_error_w_m2": {"rmse": 2},
            "diagnostics": {"normalized_design_condition": 10, "unconstrained_flux_noise_gain": 3},
        } for name in ("wide_2", "wide_3") for form in ("regularized", "quadratic")
           for seed in (None, 42)],
    }
    module.plot_reports(geometry, tmp_path)
    assert (tmp_path/"eco_geometry.png").stat().st_size > 1000
