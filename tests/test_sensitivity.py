"""Closure and invariants for the first-draft sensitivity workflow."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import simpson

from adm import radiance_linear
from sensitivity import (
    angular_basis, angular_integral, block_mean, fit_scene_adm, paired_summary,
    planck_derivative, read_settings, retrieve_sunny, summary, switch_labels,
    tile_features,
)
from scene_features import build_scene_features
from netcdf_io import write_dataset


def test_zero_change_closes():
    flux = np.array([100., 200., 300.])
    for unit in paired_summary(flux, flux).values():
        assert unit["bias"] == unit["sd"] == unit["rmse"] == 0
    with pytest.raises(ValueError):
        summary([1, np.nan])


def test_assignment_exact_reproducible_and_native_labels():
    posterior = np.tile([0.1, 0.2, 0.7], (101, 1))
    labels, selected = switch_labels(posterior, .1, 42)
    assert len(selected) == 10
    assert len(np.unique(selected)) == 10
    assert np.count_nonzero(labels == 1) == 10
    np.testing.assert_array_equal(labels, switch_labels(posterior, .1, 42)[0])
    assert np.all(switch_labels(posterior, 0, 42)[0] == 2)
    with pytest.raises(ValueError):
        switch_labels(posterior, 1.1, 42)


@pytest.mark.parametrize("form,parameters", [
    ("regularized", np.array([-.3])),
    ("quadratic", np.array([-.3, .1])),
])
def test_sunny_physical_flux_closure(form, parameters):
    angles = np.arange(0, 75, 5)
    shape = 1 + angular_basis(angles, form) @ parameters
    radiance = 80 * shape[None, :, None]
    flux = retrieve_sunny(radiance, angles, form)
    np.testing.assert_allclose(flux, 80 * np.pi * angular_integral(parameters, form), rtol=1e-10)
    assert angular_integral(np.zeros(len(parameters)), form) == pytest.approx(1, abs=1e-10)


def test_original_sunny_retrieval_is_preserved():
    from evaluate_eco_spectral_reconstruction import retrieve_narrowband_fluxes

    angles = np.arange(0, 75, 5)
    radiance = 80 * radiance_linear(angles, -.3)[None, :, None]
    original = retrieve_narrowband_fluxes(radiance, angles)[0]
    np.testing.assert_allclose(retrieve_sunny(radiance, angles, "regularized"),
                               original, rtol=1e-9)


@pytest.mark.parametrize("form", ["regularized", "quadratic"])
def test_unphysical_limb_extrapolation_is_constrained_and_reported(form):
    angles = np.arange(0, 75, 5)
    radiance = np.broadcast_to(
        80 * radiance_linear(angles, 4)[None, :, None], (3, len(angles), 2)
    )
    diagnostics = {}
    flux = retrieve_sunny(radiance, angles, form, diagnostics)
    assert np.all(np.isfinite(flux)) and np.all(flux > 0)
    assert diagnostics["positivity_constrained_scene_channels"] == 6


def test_planck_derivative_matches_finite_difference():
    wavelength = np.linspace(5, 16, 100)
    h, c, k = 6.62607015e-34, 299792458., 1.380649e-23

    def planck(t):
        lam = wavelength * 1e-6
        return 2 * h * c**2 / lam**5 / np.expm1(h * c / (lam * k * t)) * 1e-6

    np.testing.assert_allclose(planck_derivative(wavelength, 255),
                               (planck(255.001) - planck(254.999)) / .002, rtol=1e-8)


def test_block_average_complete_and_linear():
    data = np.arange(100).reshape(10, 10).astype(float)
    np.testing.assert_allclose(block_mean(data, 5),
                               [[22, 27], [72, 77]])
    assert block_mean(data, 5).mean() == data.mean()
    with pytest.raises(ValueError):
        block_mean(data[:9], 5)


def test_tile_features_match_production():
    rng = np.random.default_rng(2)
    bt = rng.normal(270, 5, (30, 35, 2, 6)).astype(np.float32)
    features, valid = tile_features(bt)
    cube16 = np.ones((30, 35, 9), dtype=np.float32)
    cube18 = cube16.copy()
    channels = [0, 3, 4, 6, 7, 8]
    cube16[:, :, channels] = bt[:, :, 0]
    cube18[:, :, channels] = bt[:, :, 1]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.nc"
        write_dataset(path, {
            "BT_G16_interp": cube16, "BT_G18_interp": cube18,
            "lat_interp_grid": np.zeros((30, 35)), "lon_interp_grid": np.zeros((30, 35)),
        }, {
            "BT_G16_interp": ("y", "x", "channel"), "BT_G18_interp": ("y", "x", "channel"),
            "lat_interp_grid": ("y", "x"), "lon_interp_grid": ("y", "x"),
        })
        production = build_scene_features(path)[0]
    np.testing.assert_allclose(features[valid], production, atol=3e-5)


def test_pooled_quadratic_adm_recovers_known_shape():
    angles = np.column_stack((np.linspace(5, 65, 300), np.linspace(65, 5, 300)))
    params = np.array([-.3, .05])
    radiance = 70 * (1 + angular_basis(angles, "quadratic") @ params)
    fitted = fit_scene_adm(angles, radiance, "quadratic")
    np.testing.assert_allclose(fitted, params, atol=2e-6)
    np.testing.assert_allclose(
        angular_basis(angles, "quadratic") @ fitted,
        angular_basis(angles, "quadratic") @ params, atol=1e-7,
    )
    np.testing.assert_allclose(
        angular_integral(fitted, "quadratic"),
        angular_integral(params, "quadratic"), atol=1e-7,
    )


def test_pooled_regularized_adm_is_positive_even_for_inadmissible_input_shape():
    angles = np.column_stack((np.linspace(5, 65, 300), np.linspace(65, 5, 300)))
    radiance = 70 * radiance_linear(angles, 4)
    fitted = fit_scene_adm(angles, radiance, "regularized")
    assert angular_integral(fitted, "regularized") > 0
    assert np.all(1 + angular_basis(np.linspace(0, 90, 901), "regularized") @ fitted > 0)

def test_settings_holdout_and_halo():
    settings = read_settings(Path(__file__).resolve().parents[1] / "config/sensitivity_run.yaml")
    assert not set(settings["training_days"]) & set(settings["evaluation_days"])
    assert settings["abi"]["halo"] >= 4 * settings["abi"]["block_size"]
