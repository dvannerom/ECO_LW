"""Error-class identities and scene-weighted pooled retrieval invariants."""

import numpy as np
import pytest

from nominal_budget import (
    integrate_domain, file_scene, population_weights, fit_pooled_shape, library_flux, error_classes,
    independent_budget_sum, lognormal_radiance,
)
from sensitivity import angular_basis, angular_integral


def test_lognormal_positive_moment_matching_and_zero_noise():
    radiance = np.array([100., 1., .01])
    sd = np.array([.4, 1., .02])
    draws = np.random.default_rng(42).normal(size=(400000, 3))
    perturbed = lognormal_radiance(radiance, sd, draws)
    assert np.all(np.isfinite(perturbed) & (perturbed > 0))
    np.testing.assert_allclose(perturbed.mean(axis=0), radiance, rtol=.015)
    np.testing.assert_allclose(perturbed.std(axis=0), sd, rtol=.05)
    np.testing.assert_array_equal(lognormal_radiance(radiance, 0., draws),
                                  np.broadcast_to(radiance, draws.shape))
    # A draw that makes additive Gaussian radiance negative remains positive.
    assert 1. + 2. * -3. < 0
    assert lognormal_radiance(np.array([1.]), 2., np.array([-3.]))[0] > 0
    # At high SNR, the original additive Gaussian behavior is recovered.
    z = np.array([-2., 0., 2.])
    np.testing.assert_allclose(lognormal_radiance(100., .001, z), 100. + .001*z,
                               atol=2e-8, rtol=0)


@pytest.mark.parametrize("radiance,sd,draw", [
    (0., 1., 0.), (-1., 1., 0.), (np.nan, 1., 0.),
    (1., -1., 0.), (1., np.inf, 0.), (1., 1., np.nan),
])
def test_lognormal_invalid_inputs(radiance, sd, draw):
    with pytest.raises(ValueError, match="Lognormal noise requires"):
        lognormal_radiance(radiance, sd, draw)


@pytest.mark.parametrize("draw", [-1e6, 1e6])
def test_lognormal_numerical_range_fails_without_clipping(draw):
    with pytest.raises(ValueError, match="overflowed or underflowed"):
        lognormal_radiance(1., 1., draw)


def test_exact_domain_boundaries():
    wavelength = np.array([2., 5., 50., 110.])
    assert integrate_domain(wavelength, np.ones(4), 4, 100) == pytest.approx(96)
    with pytest.raises(ValueError):
        integrate_domain(wavelength, np.ones(4), 1, 100)


def test_population_weights_and_missing_support():
    weights, report = population_weights([0, 0, 1], [.6, .3, .1])
    np.testing.assert_allclose(weights, [1/3, 1/3, 1/3])
    assert report["covered_population"] == pytest.approx(.9)
    assert report["missing_scenes"] == [2]
    weights, _ = population_weights([0, 0, 1], [.8, .2])
    np.testing.assert_allclose(weights, [.4, .4, .2])


def test_file_majority_and_posterior_ties():
    posterior = np.array([[[.8, .2], [.4, .6]], [[.7, .3], [.9, .1]]])
    labels, agreement = file_scene(posterior)
    np.testing.assert_array_equal(labels, [0, 0])
    np.testing.assert_allclose(agreement, [.5, 1])


def test_pooled_shape_does_not_depend_on_file_brightness():
    angles = np.arange(0, 75, 5)
    profile = 1-.3*angular_basis(angles, "regularized")[:, 0]
    radiance = np.array([10., 100., 1000.])[:, None, None]*profile[None, :, None]
    fitted = fit_pooled_shape(radiance, angles)
    np.testing.assert_allclose(fitted, [[-.3]], atol=1e-7)
    flux = library_flux(radiance, angles, np.zeros(3, dtype=int), fitted[None])
    expected = np.pi*np.array([10., 100., 1000.])*angular_integral(fitted, "regularized").item()
    np.testing.assert_allclose(flux[:, 0], expected, rtol=1e-7)


def test_deterministic_error_classes_exact_identity():
    weights = np.array([.2, .3, .5])
    error = np.array([1., 2., -1.])[:, None]
    result = error_classes(error, weights)
    assert result["random_sd_w_m2"] == 0
    assert result["bias_w_m2"] == pytest.approx(.3)
    assert result["class_rmse_w_m2"] == pytest.approx(result["ensemble_rmse_w_m2"])


def test_noise_random_vs_scene_variance_correction():
    rng = np.random.default_rng(42)
    errors = np.array([1., 3., 5.])[:, None]+rng.normal(size=(3, 10000))
    result = error_classes(errors, np.full(3, 1/3))
    assert result["bias_w_m2"] == pytest.approx(3, abs=.03)
    assert result["random_sd_w_m2"] == pytest.approx(1, abs=.03)
    assert result["scene_sd_w_m2"] == pytest.approx(np.sqrt(8/3), abs=.03)
    with pytest.raises(ValueError):
        error_classes([[np.nan]], [1])


def test_independent_spatial_combination_adds_signed_bias_not_rmse():
    first = error_classes(np.array([1., 3.])[:, None], [.5, .5])
    second = error_classes(np.array([-2., 0.])[:, None], [.5, .5])
    result = independent_budget_sum(first, second)
    assert result["bias_w_m2"] == pytest.approx(1)
    assert result["scene_sd_w_m2"] == pytest.approx(np.sqrt(2))
    assert result["random_sd_w_m2"] == 0
    assert result["ensemble_rmse_w_m2"] == pytest.approx(np.sqrt(3))
    noise = dict(first, scene_sd_w_m2=0., random_sd_w_m2=1.)
    mixed = independent_budget_sum(noise, second)
    assert mixed["scene_sd_w_m2"] == pytest.approx(1)
    assert mixed["random_sd_w_m2"] == pytest.approx(1)
