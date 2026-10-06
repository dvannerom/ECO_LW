"""Physical closure, rank exclusions and nested assignment controls."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from geometry_sensitivity import nested_switch_order, retrieve_geometry
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


def test_assignment_curve_has_exact_nested_counts_and_zero_closure():
    posterior = np.tile([.1, .2, .7], (101, 1))
    first, second, order = nested_switch_order(posterior, 42)
    assert np.all(first == 2) and np.all(second == 1)
    previous = set()
    for fraction in [0, .01, .05, .1, .2, .3, .4, .5]:
        selected = set(order[:int(fraction*len(first))])
        assert previous <= selected
        assert len(selected) == int(fraction*len(first))
        previous = selected
    np.testing.assert_array_equal(order, nested_switch_order(posterior, 42)[2])
    assert not np.array_equal(order, nested_switch_order(posterior, 73)[2])
    with pytest.raises(ValueError):
        nested_switch_order([[.1, .1]], 42)


def test_extension_plot_smoke(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "extension_plot", Path("plotting/plot_sensitivity_extensions.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
    assignment = {
        "settings": {"assignment_seeds": [42, 73, 109]},
        "rows": [{
            "seed": seed, "fraction": fraction,
            "w_m2": {"estimate": {"rmse": fraction, "bias": 0},
                     "intervals": {"rmse": [fraction, fraction], "bias": [0, 0]}},
        } for seed in (42, 73, 109) for fraction in (0, .01, .05, .1, .2, .3, .4, .5)],
    }
    module.plot_reports(geometry, assignment, tmp_path)
    assert (tmp_path/"eco_geometry.png").stat().st_size > 1000
    assert (tmp_path/"assignment_curve.png").stat().st_size > 1000
