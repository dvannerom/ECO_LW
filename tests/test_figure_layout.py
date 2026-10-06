"""Legacy presentation paths migrate without touching scientific inputs."""

from pathlib import Path

import pytest

from figure_layout import figure_path


@pytest.mark.parametrize("old,new", [
    ("figures/ADM/a.png", "figures/diagnostics/adm/a.png"),
    ("figures/gmm_diagnostics/a.csv", "figures/diagnostics/gmm/production/a.csv"),
    ("figures/abi_sunny_comparison_smoke/a.png", "figures/diagnostics/abi_sunny/smoke/a.png"),
    ("figures/uncertainty/convergence/a.png", "figures/diagnostics/convergence/a.png"),
    ("figures/broadband_flux/a.png", "figures/products/broadband_flux/a.png"),
    ("figures/broadband_flux/irradiance_fit_a.png", "figures/diagnostics/n2bc/irradiance_fit_a.png"),
    ("figures/uncertainty/eco_uncertainty_numerical_budget.png",
     "figures/uncertainty/historical/first_draft/eco_uncertainty_numerical_budget.png"),
    ("figures/uncertainty/eco_uncertainty_evidence_map.png",
     "figures/uncertainty/registry/eco_uncertainty_evidence_map.png"),
    ("figures/uncertainty/nominal/a.png", "figures/uncertainty/nominal/a.png"),
    ("custom/a.png", "custom/a.png"),
])
def test_relative_absolute_and_idempotent_paths(old, new):
    assert figure_path(old) == Path(new)
    assert figure_path(figure_path(old)) == Path(new)
    assert figure_path(Path("/workspace") / old) == Path("/workspace") / new
