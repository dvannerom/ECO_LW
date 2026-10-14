"""Legacy presentation paths migrate without touching scientific inputs."""

from pathlib import Path

import pytest

from figure_layout import figure_path


@pytest.mark.parametrize("old,new", [
    ("figures/ADM/a.png", "figures/abi_proxy/diagnostics/adm/a.png"),
    ("figures/gmm_diagnostics/a.csv", "figures/nominal_production/diagnostics/gmm/production/a.csv"),
    ("figures/abi_sunny_comparison_smoke/a.png",
     "figures/nominal_production/diagnostics/abi_sunny/smoke/a.png"),
    ("figures/uncertainty/convergence/a.png",
     "figures/nominal_production/diagnostics/convergence/a.png"),
    ("figures/broadband_flux/a.png", "figures/abi_proxy/products/broadband_flux/a.png"),
    ("figures/broadband_flux/irradiance_fit_a.png",
     "figures/abi_proxy/diagnostics/n2bc/irradiance_fit_a.png"),
    ("figures/uncertainty/eco_uncertainty_variance_decomposition.png",
     "figures/nominal_production/diagnostics/uncertainty/eco_uncertainty_variance_decomposition.png"),
    ("figures/uncertainty/eco_spectral_k2_scatter.png",
     "figures/nominal_production/uncertainties/eco_spectral_assessment/eco_spectral_k2_scatter.png"),
    ("figures/uncertainty/nominal/a.png",
     "figures/nominal_production/uncertainties/nominal_budget/a.png"),
    ("figures/diagnostics/nominal/a.png",
     "figures/nominal_production/diagnostics/a.png"),
    ("figures/uncertainty/nominal_full/a.png",
     "figures/nominal_production/uncertainties/full_profile/a.png"),
    ("figures/uncertainty/sensitivity/a.png",
     "figures/nominal_production/uncertainties/sensitivity/a.png"),
    ("figures/monthly_flux/a.png", "figures/abi_proxy/products/monthly_flux/a.png"),
    ("figures/nominal_production/uncertainties/sensitivity/a.png",
     "figures/nominal_production/uncertainties/sensitivity/a.png"),
    ("custom/a.png", "custom/a.png"),
])
def test_relative_absolute_and_idempotent_paths(old, new):
    assert figure_path(old) == Path(new)
    assert figure_path(figure_path(old)) == Path(new)
    assert figure_path(Path("/workspace") / old) == Path("/workspace") / new
