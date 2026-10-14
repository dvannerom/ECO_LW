"""Presentation-only relocation of legacy figure paths, including saved reports."""

from pathlib import Path


DIRECTORIES = {
    "figures/uncertainty/nominal_full": "figures/nominal_production/uncertainties/full_profile",
    "figures/uncertainty/nominal": "figures/nominal_production/uncertainties/nominal_budget",
    "figures/uncertainty/spectral_angular": "figures/nominal_production/uncertainties/eco_spectral_assessment",
    "figures/uncertainty/convergence": "figures/nominal_production/diagnostics/convergence",
    "figures/uncertainty/extensions": "figures/nominal_production/uncertainties/extensions",
    "figures/uncertainty/sensitivity": "figures/nominal_production/uncertainties/sensitivity",
    "figures/diagnostics/nominal_full": "figures/nominal_production/diagnostics/full_profile",
    "figures/diagnostics/nominal": "figures/nominal_production/diagnostics",
    "figures/diagnostics/gmm/production": "figures/nominal_production/diagnostics/gmm/production",
    "figures/diagnostics/abi_sunny/smoke": "figures/nominal_production/diagnostics/abi_sunny/smoke",
    "figures/diagnostics/abi_sunny/comparison": "figures/nominal_production/diagnostics/abi_sunny/comparison",
    "figures/diagnostics/n2bc": "figures/abi_proxy/diagnostics/n2bc",
    "figures/diagnostics/scene_id": "figures/abi_proxy/diagnostics/scene_id",
    "figures/diagnostics/adm": "figures/abi_proxy/diagnostics/adm",
    "figures/diagnostics/broadband_flux": "figures/abi_proxy/diagnostics/broadband_flux",
    "figures/diagnostics/radiance_difference": "figures/abi_proxy/diagnostics/radiance_difference",
    "figures/diagnostics/parallax": "figures/abi_proxy/diagnostics/parallax",
    "figures/diagnostics/instrument": "figures/nominal_production/diagnostics/instrument",
    "figures/products/broadband_flux": "figures/abi_proxy/products/broadband_flux",
    "figures/products/monthly_flux": "figures/abi_proxy/products/monthly_flux",
    "figures/ADM": "figures/abi_proxy/diagnostics/adm",
    "figures/abi_sunny_comparison_smoke": "figures/nominal_production/diagnostics/abi_sunny/smoke",
    "figures/abi_sunny_comparison": "figures/nominal_production/diagnostics/abi_sunny/comparison",
    "figures/gmm_diagnostics": "figures/nominal_production/diagnostics/gmm/production",
    "figures/scene_id": "figures/abi_proxy/diagnostics/scene_id",
    "figures/parallax": "figures/abi_proxy/diagnostics/parallax",
    "figures/radiance_difference": "figures/abi_proxy/diagnostics/radiance_difference",
    "figures/broadband_flux": "figures/abi_proxy/products/broadband_flux",
    "figures/monthly_flux": "figures/abi_proxy/products/monthly_flux",
}

FILES = {
    "figures/eco_channel_responses.png": "figures/nominal_production/diagnostics/instrument/eco_channel_responses.png",
    **{f"figures/uncertainty/{name}.png": f"figures/nominal_production/uncertainties/eco_spectral_assessment/{name}.png"
       for name in ("eco_spectral_k2_scatter", "eco_goal_residuals_by_regime",
                    "eco_adm_narrowband_flux_error", "abi_adm_proxy_variability")},
    "figures/uncertainty/eco_uncertainty_variance_decomposition.png":
        "figures/nominal_production/diagnostics/uncertainty/eco_uncertainty_variance_decomposition.png",
}


def figure_path(path):
    """Resolve legacy relative or absolute paths; current/custom paths are unchanged."""
    path = Path(path)
    parts = path.parts
    if "figures" not in parts:
        return path
    start = parts.index("figures")
    prefix = Path(*parts[:start])
    relative = Path(*parts[start:]).as_posix()
    if relative in FILES:
        return prefix / FILES[relative]
    # These are fit diagnostics, not retrieved broadband-flux maps.
    if relative.startswith("figures/broadband_flux/"):
        name = path.name
        if name.startswith(("irradiance_fit", "narrowband_to_broadband")):
            return prefix / "figures/abi_proxy/diagnostics/n2bc" / name
    for old, new in DIRECTORIES.items():
        if relative == old or relative.startswith(old+"/"):
            return prefix / (new+relative[len(old):])
    return path
