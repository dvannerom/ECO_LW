"""Presentation-only relocation of legacy figure paths, including saved reports."""

from pathlib import Path


DIRECTORIES = {
    "figures/ADM": "figures/diagnostics/adm",
    "figures/abi_sunny_comparison_smoke": "figures/diagnostics/abi_sunny/smoke",
    "figures/abi_sunny_comparison": "figures/diagnostics/abi_sunny/comparison",
    "figures/gmm_diagnostics": "figures/diagnostics/gmm/production",
    "figures/scene_id": "figures/diagnostics/scene_id",
    "figures/parallax": "figures/diagnostics/parallax",
    "figures/radiance_difference": "figures/diagnostics/radiance_difference",
    "figures/broadband_flux": "figures/products/broadband_flux",
    "figures/monthly_flux": "figures/products/monthly_flux",
    "figures/uncertainty/convergence": "figures/diagnostics/convergence",
    "figures/uncertainty/extensions": "figures/diagnostics/extensions",
    "figures/uncertainty/sensitivity": "figures/diagnostics/sensitivity",
}

FILES = {
    "figures/eco_channel_responses.png": "figures/diagnostics/instrument/eco_channel_responses.png",
    **{f"figures/uncertainty/{name}.png": f"figures/uncertainty/spectral_angular/{name}.png"
       for name in ("eco_spectral_k2_scatter", "eco_goal_residuals_by_regime",
                    "eco_adm_narrowband_flux_error", "abi_adm_proxy_variability")},
    **{f"figures/uncertainty/eco_uncertainty_{name}.png":
       f"figures/uncertainty/registry/eco_uncertainty_{name}.png"
       for name in ("evidence_map", "residual_summary", "variance_decomposition")},
    **{f"figures/uncertainty/eco_uncertainty_{name}.png":
       f"figures/uncertainty/historical/first_draft/eco_uncertainty_{name}.png"
       for name in ("numerical_budget", "scenario_budgets", "channel_diagnostics")},
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
            return prefix / "figures/diagnostics/n2bc" / name
    for old, new in DIRECTORIES.items():
        if relative == old or relative.startswith(old+"/"):
            return prefix / (new+relative[len(old):])
    return path
