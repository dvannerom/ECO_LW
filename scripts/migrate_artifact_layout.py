"""Preview or apply same-filesystem artifact renames into the workflow layout.

No payload is copied, overwritten or deleted. Existing provenance is preserved
verbatim and remains subject to normal freshness checks after relocation.
"""

import argparse
import json
from pathlib import Path


DIRECTORIES = {
    "data/Sunny": "data/inputs/sunny",
    "data/goes_channels": "data/inputs/goes_channels",
    "data/preprocessed_files": "data/preprocessing/abi",
    "data/uncertainty_experiments/nominal_budget/features": "data/preprocessing/features",
    "data/scene_id": "data/abi_proxy/products/scene_id",
    "data/narrowband_flux": "data/abi_proxy/products/narrowband_flux",
    "data/broadband_flux": "data/abi_proxy/products/broadband_flux",
    "data/monthly": "data/abi_proxy/products/monthly_flux",
    "data/ADM": "legacy/data/ADM",
    "data/diagnostics/nominal_production": "data/nominal_production/diagnostics",
    "data/diagnostics/nominal_full": "data/gmm_component_selection/diagnostics",
    "data/uncertainty/sensitivity": "data/nominal_production/uncertainties/sensitivity",
    "data/uncertainty/eco_adm_retrieval": "data/nominal_production/products/spectral_assessment/narrowband_flux",
    "data/uncertainty_experiments/sensitivity_extensions": "data/nominal_production/uncertainties/geometry",
    "data/abi_sunny_comparison": "data/nominal_production/diagnostics/abi_sunny/comparison",
    "data/abi_sunny_comparison_smoke": "legacy/data/abi_sunny_comparison_smoke",
    "data/uncertainty/nominal": "legacy/data/uncertainty/nominal",
    "data/uncertainty/nominal_fast": "legacy/data/uncertainty/nominal_fast",
    "data/diagnostics/nominal": "legacy/data/diagnostics/nominal",
    "data/diagnostics/nominal_fast": "legacy/data/diagnostics/nominal_fast",
}
FILES = {
    "data/models/spectral_gmm.joblib": "data/scene_classification/models/spectral_gmm.joblib",
    "data/models/spectral_gmm.joblib.json": "data/scene_classification/models/spectral_gmm.joblib.json",
    "data/models/scene_taxonomy_gate.json": "data/scene_classification/scene_taxonomy_gate.json",
    "data/models/selected_n_components.json": "legacy/data/models/selected_n_components.json",
    "data/models/selected_n_components_multiday2.json": "legacy/data/models/selected_n_components_multiday2.json",
    "data/models/channel_radiance_power_law.json": "data/abi_proxy/models/n2b/channel_radiance_power_law.json",
    "data/models/narrowband_to_broadband_coeffs.json": "data/abi_proxy/models/n2b/narrowband_to_broadband_coeffs.json",
    "data/uncertainty/eco_uncertainty_registry.json": "data/nominal_production/reports/uncertainty_registry.json",
    "data/uncertainty/eco_uncertainty_gaps.md": "data/nominal_production/reports/uncertainty_gaps.md",
    "data/uncertainty/abi_adm_proxy_summary.json": "data/abi_proxy/diagnostics/adm_proxy_summary.json",
    "data/uncertainty/eco_spectral_reconstruction.json": "data/nominal_production/diagnostics/eco_spectral_reconstruction.json",
    "data/uncertainty/eco_spectral_residuals.csv": "data/nominal_production/diagnostics/eco_spectral_residuals.csv",
    "data/uncertainty/eco_n2bc_cv_models.joblib": "data/nominal_production/models/eco_n2bc_cv_models.joblib",
    "data/uncertainty/eco_n2bc_stage_metrics.json": "data/nominal_production/diagnostics/eco_n2bc_stage_metrics.json",
    "data/uncertainty/eco_adm_stage_metrics.json": "data/nominal_production/diagnostics/eco_adm_stage_metrics.json",
    "docs/Uncertainty budget and propagation for ECO_LW - Sheet1.csv": "legacy/reports/uncertainty_budget_snapshot.csv",
}


def migration_plan(root):
    """Plan O(number of directory entries) renames without reading large arrays."""
    root = Path(root).resolve()
    plan = []
    reserved = set()

    def add(old, new):
        source = root / old
        destination = root / new
        if not source.exists():
            return
        if destination in reserved:
            destination = root / "legacy" / old
        if destination.exists() or destination in reserved:
            raise FileExistsError(f"Migration will not overwrite {destination}")
        reserved.add(destination)
        plan.append((source, destination))

    for old, new in {**DIRECTORIES, **FILES}.items():
        add(old, new)
    # Both profiles used to duplicate fits. Keep production's copies canonical;
    # preserve colliding full-profile artifacts in legacy, never overwrite them.
    for profile in ("nominal_production", "nominal_full"):
        old = Path("data/uncertainty") / profile
        directory = root / old
        if not directory.is_dir():
            continue
        default = ("data/nominal_production/uncertainties/nominal_budget"
                   if profile == "nominal_production" else "data/gmm_component_selection/work")
        for child in sorted(directory.iterdir()):
            if child.name == "training_pca":
                new = Path("data/preprocessing/training_pca")
            elif child.name == "component_adms":
                new = Path("data/abi_proxy/models/adm")
            elif child.name.startswith(("candidate_", "initialization_sweep", "stability_sample", "sweep_logs")):
                new = Path("data/gmm_component_selection/models") / child.name
            elif child.name in ("cv_models.joblib", "nominal_models.joblib", "nominal_models.joblib.json"):
                new = Path("data/nominal_production/models") / child.name
            else:
                new = Path(default) / child.name
            add(old / child.name, new)
    models = root / "data/models"
    if models.is_dir():
        for child in sorted(models.glob("gmm_pipeline_*.joblib")):
            add(child.relative_to(root), Path("legacy/data/models") / child.name)
    return plan


def apply_plan(root, plan, manifest):
    """Preflight all paths/devices, then rename; publish a record of each move."""
    root = Path(root).resolve()
    manifest = Path(manifest).resolve()
    if manifest.exists():
        raise FileExistsError(f"Migration manifest already exists: {manifest}")
    for source, destination in plan:
        if not source.exists() or destination.exists():
            raise FileExistsError(f"Migration paths changed: {source} -> {destination}")
        ancestor = destination.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        if source.stat().st_dev != ancestor.stat().st_dev:
            raise ValueError(f"Cross-filesystem migration would copy payload: {source}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x") as record:
        for source, destination in plan:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            record.write(json.dumps({
                "source": str(source.relative_to(root)),
                "destination": str(destination.relative_to(root)),
                "provenance": "unchanged; freshness must be rechecked",
            }) + "\n")
            record.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true", help="Rename after stopping all workflow writers")
    parser.add_argument("--manifest", type=Path,
                        default=Path("logs/artifact_layout_migration.jsonl"))
    args = parser.parse_args()
    plan = migration_plan(args.root)
    for source, destination in plan:
        print(f"{source.relative_to(args.root.resolve())} -> {destination.relative_to(args.root.resolve())}")
    print(f"{len(plan)} renames; no copying or provenance rewriting.")
    if args.apply:
        apply_plan(args.root, plan, args.manifest)


if __name__ == "__main__":
    main()
