"""Configuration, artifact ownership, and non-destructive migration contracts."""

import importlib.util
import importlib
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest

from product_paths import broadband_path, narrowband_path, preprocessed_path, scene_id_path, nominal_product_path
from scene_taxonomy import load_scene_taxonomy, verify_selection
from workflow_config import load_settings, resolve_components, settings_paths


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("artifact_migration", ROOT / "scripts/migrate_artifact_layout.py")
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


def test_profiles_share_population_and_models_but_not_diagnostics():
    full = load_settings(ROOT / "config/gmm_component_selection.yaml")
    production = load_settings(ROOT / "config/nominal_production.yaml")
    for key in ("training_days", "validation_days", "cache_root", "training_dir",
                "candidate_dir", "model_path", "adm_dir", "gmm_seed"):
        assert full[key] == production[key]
    assert full["diagnostic_dir"] != production["diagnostic_dir"]
    taxonomy = load_scene_taxonomy(ROOT / "config/scene_taxonomy.yaml")
    sensitivity = load_settings(ROOT / "config/sensitivity_run.yaml")
    resolved = resolve_components(production, taxonomy, sensitivity)
    count = taxonomy["selection"]["n_components"]
    assert resolved["components"] == [count - 1, count, count + 1]
    disabled = {**sensitivity, "abi_variants": [], "sunny_variants": []}
    assert resolve_components(production, taxonomy, disabled)["components"] == [count]


def test_settings_composition_is_shallow_and_rejects_cycles(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("x: 1\nnested: {a: 2, b: 3}\n")
    child.write_text("extends: base.yaml\nnested: {b: 4}\n")
    assert load_settings(child) == {"x": 1, "nested": {"b": 4}}
    assert settings_paths(child) == [child, base]
    base.write_text("extends: child.yaml\n")
    for loader in (load_settings, settings_paths):
        with pytest.raises(ValueError, match="Cyclic"):
            loader(child)
    base.write_text("- not\n- a mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        load_settings(child)


def test_product_filenames_support_explicit_roots():
    assert preprocessed_path(245, root="custom/input") == "custom/input/abi_245_res2km_step1.nc"
    assert scene_id_path(245, 2, 7).startswith("data/abi_proxy/products/scene_id/")
    assert narrowband_path(245, 2, 0, "custom/nb") == "custom/nb/narrowband_flux_245_res2km_C0.nc"
    assert broadband_path(245, 2, "custom/bb") == "custom/bb/broadband_flux_245_res2km.nc"
    assert nominal_product_path("products", "CUSTOM six", "narrowband_flux") == (
        "products/narrowband_flux/eco_sunny_custom_six_grouped_heldout_clean_narrowband_flux.npz"
    )
    with pytest.raises(ValueError, match="scenario"):
        nominal_product_path("products", "---", "broadband_flux")


@pytest.mark.parametrize(("module_name", "defaults"), [
    ("fit_irradiance", {
        "filter_dir": "data/inputs/goes_channels",
        "output": "data/abi_proxy/models/n2b/channel_radiance_power_law.json",
    }),
    ("compute_temperature_SBDART", {
        "sunny_dir": "data/inputs/sunny",
        "filter_dir": "data/inputs/goes_channels",
        "power_law_file": "data/abi_proxy/models/n2b/channel_radiance_power_law.json",
        "output": "data/abi_proxy/models/n2b/narrowband_to_broadband_coeffs.json",
    }),
    ("compare_abi_sunny_scenes", {
        "sunny_dir": "data/inputs/sunny",
        "filter_dir": "data/inputs/goes_channels",
        "output_dir": "data/nominal_production/diagnostics/abi_sunny/comparison",
    }),
    ("evaluate_eco_spectral_reconstruction", {
        "scene_model": "data/scene_classification/models/spectral_gmm.joblib",
        "sunny_abi": "data/nominal_production/uncertainties/nominal_budget/sunny.npz",
        "adm_output_dir": "data/nominal_production/products/spectral_assessment/narrowband_flux",
    }),
])
def test_standalone_cli_defaults_match_workflow_layout(module_name, defaults, monkeypatch):
    module = importlib.import_module(module_name)

    def inspect_defaults(parser, *args, **kwargs):
        for key, relative in defaults.items():
            assert Path(parser.get_default(key)).resolve() == ROOT / relative
        raise SystemExit("defaults inspected")

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", inspect_defaults)
    with pytest.raises(SystemExit, match="defaults inspected"):
        module.main()


def test_migration_preserves_payload_mtime_and_provenance(tmp_path):
    old = tmp_path / "data/preprocessed_files"
    old.mkdir(parents=True)
    payload = old / "example.nc"
    payload.write_bytes(b"unmodified array payload")
    metadata = old / "example.nc.json"
    metadata.write_text('{"provenance": {"old_path": "data/preprocessed_files/example.nc"}}')
    original = (payload.stat().st_ino, payload.stat().st_mtime_ns, metadata.read_bytes())
    plan = migration.migration_plan(tmp_path)
    assert payload.exists()  # preview does not move anything
    manifest = tmp_path / "migration.jsonl"
    migration.apply_plan(tmp_path, plan, manifest)
    target = tmp_path / "data/preprocessing/abi/example.nc"
    assert (target.stat().st_ino, target.stat().st_mtime_ns,
            target.with_name("example.nc.json").read_bytes()) == original
    record = json.loads(manifest.read_text())
    assert record["provenance"].startswith("unchanged")
    assert migration.migration_plan(tmp_path) == []


def test_migration_preflights_conflicts_before_any_move(tmp_path):
    for name in ("data/Sunny", "data/preprocessed_files", "data/preprocessing/abi"):
        (tmp_path / name).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="will not overwrite"):
        migration.migration_plan(tmp_path)
    assert (tmp_path / "data/Sunny").exists()


def test_duplicate_profile_candidates_are_archived_not_overwritten(tmp_path):
    for profile in ("nominal_production", "nominal_full"):
        source = tmp_path / "data/uncertainty" / profile
        source.mkdir(parents=True)
        (source / "candidate_k7_s42.joblib").write_text(profile)
    plan = migration.migration_plan(tmp_path)
    migration.apply_plan(tmp_path, plan, tmp_path / "moves.jsonl")
    assert (tmp_path / "data/gmm_component_selection/models/candidate_k7_s42.joblib").read_text() == "nominal_production"
    assert (tmp_path / "legacy/data/uncertainty/nominal_full/candidate_k7_s42.joblib").read_text() == "nominal_full"


def test_public_dag_rules_are_available():
    pytest.importorskip("snakemake")
    result = subprocess.run(
        [sys.executable, "-m", "snakemake", "-s", "workflow/Snakefile",
         "--configfile", "config.yaml", "--list-rules"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    rules = set(result.stdout.splitlines())
    assert {"all", "all_goes_proxy", "gmm_selection_diagnostics",
            "uncertainty_assessment", "uncertainty_nominal_budget",
            "export_uncertainty_budget", "export_nominal_budget"} <= rules


def test_selection_review_rejects_production_subset():
    pytest.importorskip("snakemake")
    result = subprocess.run(
        [sys.executable, "-m", "snakemake", "-s", "workflow/Snakefile",
         "--configfile", "config.yaml", "--dry-run", "gmm_selection_diagnostics"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "requires the full review range" in result.stdout + result.stderr
