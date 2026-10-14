"""Manual GMM component-count gate for the shared spectral scene taxonomy."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import scene_taxonomy
from scene_features import spectral_feature_names
from uncertainty import sha256_file

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("n_components", "full_heldout_loglik", "adm_ratio_rmse", "ari_min",
           "full_minimum_occupancy")


def write_case(tmp_path, n_components=None, digest="auto", unquoted_date=False, **record):
    settings = tmp_path / "nominal.yaml"
    settings.write_text("output_dir: x\n")
    table = tmp_path / "diag.csv"
    with table.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for count in (2, 3, 4):
            writer.writerow([count, -10 + count, 1 / count, .9 - .1 * count, .05 / count])
    taxonomy = {
        "schema_version": 1, "feature_set": "spectral_10",
        "features": spectral_feature_names(),
        "diagnostics": {"settings": "nominal.yaml", "csv": "diag.csv",
                        "csv_sha256": sha256_file(table) if digest == "auto" else digest},
        "selection": {"n_components": n_components, "rationale": "plateau",
                      "selected_by": "reviewer", "date": "2026-10-09", **record},
    }
    text = yaml.safe_dump(taxonomy, sort_keys=False)
    if unquoted_date:
        text = text.replace("'2026-10-09'", "2026-10-09")
    path = tmp_path / "taxonomy.yaml"
    path.write_text(text)
    return path


def verify(path, active="nominal.yaml"):
    return scene_taxonomy.verify_selection(
        scene_taxonomy.load_scene_taxonomy(path), path.parent, active, path)


def test_repository_taxonomy_is_structurally_valid():
    taxonomy = scene_taxonomy.load_scene_taxonomy(ROOT / "config/scene_taxonomy.yaml")
    assert taxonomy["features"] == spectral_feature_names()


@pytest.mark.parametrize("change, message", [
    ({"feature_set": "texture_22"}, "feature_set"),
    ({"features": spectral_feature_names()[:9]}, "features"),
    ({"selection": {"n_components": 1}}, "integer >= 2"),
    ({"selection": {"n_components": True}}, "integer >= 2"),
    ({"diagnostics": {"settings": "x"}}, "diagnostics.csv"),
])
def test_load_rejects_malformed_taxonomy(tmp_path, change, message):
    path = write_case(tmp_path)
    taxonomy = yaml.safe_load(path.read_text())
    taxonomy.update(change)
    path.write_text(yaml.safe_dump(taxonomy))
    with pytest.raises(ValueError, match=message):
        scene_taxonomy.load_scene_taxonomy(path)


def test_unselected_k_fails_with_review_instructions(tmp_path):
    with pytest.raises(scene_taxonomy.TaxonomySelectionError, match="gmm_selection_diagnostics"):
        verify(write_case(tmp_path))


def test_hash_mismatch_forces_new_review(tmp_path):
    with pytest.raises(scene_taxonomy.TaxonomySelectionError, match="csv_sha256"):
        verify(write_case(tmp_path, 3, digest="0" * 64))


def test_incomplete_record_and_unevaluated_k_are_rejected(tmp_path):
    with pytest.raises(scene_taxonomy.TaxonomySelectionError, match="rationale"):
        verify(write_case(tmp_path, 3, rationale=" "))
    with pytest.raises(scene_taxonomy.TaxonomySelectionError, match="not in the diagnostics"):
        verify(write_case(tmp_path, 7))


def test_active_settings_must_match_reviewed_profile(tmp_path):
    path = write_case(tmp_path, 3)
    (tmp_path / "other.yaml").write_text("output_dir: y\n")
    with pytest.raises(scene_taxonomy.TaxonomySelectionError, match="differ"):
        verify(path, "other.yaml")


def test_production_settings_may_share_reviewed_population_profile(tmp_path):
    path = write_case(tmp_path, 3)
    profile = {
        "schema_version": 1, "abi_mode": "full",
        "training_days": [245, 247], "validation_days": [246, 248],
        "cache_root": "data/features", "chunk_rows": 32, "batch_points": 100,
        "netcdf_chunk_cache_mb": 128, "gmm_chunk_points": 64,
        "gmm_initializations": 2, "gmm_max_iter": 300, "gmm_tol": .001,
        "gmm_seed": 42,
    }
    (tmp_path / "nominal.yaml").write_text(yaml.safe_dump({
        **profile, "components": [2, 3, 4], "output_dir": "full",
    }))
    (tmp_path / "production.yaml").write_text(yaml.safe_dump({
        **profile, "components": [2, 3, 4], "output_dir": "production",
    }))
    assert verify(path, "production.yaml")["n_components"] == 3

    changed_profile = {**profile, "training_days": [245, 249]}
    (tmp_path / "production.yaml").write_text(yaml.safe_dump(changed_profile))
    with pytest.raises(scene_taxonomy.TaxonomySelectionError, match="training_days"):
        verify(path, "production.yaml")


def test_gate_record_and_neighbour_deltas(tmp_path):
    gate = verify(write_case(tmp_path, 3))
    assert gate["n_components"] == 3
    structural = gate["structural_taxonomy_choice"]
    assert structural["neighbours"] == [2, 4]
    assert structural["deltas"]["4"]["full_heldout_loglik"] == pytest.approx(1.)
    assert structural["max_abs_delta"]["adm_ratio_rmse"] == pytest.approx(1/2 - 1/3)
    edge = verify(write_case(tmp_path, 4))["structural_taxonomy_choice"]
    assert edge["neighbours"] == [3] and edge["missing_neighbours"] == [5]


def test_cli_writes_json_gate_and_reports_failures(tmp_path):
    output = tmp_path / "gate.json"
    command = [sys.executable, str(ROOT / "scripts/check_scene_taxonomy.py"),
               "--taxonomy", str(write_case(tmp_path, 3, unquoted_date=True)),
               "--root", str(tmp_path), "--active-settings", "nominal.yaml",
               "--output", str(output)]
    env = {"PYTHONPATH": str(ROOT / "src")}
    subprocess.run(command, check=True, env=env, cwd=tmp_path)
    assert json.loads(output.read_text())["selection"]["date"] == "2026-10-09"
    command[command.index("--taxonomy") + 1] = str(write_case(tmp_path))
    failed = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, text=True)
    assert failed.returncode == 2
    assert "SCENE TAXONOMY GATE:" in failed.stderr


def test_production_classifier_uses_shared_spectral_model(tmp_path):
    import joblib
    import numpy as np
    import xarray as xr
    from netCDF4 import Dataset
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from scene_features import iter_abi_spectral_features

    height, width = 7, 9
    rng = np.random.default_rng(1)
    bt = np.where(np.arange(width) < 4, 230., 285.)[None, :, None] + rng.normal(0, 1, (height, width, 9))
    bt[0, 0] = np.nan
    source = tmp_path / "abi_245_res2km_step1.nc"
    with Dataset(source, "w") as handle:
        for name, size in (("y", height), ("x", width), ("channel", 9)):
            handle.createDimension(name, size)
        for name in ("lat_interp_grid", "lon_interp_grid"):
            handle.createVariable(name, "f4", ("y", "x"))[:] = 1.
        for satellite in ("G16", "G18"):
            handle.createVariable(f"BT_{satellite}_interp", "f4", ("y", "x", "channel"))[:] = bt
    features = np.concatenate([chunk for chunk, _ in iter_abi_spectral_features(source, 2, 1)])
    model = Pipeline([("scaler", StandardScaler()), ("pca", PCA(3)),
                      ("gmm", GaussianMixture(2, random_state=0))]).fit(features)
    joblib.dump({"model": model, "population": np.array([.5, .5])}, tmp_path / "spectral_gmm.joblib")
    (tmp_path / "gate.json").write_text(json.dumps(
        {"n_components": 2, "diagnostics_csv_sha256": "abc"}))
    subprocess.run([sys.executable, str(ROOT / "scripts/scene_id_pca.py"), "-f", str(source),
                    "--model", "spectral_gmm.joblib", "--gate", "gate.json", "--chunk-rows", "2"],
                   check=True, cwd=tmp_path, env={"PYTHONPATH": str(ROOT / "src")})
    with xr.open_dataset(tmp_path / "data/scene_id/scene_id_245_res2km_2comp.nc") as result:
        assert result.attrs["feature_set"] == "spectral_10"
        assert result.sizes["valid_pixel"] == height * width - 1
        sorted_labels = result["sorted_labels"].values.reshape(height, width)
        assert np.isnan(sorted_labels[0, 0])
        # Cold columns sort before warm columns under the C14 ordering.
        assert set(np.unique(sorted_labels[1:, :4])) == {0.}
        assert set(np.unique(sorted_labels[:, 4:])) == {1.}
        np.testing.assert_array_equal(result["labels_nozeros"].values,
                                      model.predict(features))


def test_review_figures_mark_no_k(tmp_path, monkeypatch):
    import importlib.util
    candidates = [{"components": count, "seed": seed,
                   "heldout_mean_log_likelihood": -10 + count + seed / 1000,
                   "heldout_day_log_likelihood": [-10 + count, -10.1 + count],
                   "minimum_occupancy": .1 / count}
                  for count in (2, 3, 4) for seed in (42, 73)]
    (tmp_path / "candidates.json").write_text(json.dumps({"provenance": {}, "candidates": candidates}))
    write_case(tmp_path)
    table = tmp_path / "diag.csv"
    rows = list(csv.DictReader(table.open()))
    # Columns written by nominal_streaming.diagnostics/combine_diagnostics.
    numeric = ("heldout_loglik", "icl", "ari_std", "worst_channel_adm_rmse",
               "worst_scene_adm_rmse", "minimum_angular_coverage", "exact_adm_ratio_rmse",
               "exact_worst_channel_adm_rmse", "exact_worst_scene_adm_rmse",
               "exact_corrected_flux_std", "exact_raw_flux_std", "exact_flux_improvement")
    flags = ("shortlisted", "criterion_adm_plateau", "criterion_occupancy",
             "criterion_stability", "criterion_angular_coverage", "criterion_adm_coverage",
             "criterion_exact_adm_coverage", "criterion_exact_improvement", "all_criteria")
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, [*rows[0], *numeric, *flags])
        writer.writeheader()
        writer.writerows([{**row, **{name: .1 * int(row["n_components"]) for name in numeric},
                           **{name: True for name in flags}} for row in rows])
    (tmp_path / "diag.csv.json").write_text(json.dumps({"provenance": {}}))
    spec = importlib.util.spec_from_file_location("plot_gmm_selection",
                                                  ROOT / "plotting/plot_gmm_selection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "check_provenance", lambda *args: {"status": "current"})
    monkeypatch.setattr(sys, "argv", ["plot", "--diagnostics", str(table), "--candidates",
                                      str(tmp_path / "candidates.json"),
                                      "--output-dir", str(tmp_path / "figures")])
    module.main()
    for name in ("gmm_selection_overview.png", "gmm_selection_overview_criteria.png",
                 "gmm_selection_stability.png"):
        assert (tmp_path / "figures" / name).stat().st_size > 0
