import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
from netCDF4 import Dataset

from nominal_provenance import (
    STAGE_SCOPED_CODE, candidate_identity, check_stage_provenance,
    configuration_identity, scoped_code_sha256, stage_code_paths, stage_signature,
    sweep_identity, initialization_identity,
)
from uncertainty import build_provenance, check_provenance


ROOT = Path(__file__).resolve().parents[1]


def _settings():
    return {
        "maximum_vza_deg": 70,
        "training_days": [245, 247],
        "gmm_chunk_points": 4096,
        "batch_points": 2048,
        "gmm_initializations": 2,
        "gmm_max_iter": 300,
        "gmm_tol": 0.001,
        "components": [4, 6],
        "gmm_seed": 42,
        "spatial": {"block_sizes": [1, 2, 5]},
        "noise_seed": 42,
        "figure_dir": "figures/nominal",
    }


def test_stage_provenance_checks_selected_configuration_and_legacy_scope(tmp_path):
    code = tmp_path / "producer.py"
    source = tmp_path / "source.dat"
    code.write_text("producer = 1\n")
    source.write_text("sample\n")
    settings = _settings()
    identity = configuration_identity("features", settings, {"day": 245})
    provenance = build_provenance(
        root=tmp_path,
        code_paths=[code],
        configuration_paths=(),
        input_paths={"source": source},
        stage="features",
        configuration_identity=identity,
    )

    assert provenance["schema_version"] == 2
    assert check_provenance(
        provenance, tmp_path, expected_stage="features",
        expected_configuration_identity=identity,
    )["status"] == "current"
    assert check_provenance(provenance, tmp_path)["status"] == "stale"
    assert check_provenance(
        provenance, tmp_path, expected_stage="pca",
        expected_configuration_identity=identity,
    )["status"] == "stale"

    changed_settings = {**settings, "maximum_vza_deg": 65}
    changed_identity = configuration_identity("features", changed_settings, {"day": 245})
    assert check_provenance(
        provenance, tmp_path, expected_stage="features",
        expected_configuration_identity=changed_identity,
    )["status"] == "stale"
    code.write_text("producer = 2\n")
    assert check_provenance(
        provenance, tmp_path, expected_stage="features",
        expected_configuration_identity=identity,
    )["status"] == "stale"

    legacy = build_provenance(
        root=tmp_path, code_paths=[code], configuration_paths=(), input_paths={"source": source}
    )
    assert check_provenance(legacy, tmp_path)["status"] == "current"
    assert check_provenance(
        legacy, tmp_path, expected_stage="features",
        expected_configuration_identity=identity,
    )["status"] == "stale"


def test_standalone_stage_checker_and_candidate_identities(tmp_path):
    settings = _settings()
    source = tmp_path / "source.dat"
    source.write_text("pixels")
    feature_identity = configuration_identity("features", settings, {"day": 245})
    provenance = build_provenance(
        root=tmp_path,
        code_paths=[],
        configuration_paths=(),
        input_paths={"source": source},
        stage="features",
        configuration_identity=feature_identity,
    )
    # Supply a custom root to verify the helper is usable without the runner.
    assert check_stage_provenance(
        provenance, "features", settings, {"day": 245}, root=tmp_path,
    )["status"] == "current"
    assert candidate_identity(settings, 4, 42) == {
        "components": 4, "seed": 42, "initializations": [0, 1],
    }
    assert initialization_identity(4, 42, 1) == {
        "components": 4, "seed": 42, "initialization": 1,
    }

    expanded = {**settings, "components": [4, 6, 8]}
    assert candidate_identity(expanded, 4, 42) == candidate_identity(settings, 4, 42)
    assert sweep_identity(expanded) != sweep_identity(settings)
    assert sweep_identity({**settings, "gmm_seed": 73}) != sweep_identity(settings)
    assert stage_signature(
        "candidate", expanded, candidate_identity(expanded, 4, 42)
    ) == stage_signature("candidate", settings, candidate_identity(settings, 4, 42))
    extra_init = {**settings, "gmm_initializations": 3}
    assert stage_signature(
        "candidate", extra_init, candidate_identity(extra_init, 4, 42)
    ) != stage_signature("candidate", settings, candidate_identity(settings, 4, 42))


def test_stage_signatures_ignore_unrelated_fields_and_candidate_set_growth():
    settings = _settings()
    base_signature = stage_signature("gmm", settings, {"components": 4, "seed": 42})
    expanded = {
        **settings,
        "components": [4, 6, 8],
        "gmm_seed": 42,
        "spatial": {"block_sizes": [1, 3, 5, 7]},
        "noise_seed": 99,
        "figure_dir": "figures/new-location",
    }
    assert stage_signature("gmm", expanded, {"components": 4, "seed": 42}) == base_signature
    assert stage_signature("gmm", expanded, {"components": 8, "seed": 42}) != base_signature
    start_identity = {"components": 4, "seed": 42, "initialization": 0}
    start_signature = stage_signature("gmm", settings, start_identity)
    assert stage_signature("gmm", expanded, start_identity) == start_signature
    assert stage_signature("gmm", expanded, {
        "components": 4, "seed": 42, "initialization": 1,
    }) != start_signature

    unrelated = {**settings, "spatial": {"block_sizes": [1]}, "figure_dir": "other"}
    assert stage_signature("features", unrelated, {"day": 245}) == stage_signature(
        "features", settings, {"day": 245}
    )
    changed = {**settings, "maximum_vza_deg": 65}
    assert stage_signature("features", changed, {"day": 245}) != stage_signature(
        "features", settings, {"day": 245}
    )
    changed = {**settings, "training_days": [245, 249]}
    assert stage_signature("pca", changed) != stage_signature("pca", settings)
    changed = {**settings, "gmm_max_iter": 301}
    assert stage_signature("gmm", changed, {"components": 4, "seed": 42}) != base_signature


def test_scoped_source_hash_ignores_unrelated_downstream_function(tmp_path):
    source = tmp_path / "combined.py"
    source.write_text("def feature():\n    return 1\n\ndef geometry():\n    return 2\n")
    original = scoped_code_sha256(source, ["feature"])
    source.write_text("def feature():\n    return 1\n\ndef geometry():\n    return 99\n")
    assert scoped_code_sha256(source, ["feature"]) == original
    source.write_text("def feature():\n    return 3\n\ndef geometry():\n    return 99\n")
    assert scoped_code_sha256(source, ["feature"]) != original


def test_runner_downstream_implementation_is_outside_upstream_scope(tmp_path):
    relative = "scripts/run_nominal_budget.py"
    source = tmp_path / "run_nominal_budget.py"
    original_text = (ROOT / relative).read_text()
    symbols = STAGE_SCOPED_CODE["features"][relative]
    source.write_text(original_text)
    original = scoped_code_sha256(source, symbols)
    source.write_text(original_text.replace(
        "def calculate(settings):",
        "def calculate(settings):\n    downstream_geometry_marker = True",
        1,
    ))
    assert scoped_code_sha256(source, symbols) == original


def test_gmm_source_identity_ignores_callbacks_but_tracks_fit_arguments(tmp_path):
    relative = "src/nominal_streaming.py"
    source = tmp_path / "nominal_streaming.py"
    original_text = (ROOT / relative).read_text()
    symbols = STAGE_SCOPED_CODE["gmm"][relative]
    source.write_text(original_text)
    original = scoped_code_sha256(source, symbols)

    source.write_text(original_text.replace(
        "f\"Starting GMM k={components}",
        "f\"Starting diagnostic GMM k={components}",
        1,
    ))
    assert scoped_code_sha256(source, symbols) == original

    source.write_text(original_text.replace(
        "lambda: feature_batches(settings, settings[\"training_days\"]),\n"
        "            components, seed, settings[\"gmm_initializations\"],",
        "lambda: feature_batches(settings, settings[\"training_days\"]),\n"
        "            components, seed + 1, settings[\"gmm_initializations\"],",
        1,
    ))
    assert scoped_code_sha256(source, symbols) != original


def test_upstream_workflow_uses_stage_local_code_and_signatures():
    workflow = (ROOT / "workflow/nominal_budget.smk").read_text()

    def rule_body(name):
        match = re.search(
            rf"rule {re.escape(name)}:\n(.*?)(?=\nrule |\Z)", workflow, flags=re.DOTALL
        )
        assert match is not None
        return match.group(1)

    features = rule_body("prepare_nominal_full_abi")
    pca = rule_body("prepare_nominal_full_training")
    sweep = rule_body("fit_nominal_full_sweep")
    candidate = rule_body("fit_nominal_full_candidate")

    assert "code=NOM_FEATURE_CODE" in features and "stage_signature=" in features
    assert "code=NOM_PCA_CODE" in pca and "stage_signature=" in pca
    assert "code=NOM_GMM_CODE" in sweep and "stage_signature(" in sweep
    assert '"sweep", NOMINAL' in sweep and "requested_jobs_signature=" in sweep
    assert "sweep=ancient(" in candidate and "code=NOM_GMM_CODE" in candidate
    assert '"candidate", NOMINAL' in candidate and "initializations" in candidate
    for body in (features, pca, sweep, candidate):
        assert "NOM_CODE" not in body
        assert "NOM_SETTINGS" not in body
        assert '"config.yaml"' not in body

    for stage in ("features", "pca", "gmm"):
        paths = stage_code_paths(stage)
        assert not any("geometry" in path or "adm" in path or "sensitivity" in path
                       for path in paths)
        assert "scripts/run_nominal_budget.py" not in paths
        assert "src/nominal_streaming.py" not in paths
        assert "src/streaming_gmm.py" not in paths
    gmm_source = STAGE_SCOPED_CODE["gmm"]["src/streaming_gmm.py"]
    assert {"MappedChunk", "MappedTrainingData", "_mapped_file",
            "_mapped_chunk_statistics", "_expectation_statistics",
            "_maximization", "fit_streaming_gmm"} <= set(gmm_source)
    assert "src/gmm_scheduler.py" not in stage_code_paths("gmm")


def _write_fixture_abi(path):
    height, width = 15, 16
    y, x = np.indices((height, width))
    channels = 9
    planck = np.broadcast_to([20000., 1300., 0., 1.], (channels, 4))
    with Dataset(path, "w") as source:
        for name, size in (("y", height), ("x", width), ("channel", channels),
                           ("coefficient", 4)):
            source.createDimension(name, size)
        source.createVariable("lat_interp_grid", "f4", ("y", "x"))[:] = 0.
        for satellite, offset in (("G16", 0.), ("G18", .4)):
            temperature = (230. + .7*y[:, :, None] + .25*x[:, :, None]
                           + 1.7*np.arange(channels)[None, None, :] + offset)
            radiance = planck[None, None, :, 0] / np.expm1(
                planck[None, None, :, 1] / temperature
            )
            source.createVariable(f"planck_{satellite}", "f8",
                                  ("channel", "coefficient"))[:] = planck
            source.createVariable(f"rad_{satellite}_interp", "f8",
                                  ("y", "x", "channel"))[:] = radiance
            source.createVariable(f"rad_{satellite}_interp_corr", "f8",
                                  ("y", "x", "channel"))[:] = radiance * 1.01
            source.createVariable(f"lza_{satellite}_interp_corr", "f8",
                                  ("y", "x"))[:] = 30. + x*.1 + y*.1


def test_real_feature_pca_and_initialization_provenance_resume(tmp_path, monkeypatch):
    import nominal_streaming as streaming
    import run_nominal_budget as runner
    from uncertainty import check_provenance

    source = tmp_path / "abi_245.nc"
    _write_fixture_abi(source)
    monkeypatch.setattr(streaming, "source_path", lambda day: source)
    settings = {
        "training_days": [245], "validation_days": [246, 247],
        "cache_root": str(tmp_path / "features"),
        "output_dir": str(tmp_path / "nominal"),
        "diagnostic_dir": str(tmp_path / "diagnostics"),
        "maximum_vza_deg": 70, "chunk_rows": 4, "batch_points": 64,
        "gmm_chunk_points": 32, "gmm_initializations": 1,
        "gmm_max_iter": 3, "gmm_tol": .01,
        "components": [2], "gmm_seed": 42,
    }
    fixture_settings = [settings]

    class RealStageRunner:
        ROOT = runner.ROOT
        write = staticmethod(runner.write)

        @staticmethod
        def provenance(inputs, stage=None, identity=None, **kwargs):
            return runner.provenance(
                inputs, stage=stage, identity=identity,
                settings=kwargs.get("settings", fixture_settings[0]),
            )

        @staticmethod
        def check_stage_provenance(record, stage, identity=None, settings=None):
            return runner.check_stage_provenance(
                record, stage, identity=identity,
                settings=settings if settings is not None else fixture_settings[0],
            )

        @staticmethod
        def verify(path, stage=None, identity=None, settings=None):
            return runner.verify(
                path, stage=stage, identity=identity,
                settings=settings if settings is not None else fixture_settings[0],
            )

    stage_runner = RealStageRunner()
    streaming.prepare_cache(settings, 245, stage_runner)
    cache = Path(settings["cache_root"]) / "245"
    feature_index = __import__("json").loads((cache / "index.json").read_text())
    feature_provenance = feature_index["provenance"]
    assert runner.check_stage_provenance(
        feature_index, "features", identity={"day": 245}, settings=settings,
    )["status"] == "current"

    streaming.prepare_training(settings, stage_runner)
    pca_dir = Path(settings["output_dir"]) / "training_pca"
    pca_index = __import__("json").loads((pca_dir / "index.json").read_text())
    pca_provenance = pca_index["provenance"]
    assert check_provenance(
        pca_provenance, runner.ROOT, expected_stage="pca",
        expected_configuration_identity=configuration_identity("pca", settings),
    )["status"] == "current"

    # The training-options loader checks PCA metadata with the real stage-aware
    # runner checker before the fit consumes its mapped scores.
    streaming.fit_candidate_initialization(settings, 2, 42, 0, stage_runner, workers=1)
    init_path = streaming.candidate_initialization_path(settings, 2, 42, 0)
    init_metadata_path = Path(str(init_path) + ".json")
    init_metadata = __import__("json").loads(init_metadata_path.read_text())
    init_provenance = init_metadata["provenance"]
    init_identity = {"components": 2, "seed": 42, "initialization": 0}
    assert runner.check_stage_provenance(
        init_provenance, "gmm", identity=init_identity, settings=settings,
    )["status"] == "current"

    unrelated = {
        **settings,
        "noise_seed": 99, "spatial": {"block_sizes": [1, 2, 5]},
    }
    assert runner.provenance(
        [cache, pca_dir], stage="gmm", identity=init_identity, settings=unrelated,
    ) == init_provenance
    assert runner.check_stage_provenance(
        init_metadata, "gmm", identity=init_identity, settings=unrelated,
    )["status"] == "current"
    assert runner.check_stage_provenance(
        feature_index, "features", identity={"day": 245}, settings=unrelated,
    )["status"] == "current"
    assert runner.check_stage_provenance(
        pca_index, "pca", settings=unrelated,
    )["status"] == "current"

    changed_feature_settings = {**unrelated, "maximum_vza_deg": 65}
    assert runner.check_stage_provenance(
        feature_index, "features", identity={"day": 245},
        settings=changed_feature_settings,
    )["status"] == "stale"
    changed_pca_settings = {**unrelated, "gmm_chunk_points": 64}
    assert runner.check_stage_provenance(
        pca_index, "pca", settings=changed_pca_settings,
    )["status"] == "stale"
    changed_gmm_settings = {**unrelated, "gmm_tol": .02}
    assert runner.check_stage_provenance(
        init_metadata, "gmm", identity=init_identity, settings=changed_gmm_settings,
    )["status"] == "stale"

    # A sweep restart takes the no-work path for the unchanged per-init identity.
    # If it tries to refit, the test fails; only manifest telemetry is rewritten.
    init_stat = init_path.stat().st_mtime_ns
    metadata_stat = init_metadata_path.stat().st_mtime_ns
    fixture_settings[0] = unrelated
    streaming.fit_initialization_sweep(unrelated, stage_runner, cpus=1)
    assert init_path.stat().st_mtime_ns == init_stat
    assert init_metadata_path.stat().st_mtime_ns == metadata_stat

    # Exercise the real manifest -> combined candidate -> held-out scoring
    # path, then prove an unrelated downstream configuration edit does not
    # invalidate either the fitted candidate or the validation feature caches.
    for day in unrelated["validation_days"]:
        streaming.prepare_cache(unrelated, day, stage_runner)
    streaming.combine_candidate_initializations(unrelated, 2, 42, stage_runner)
    candidate_path = Path(unrelated["output_dir"]) / "candidate_k2_s42.joblib"
    candidate_record = __import__("json").loads(
        Path(str(candidate_path) + ".json").read_text()
    )
    candidate_id = candidate_identity(unrelated, 2, 42)
    assert runner.check_stage_provenance(
        candidate_record, "candidate", identity=candidate_id, settings=unrelated,
    )["status"] == "current"

    changed_unrelated = {
        **unrelated,
        "noise_seed": 1234,
        "spatial": {"block_sizes": [1, 3, 5, 7]},
    }
    fixture_settings[0] = changed_unrelated
    assert runner.check_stage_provenance(
        candidate_record, "candidate", identity=candidate_identity(changed_unrelated, 2, 42),
        settings=changed_unrelated,
    )["status"] == "current"
    streaming.score_candidate(changed_unrelated, 2, 42, stage_runner)
    score_path = (
        Path(changed_unrelated["output_dir"]) / "candidate_scores/k2_s42.json"
    )
    score_record = __import__("json").loads(score_path.read_text())
    assert score_record["components"] == 2
    assert score_record["seed"] == 42
    assert feature_provenance["stage"] == "features"


def test_real_snakemake_dryrun_reuses_candidate_on_unrelated_config_and_k_growth(tmp_path):
    """Parse the actual DAG with synthetic completed artifacts, never execute jobs."""
    if importlib.util.find_spec("snakemake") is None:
        import pytest

        pytest.skip("Snakemake is not installed in this Python environment")

    import yaml

    root = ROOT
    work = tmp_path / "workflow-root"
    work.mkdir()
    for name in ("src", "scripts", "workflow"):
        (work / name).symlink_to(root / name, target_is_directory=True)
    config_dir = work / "config"
    config_dir.symlink_to(root / "config", target_is_directory=True)

    nominal = {
        "schema_version": 1,
        "abi_mode": "full",
        "output_dir": str(tmp_path / "nominal"),
        "diagnostic_dir": str(tmp_path / "diagnostics"),
        "figure_dir": str(tmp_path / "figures"),
        "diagnostic_figure_dir": str(tmp_path / "diagnostic-figures"),
        "training_days": [245],
        "validation_days": [246, 247],
        "cache_root": str(tmp_path / "features"),
        "chunk_rows": 4,
        "batch_points": 64,
        "gmm_chunk_points": 32,
        "maximum_vza_deg": 70,
        "gmm_initializations": 1,
        "gmm_max_iter": 3,
        "gmm_tol": 0.01,
        "components": [2],
        "gmm_seed": 42,
        "noise_seed": 10,
        "spatial": {"block_sizes": [1, 2, 5]},
    }
    settings_path = tmp_path / "nominal.yaml"
    settings_path.write_text(yaml.safe_dump(nominal))
    project_config = yaml.safe_load((root / "config.yaml").read_text())
    project_config.update({
        "days": [245, 246, 247],
        "scene_id_diagnostic_day": 247,
        "nominal_budget_settings": str(settings_path),
        "scene_taxonomy_file": str(config_dir / "scene_taxonomy.yaml"),
        "sensitivity_settings": str(config_dir / "sensitivity_run.yaml"),
        "eco_channel_scenarios_file": str(config_dir / "eco_channel_scenarios.yaml"),
    })
    config_path = work / "config.yaml"
    config_path.write_text(yaml.safe_dump(project_config))

    # Seed synthetic ABI input and completed nominal outputs. Future mtimes keep
    # this strictly a dry-run DAG validation; no scientific workflow job runs.
    preprocessed = work / "data/preprocessed_files/abi_245_res2km_step1.nc"
    preprocessed.parent.mkdir(parents=True)
    preprocessed.write_bytes(b"dry-run input placeholder")
    cache = Path(nominal["cache_root"]) / "245"
    cache.mkdir(parents=True)
    (cache / ".snakemake_timestamp").touch()
    pca = Path(nominal["output_dir"]) / "training_pca"
    pca.mkdir(parents=True)
    (pca / ".snakemake_timestamp").touch()
    sweep = Path(nominal["output_dir"]) / "initialization_sweep.json"
    sweep.parent.mkdir(parents=True, exist_ok=True)
    candidate = Path(nominal["output_dir"]) / "candidate_k2_s42.joblib"
    sweep.write_text("{}")
    candidate.write_bytes(b"dry-run fixture")
    Path(str(candidate) + ".json").write_text("{}")
    future = 2_000_000_000
    for path in (cache, cache / ".snakemake_timestamp", pca,
                 pca / ".snakemake_timestamp", sweep, candidate,
                 Path(str(candidate) + ".json")):
        os.utime(path, (future, future))

    python = sys.executable
    env = {**os.environ, "PYTHONPATH": str(work / "src") + os.pathsep + str(work / "scripts")}

    def invoke(*flags_and_targets):
        return subprocess.run([
            python, "-m", "snakemake",
            "--snakefile", str(root / "workflow" / "Snakefile"),
            "--directory", str(work),
            "--configfile", str(config_path),
            "--cores", "1", *flags_and_targets,
        ], cwd=work, env=env, text=True, capture_output=True, check=True).stdout

    def dryrun(*targets):
        return invoke("--dry-run", "--printshellcmds", *targets)

    candidate_target = str(candidate)
    baseline = dryrun(candidate_target)
    assert "Nothing to be done" in baseline or "0 jobs" in baseline

    # Rebuilt training inputs must recombine candidates even though the sweep
    # manifest is ancient and therefore does not propagate its own timestamp.
    os.utime(cache / ".snakemake_timestamp", (future + 1, future + 1))
    refreshed_training = dryrun(candidate_target)
    assert "candidate-combine --components 2 --seed 42" in refreshed_training
    os.utime(cache / ".snakemake_timestamp", (future, future))

    # Downstream noise/geometry edits alter the whole YAML but leave the real
    # candidate stage signature and its already-complete rule outputs current.
    nominal["noise_seed"] = 99
    nominal["spatial"] = {"block_sizes": [1, 3, 5, 7]}
    settings_path.write_text(yaml.safe_dump(nominal))
    unrelated = dryrun(candidate_target)
    assert "Nothing to be done" in unrelated or "0 jobs" in unrelated

    # Queue growth creates a job for K=3 but the existing K=2 output remains
    # complete; the candidate rule's ancient sweep input prevents manifest
    # refresh from rebuilding K=2.
    nominal["components"] = [2, 3]
    settings_path.write_text(yaml.safe_dump(nominal))
    grown = dryrun(
        candidate_target,
        str(Path(nominal["output_dir"]) / "candidate_k3_s42.joblib"),
    )
    assert "--components 3 --seed 42" in grown
    assert "--components 2 --seed 42" not in grown
