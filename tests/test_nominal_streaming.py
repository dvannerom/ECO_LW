"""Exhaustive native-pixel coverage and chunk-independent full ABI assessment."""

import csv
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from netCDF4 import Dataset
from sklearn.metrics import adjusted_rand_score

import nominal_streaming as streaming
import run_nominal_budget as runner
from sensitivity import angular_basis, angular_integral


class ThresholdModel:
    def __getitem__(self, index):
        assert index == -1
        return SimpleNamespace(n_components=2)

    def predict(self, features):
        return (features[:, 0] > 250).astype(np.int64)


def write_abi(path, height=23, width=27):
    y, x = np.indices((height, width))
    temperature = np.where(x < width//2, 240., 260.) + y*.05
    # Include near-boundary cases so added-noise transitions are nonzero.
    temperature[:, width//2] = 249.95
    labels = (temperature > 250).astype(int)
    raw = np.broadcast_to((20000./np.expm1(1300./temperature))[:, :, None, None],
                          (height, width, 2, 9)).copy()
    angles = np.stack((20 + y*.8 + x*.1, 50 - y*.2 + x*.05), axis=-1)
    parameters = np.array([-.2, -.4])[labels]
    shape = 1 + angular_basis(angles, "regularized")[..., 0]*parameters[:, :, None]
    corrected = raw*shape[..., None]
    # Eligibility should be pixel-wise, without texture screening or tile cutoffs.
    raw[0, 0] = np.nan
    angles[4, 4, 0] = 71.
    corrected[8, 8] = np.nan
    with Dataset(path, "w") as source:
        for name, size in (("y", height), ("x", width), ("channel", 9), ("coefficient", 4)):
            source.createDimension(name, size)
        source.createVariable("lat_interp_grid", "f4", ("y", "x"))[:] = 0
        for index, satellite in enumerate(("G16", "G18")):
            source.createVariable(f"planck_{satellite}", "f8", ("channel", "coefficient"))[:] = (
                np.broadcast_to([20000., 1300., 0., 1.], (9, 4)))
            source.createVariable(f"rad_{satellite}_interp", "f8", ("y", "x", "channel"))[:] = raw[:, :, index]
            source.createVariable(f"rad_{satellite}_interp_corr", "f8", ("y", "x", "channel"))[:] = corrected[:, :, index]
            source.createVariable(f"lza_{satellite}_interp_corr", "f8", ("y", "x"))[:] = angles[:, :, index]


@pytest.fixture
def full_case(tmp_path, monkeypatch):
    days = [245, 246, 247]
    for day in days:
        write_abi(tmp_path/f"{day}.nc")
    monkeypatch.setattr(streaming, "source_path", lambda day: tmp_path/f"{day}.nc")
    monkeypatch.setattr(streaming, "check_provenance", lambda *args: {"status": "current"})
    settings = {
        "abi_mode": "full", "training_days": [245], "validation_days": [246, 247],
        "cache_root": str(tmp_path/"features"), "output_dir": str(tmp_path/"budget"),
        "diagnostic_dir": str(tmp_path/"diagnostics"), "chunk_rows": 3,
        "batch_points": 57, "maximum_vza_deg": 70,
        "noise_realizations": 3, "noise_seed": 42, "nedt_k_at_255": .4,
        "components": [2], "nominal_components": 2, "seeds": [42],
        "gmm_initializations": 1, "gmm_max_iter": 300, "gmm_tol": .001,
        "product_resolution_km": 10,
        "spatial": {"block_sizes": [1, 2, 5], "nominal_block_size": 5,
                    "native_resolution_km": 2, "independent_of_other_sources": True},
    }
    hooks = SimpleNamespace(write=runner.write, provenance=lambda inputs: {},
                            verify=lambda path: None, abi_noise_features=runner.abi_noise_features,
                            spatial_tile=runner.spatial_tile)
    for day in days:
        streaming.prepare_cache(settings, day, hooks)
    output = Path(settings["output_dir"])
    output.mkdir()
    joblib.dump({"model": ThresholdModel(), "population": np.array([.5, .5])},
                output/"spectral_gmm.joblib")
    return settings, hooks


def test_complete_native_coverage_and_edge_chunks(full_case):
    settings, _ = full_case
    index, features = streaming.open_features(settings, 245)
    assert index["pixels"] == 23*27 - 3
    expected = np.concatenate([chunk[6][chunk[5]] for chunk in streaming.observations(settings, 245)])
    np.testing.assert_array_equal(features, expected)
    batches = list(streaming.feature_batches(settings, [245]))
    assert max(map(len, batches)) <= settings["batch_points"]
    np.testing.assert_array_equal(np.concatenate(batches), expected)
    settings["chunk_rows"] = 11
    other = np.concatenate([chunk[6][chunk[5]] for chunk in streaming.observations(settings, 245)])
    np.testing.assert_array_equal(other, expected)


def test_all_pixel_assignment_and_noise_chunk_invariance(full_case):
    settings, hooks = full_case
    path = Path(settings["output_dir"])/"abi_assignment.json"
    streaming.assignment(settings, hooks)
    first = json.loads(path.read_text())
    assert first["sample_pixels"] == 2*(23*27-3)
    assert np.sum(first["transition_counts"]) == first["sample_pixels"]*settings["noise_realizations"]
    assert first["sample_change_probability"] > 0
    assert first["noise_model"] == "mean_preserving_lognormal"
    settings["chunk_rows"] = 11
    streaming.assignment(settings, hooks)
    second = json.loads(path.read_text())
    assert first["transition_counts"] == second["transition_counts"]
    assert first["by_day_transition_counts"] == second["by_day_transition_counts"]


def test_dim_pixel_assignment_remains_positive_and_chunk_invariant(full_case):
    settings, hooks = full_case
    cold_radiance = 20000. / np.expm1(1300. / 100.)
    argument = 1300. / 255.
    sd = .4 * 20000. * np.exp(argument) * 1300. / (np.expm1(argument)**2 * 255.**2)
    draws = np.random.default_rng(42).normal(size=(100, 2, 6))
    assert np.any(cold_radiance + sd * draws <= 0)
    for day in settings["validation_days"]:
        with Dataset(streaming.source_path(day), "r+") as source:
            for satellite in ("G16", "G18"):
                # An eligible cold pixel with radiance much smaller than noise SD.
                source.variables[f"rad_{satellite}_interp"][1, 1, :] = cold_radiance
        streaming.prepare_cache(settings, day, hooks)
    path = Path(settings["output_dir"])/"abi_assignment.json"
    streaming.assignment(settings, hooks)
    first = json.loads(path.read_text())
    assert np.sum(first["transition_counts"]) == first["sample_pixels"]*settings["noise_realizations"]
    settings["chunk_rows"] = 11
    streaming.assignment(settings, hooks)
    second = json.loads(path.read_text())
    assert first["transition_counts"] == second["transition_counts"]
    assert first["by_day_transition_counts"] == second["by_day_transition_counts"]


def test_full_adm_fit_and_spatial_chunk_boundaries(full_case, monkeypatch):
    settings, hooks = full_case
    output = Path(settings["output_dir"])
    library = streaming.fit_library(settings, ThresholdModel(), output)
    np.testing.assert_allclose(library[:, :, 0], np.repeat([[-.2], [-.4]], 6, axis=1), atol=1e-6)
    assert not (output/"adm_fit_records.npy").exists()
    coefficients = np.zeros(83)
    coefficients[0] = .5
    coefficients = (coefficients, 100.)
    monkeypatch.setattr(streaming, "load_cubic_coefficients", lambda path: coefficients)
    config = {"narrowband_to_broadband_coeffs_file": "unused"}
    streaming.spatial(settings, config, hooks)
    first = json.loads((output/"spatial_report.json").read_text())
    # Compare streamed counts/moments with a single whole-grid spatial call.
    whole_settings = dict(settings, chunk_rows=100)
    _, raw, corrected, angles, planck, valid, _ = next(streaming.observations(whole_settings, 246))
    for row in first["rows"]:
        values = runner.spatial_tile(ThresholdModel(), raw, corrected, angles, planck,
                                     library, coefficients, valid, row["block_size"])
        assert row["footprints"] == 2*len(values)
        expected = runner.error_classes(values[:, None], np.full(len(values), 1/len(values)))
        for key in ("bias_w_m2", "scene_sd_w_m2", "ensemble_rmse_w_m2"):
            assert row[key] == pytest.approx(expected[key], abs=1e-10)
    settings["chunk_rows"] = 11
    streaming.spatial(settings, config, hooks)
    second = json.loads((output/"spatial_report.json").read_text())
    for first_row, second_row in zip(first["rows"], second["rows"]):
        assert first_row["footprints"] == second_row["footprints"]
        assert first_row["ensemble_rmse_w_m2"] == pytest.approx(second_row["ensemble_rmse_w_m2"], abs=1e-10)
    with np.load(output/"spatial_residuals.npz") as data:
        assert set(data.files) == {"block_sizes", "count", "mean", "m2"}


def test_moments_and_contingency_ari():
    values = np.random.default_rng(42).normal(size=1000) + 1e6
    moments = streaming.Moments()
    for chunk in np.array_split(values, 17):
        moments.update(chunk)
    assert moments.mean == pytest.approx(values.mean())
    assert moments.m2 / moments.count == pytest.approx(values.var(), abs=1e-10)
    labels = np.array([0, 1, 1, 2, 2, 0, 1, 0])
    other = np.array([1, 0, 0, 2, 0, 1, 0, 2])
    table = np.bincount(labels*3 + other, minlength=9).reshape(3, 3)
    assert streaming.contingency_ari(table) == pytest.approx(adjusted_rand_score(labels, other))
    with pytest.raises(ValueError, match="No eligible"):
        streaming.Moments().report()
    with pytest.raises(ValueError, match="Non-finite"):
        moments.update([np.nan])
    merged = streaming.Moments()
    for chunk in np.array_split(values, 17):
        part = streaming.Moments()
        part.update(chunk)
        merged.merge(part.count, part.mean, part.m2)
    assert merged.count == moments.count
    assert merged.mean == moments.mean
    assert merged.m2 == moments.m2
    with pytest.raises(ValueError, match="Invalid residual"):
        merged.merge(0, 0., 0.)


def test_full_settings_partition(full_case):
    settings, _ = full_case
    streaming.validate_settings(settings, {"days": [245, 246, 247]})
    with pytest.raises(ValueError, match="partition every"):
        streaming.validate_settings(settings, {"days": [245, 246, 247, 249]})
    with pytest.raises(ValueError, match="chunk_rows"):
        streaming.validate_settings(dict(settings, chunk_rows=0), {"days": [245, 246, 247]})
    for key in ("netcdf_chunk_cache_mb", "gmm_workers", "gmm_chunk_points"):
        for value in (0, True, 1.5):
            with pytest.raises(ValueError, match=key):
                streaming.validate_settings(dict(settings, **{key: value}),
                                            {"days": [245, 246, 247]})


def test_netcdf_chunk_cache_preserves_reads(full_case, tmp_path):
    settings, _ = full_case
    path = streaming.source_path(245)
    compressed = tmp_path / "compressed.nc"
    with Dataset(path) as original, Dataset(compressed, "w") as output:
        for name, dimension in original.dimensions.items():
            output.createDimension(name, len(dimension))
        for name, variable in original.variables.items():
            target = output.createVariable(name, variable.dtype, variable.dimensions, zlib=True)
            target[:] = variable[:]
    with Dataset(compressed) as source:
        expected = source.variables["rad_G16_interp"][:]
        streaming.configure_chunk_cache(source, dict(settings, netcdf_chunk_cache_mb=128))
        for satellite in ("G16", "G18"):
            for suffix in ("interp", "interp_corr"):
                assert source.variables[f"rad_{satellite}_{suffix}"].get_var_chunk_cache()[0] == 128*1024**2
            assert source.variables[f"lza_{satellite}_interp_corr"].get_var_chunk_cache()[0] == 128*1024**2
        np.testing.assert_array_equal(source.variables["rad_G16_interp"][:], expected)


def test_shared_preprocessing_and_fused_scores(full_case):
    settings, hooks = full_case
    import streaming_gmm
    features = np.concatenate(list(streaming.feature_batches(settings, settings["training_days"])))
    direct = streaming_gmm.fit_streaming_gmm(
        lambda: iter([features]), 2, 42, initializations=1,
        chunk_rows=settings.get("gmm_chunk_points", 65536),
    )
    streaming.prepare_training(settings, hooks)
    options = streaming.training_options(settings, workers=2)
    cached = streaming_gmm.fit_streaming_gmm(
        lambda: iter([features]), 2, 42, initializations=1, **options,
    )
    np.testing.assert_allclose(cached.score_samples(features), direct.score_samples(features),
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(cached.predict(features), direct.predict(features))
    scores, labels = streaming.score_and_labels(cached, features)
    np.testing.assert_allclose(scores, cached.score_samples(features), rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(labels, cached.predict(features))
    directory = Path(settings["output_dir"]) / "training_pca"
    transformed = np.load(directory / "245.npy", mmap_mode="r")
    assert transformed.dtype == np.float64
    np.testing.assert_allclose(
        transformed, direct[1].transform(direct[0].transform(features.astype(np.float64))),
        rtol=1e-12, atol=1e-12,
    )
    malformed = np.zeros((len(features), transformed.shape[1]), dtype=np.float32)
    np.save(directory / "245.npy", malformed)
    with pytest.raises(ValueError, match="Invalid transformed"):
        list(streaming.training_options(settings, workers=1)["transformed_batches"]())


def test_assignment_shards_match_unsplit_reference(full_case):
    settings, hooks = full_case
    model = ThresholdModel()
    expected = np.zeros((2, 2), dtype=np.int64)
    baseline = np.zeros(2, dtype=np.int64)
    for day in settings["validation_days"]:
        rngs = [np.random.default_rng(np.random.SeedSequence([settings["noise_seed"], day, r, 3]))
                for r in range(settings["noise_realizations"])]
        for _, raw, _, _, planck, features in streaming.paired_records(settings, [day]):
            labels = model.predict(features)
            baseline += np.bincount(labels, minlength=2)
            fk1, fk2, bc1, bc2 = np.moveaxis(planck, -1, 0)
            argument = fk2 / (bc1 + bc2*255.)
            sd = settings["nedt_k_at_255"]*fk1*np.exp(argument)*fk2*bc2 / (
                np.expm1(argument)**2*(bc1 + bc2*255.)**2)
            for rng in rngs:
                perturbed = runner.lognormal_radiance(raw, sd, rng.normal(size=raw.shape))
                noisy = model.predict(hooks.abi_noise_features(perturbed, planck, 0))
                expected += np.bincount(labels*2 + noisy, minlength=4).reshape(2, 2)
    for day in reversed(settings["validation_days"]):
        streaming.assignment_day(settings, day, hooks)
    streaming.combine_assignment(settings, hooks)
    output = Path(settings["output_dir"])
    report = json.loads((output / "abi_assignment.json").read_text())
    np.testing.assert_array_equal(report["transition_counts"], expected)
    np.testing.assert_array_equal(report["baseline_sample_counts"], baseline)
    path = output / "assignment_parts" / f"day{settings['validation_days'][0]}.json"
    record = json.loads(path.read_text())
    record["noise_realizations"] += 1
    runner.write(path, record)
    with pytest.raises(ValueError, match="Invalid assignment"):
        streaming.combine_assignment(settings, hooks)


def test_diagnostic_aggregation_orders_plateau_and_missing_ari(full_case):
    settings, hooks = full_case
    settings["components"] = [2, 3]
    for count, rmse in ((3, .999), (2, 1.)):
        runner.write(Path(settings["diagnostic_dir"]) / "component_parts" / f"k{count}.json", {
            "components": count, "provenance": {},
            "row": {"n_components": count, "adm_ratio_rmse": rmse,
                    "ari_min": None, "ari_std": None,
                    "criterion_adm_plateau": False, "criterion_stability": False,
                    "all_criteria": False},
        })
    streaming.combine_diagnostics(settings, hooks)
    with (Path(settings["diagnostic_dir"]) / "gmm_component_diagnostics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["n_components"]) for row in rows] == [2, 3]
    assert rows[0]["criterion_adm_plateau"] == "False"
    assert rows[1]["criterion_adm_plateau"] == "True"
    assert all(row["ari_min"] == "nan" for row in rows)
    assert all(row["all_criteria"] == "False" for row in rows)


def test_streamed_training_through_budget_and_plots(full_case, monkeypatch):
    settings, hooks = full_case
    settings.update(seeds=[42, 73], folds=5,
                    figure_dir=str(Path(settings["output_dir"])/"figures"),
                    diagnostic_figure_dir=str(Path(settings["diagnostic_dir"])/"figures"))
    hooks.ROOT = runner.ROOT
    output = Path(settings["output_dir"])
    settings.update(stability_max_points=200, stability_repeats=2, stability_seed=42)
    streaming.prepare_stability_sample(settings, hooks)
    for seed in settings["seeds"]:
        for initialization in range(settings["gmm_initializations"]):
            streaming.fit_candidate_initialization(
                settings, 2, seed, initialization, hooks,
            )
        streaming.combine_candidate_initializations(settings, 2, seed, hooks)
    streaming.train(settings, hooks)
    model = joblib.load(output/"spectral_gmm.joblib")["model"]
    selection = json.loads((Path(settings["diagnostic_dir"])/"gmm_selection.json").read_text())
    assert sum(selection["heldout_population_counts"]) == 2*(23*27-3)
    scores = np.concatenate([model.score_samples(features)
                             for features in streaming.feature_batches(settings, [246, 247])])
    assert selection["heldout_log_density_p01"] == pytest.approx(np.quantile(scores, .01))
    assert not (output/"heldout_density.npy").exists()
    coefficients = np.zeros(83)
    coefficients[0] = .5
    monkeypatch.setattr(streaming, "load_cubic_coefficients", lambda path: (coefficients, 100.))
    config = {"narrowband_to_broadband_coeffs_file": "unused"}
    fitted_counts = []
    original_fit = streaming.fit_library

    def record_fit(settings, model, *args, **kwargs):
        fitted_counts.append(model[-1].n_components)
        return original_fit(settings, model, *args, **kwargs)

    monkeypatch.setattr(streaming, "fit_library", record_fit)
    streaming.assignment(settings, hooks)
    streaming.spatial(settings, config, hooks)
    streaming.diagnostics(settings, config, hooks)
    assert fitted_counts == [2]
    assert not list(output.glob("adm_*.npy"))
    diagnostic_metadata = json.loads((Path(settings["diagnostic_dir"])/
                                     "gmm_component_diagnostics.csv.json").read_text())
    assert "bootstrap refits" in diagnostic_metadata["reliability"]
    with (Path(settings["diagnostic_dir"])/"gmm_component_diagnostics.csv").open() as handle:
        diagnostic_rows = list(csv.DictReader(handle))
    assert np.isfinite(float(diagnostic_rows[0]["ari_min"]))
    sample_metadata = json.loads((output/"stability_sample.npz.json").read_text())
    assert sample_metadata["train_points"] + sample_metadata["test_points"] == 200

    n = 60
    angles = np.arange(0, 75, 5)
    rng = np.random.default_rng(42)
    parameters = -.3 + .02*rng.normal(size=(n, 6))
    amplitude = rng.uniform(20, 60, (n, 6))
    eco = amplitude[:, None, :]*(
        1 + angular_basis(angles, "regularized")[:, 0][None, :, None]*parameters[:, None, :])
    true_band = np.pi*amplitude*angular_integral(parameters[..., None], "regularized")
    temperature = np.where(np.arange(n) % 2, 260., 240.)
    abi = np.broadcast_to((20000./np.expm1(1300./temperature))[:, None, None],
                          (n, len(angles), 6)).copy()
    np.savez(output/"sunny.npz", eco_radiance=eco, abi_radiance=abi,
             true_band_flux=true_band, reference=true_band.mean(axis=1)+10,
             planck=np.broadcast_to([20000., 1300., 0., 1.], (6, 4)),
             eco_noise_sd=np.full((n, 6), .002), angles=angles,
             groups=np.repeat(np.arange(n//2).astype(str), 2),
             regimes=np.where(np.arange(n) % 2, "cloudy", "clear"))
    monkeypatch.setattr(runner, "verify", hooks.verify)
    monkeypatch.setattr(runner, "provenance", hooks.provenance)
    monkeypatch.setattr(runner, "check_provenance", lambda *args: {"status": "current"})
    runner.calculate(settings)
    report = json.loads((output/"budget_report.json").read_text())
    assert report["settings"]["abi_mode"] == "full"
    assert report["support"]["covered_population"] == 1.
    assert all(np.isfinite(row["ensemble_rmse_w_m2"]) for row in report["rows"])
    with np.load(output/"residuals.npz") as data:
        np.testing.assert_allclose(
            data["total"], data["spectral"]+data["angular"]+data["radiometric"]+data["assignment"],
            atol=1e-10)
    spec = importlib.util.spec_from_file_location("nominal_full_plot", Path("plotting/plot_nominal_budget.py"))
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)
    monkeypatch.setattr(plotter, "check_provenance", lambda *args: {"status": "current"})
    monkeypatch.setattr("sys.argv", ["plot_nominal_budget.py", "--report", str(output/"budget_report.json")])
    plotter.main()
    with (output/"numerical_budget.csv").open(newline="") as handle:
        table = list(csv.reader(handle))
    assert len(table) == 11
    assert all(len(row) == 5 for row in table)
    assert (Path(settings["figure_dir"])/"eco_uncertainty_numerical_budget.png").exists()
    assert (Path(settings["diagnostic_figure_dir"])/"gmm_diagnostics.png").exists()
    assert (Path(settings["diagnostic_figure_dir"])/"gmm_stability.png").exists()


def test_failed_initialization_does_not_discard_successful_candidate(full_case, monkeypatch):
    settings, hooks = full_case
    settings["gmm_initializations"] = 2
    import streaming_gmm

    fit = streaming_gmm.fit_streaming_gmm

    def fail_first_initialization(*args, initialization_index, **kwargs):
        if initialization_index == 0:
            raise RuntimeError("simulated non-convergence")
        return fit(*args, initialization_index=initialization_index, **kwargs)

    monkeypatch.setattr(streaming_gmm, "fit_streaming_gmm", fail_first_initialization)
    for initialization in range(2):
        streaming.fit_candidate_initialization(
            settings, 2, 42, initialization, hooks,
        )
    streaming.combine_candidate_initializations(settings, 2, 42, hooks)

    model = joblib.load(Path(settings["output_dir"])/"candidate_k2_s42.joblib")
    assert np.isnan(model[-1].streaming_initialization_scores_[0])
    assert np.isfinite(model[-1].streaming_initialization_scores_[1])
    assert len(model[-1].streaming_initialization_failures_) == 1
