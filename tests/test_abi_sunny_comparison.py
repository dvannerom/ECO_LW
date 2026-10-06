"""Spectral-only comparison regression tests, including actual PFM/Sunny grids."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import joblib
import numpy as np
import xarray as xr

from abi_sunny_comparison import (
    abi_response_on_grid, blackbody_roundtrip, classify_spectral_features,
    convolve_abi_radiances, fit_spectral_model, sample_abi_file,
    synthetic_brightness_temperature,
)
from compare_abi_sunny_scenes import run_comparison
from parallel_gmm import ParallelGaussianMixture
from scene_features import (
    build_scene_features, iter_abi_spectral_features, spectral_features_from_bt,
)

ROOT = Path(__file__).resolve().parents[1]
PLANCK_G16 = np.array([
    [50687.1, 2331.58, 1.55228, .99667],
    [19779.9, 1703.83, .18733, .99948],
    [13432.1, 1497.61, .09102, .99971],
    [8510.22, 1286.27, .22516, .99920],
    [6454.62, 1173.03, .21702, .99916],
    [5101.27, 1084.53, .06266, .99974],
])


def write_abi(path):
    rng = np.random.default_rng(12)
    bt16 = rng.uniform(220, 300, (13, 15, 9)).astype(np.float32)
    bt18 = bt16 + rng.normal(0, 2, bt16.shape).astype(np.float32)
    bt16[0, 0, 0] = np.nan
    bt18[2, 2, 3] = 0
    planck = np.ones((9, 4))
    planck[[0, 3, 4, 6, 7, 8]] = PLANCK_G16
    xr.Dataset({
        "BT_G16_interp": (("y", "x", "channel"), bt16),
        "BT_G18_interp": (("y", "x", "channel"), bt18),
        "lat_interp_grid": (("y", "x"), np.zeros((13, 15))),
        "lon_interp_grid": (("y", "x"), np.zeros((13, 15))),
        "planck_G16": (("channel", "planck_parameter"), planck),
    }).to_netcdf(path)


class SpectralFeaturesTests(unittest.TestCase):
    def test_channel_and_difference_order(self):
        bt = np.array([[230., 250., 240., 270., 260., 220.]])
        np.testing.assert_array_equal(
            spectral_features_from_bt(bt), [[230, 250, 240, 270, 260, 220, 20, 10, 40, 50]]
        )
        with self.assertRaises(ValueError):
            spectral_features_from_bt(np.zeros((2, 9)))

    def test_stream_matches_production_spectral_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "abi.nc"
            write_abi(path)
            expected, positions, *_ = build_scene_features(path, chunk_rows=3, pixel_step=2)
            blocks = list(iter_abi_spectral_features(path, chunk_rows=2, pixel_step=2))
            np.testing.assert_array_equal(np.concatenate([b[1] for b in blocks]), positions)
            np.testing.assert_array_equal(np.concatenate([b[0] for b in blocks]), expected[:, :10])

    def test_reservoir_is_chunk_invariant_and_capped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "abi.nc"
            write_abi(path)
            first = sample_abi_file(path, 19, 42, chunk_rows=2, pixel_step=2)
            second = sample_abi_file(path, 19, 42, chunk_rows=5, pixel_step=2)
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            self.assertEqual(len(first[0]), 19)
            self.assertGreater(first[2], 19)
            self.assertEqual(len(np.unique(first[1])), 19)


class SyntheticRadiometryTests(unittest.TestCase):
    def setUp(self):
        # Include the full six-channel support with sufficiently fine sampling.
        self.wavelength = np.linspace(2.5, 25, 2251)
        self.responses, self.normalization = abi_response_on_grid(
            self.wavelength, ROOT / "data" / "goes_channels"
        )

    def test_blackbody_roundtrip_without_pi_or_band_unit_error(self):
        error = blackbody_roundtrip(
            self.wavelength, self.responses, self.normalization, PLANCK_G16
        )
        self.assertLess(error, 0.1)

    def test_view_columns_remain_separate_and_linear(self):
        spectra = np.ones((len(self.wavelength), 3)) * [1, 2, 3]
        radiance = convolve_abi_radiances(
            self.wavelength, spectra, self.responses, self.normalization
        )
        self.assertEqual(radiance.shape, (3, 6))
        np.testing.assert_allclose(radiance[1], 2 * radiance[0])
        np.testing.assert_allclose(radiance[2], 3 * radiance[0])
        bt = synthetic_brightness_temperature(radiance, PLANCK_G16)
        self.assertTrue(np.all(bt[2] > bt[1]))
        self.assertTrue(np.all(bt[1] > bt[0]))

    def test_invalid_grid_or_spectrum_fails_explicitly(self):
        with self.assertRaises(ValueError):
            abi_response_on_grid(np.linspace(10, 12, 100), ROOT / "data" / "goes_channels")
        with self.assertRaises(ValueError):
            abi_response_on_grid(self.wavelength[::-1], ROOT / "data" / "goes_channels")
        with self.assertRaises(ValueError):
            convolve_abi_radiances(
                self.wavelength, -np.ones((len(self.wavelength), 18)),
                self.responses, self.normalization,
            )


class ComparisonIntegrationTests(unittest.TestCase):
    def test_serial_and_parallel_gmm_parameters_are_sklearn_compatible(self):
        rng = np.random.default_rng(17)
        features = rng.normal(size=(80, 3))
        for jobs in (1, 2):
            model = ParallelGaussianMixture(
                n_components=2, n_jobs=jobs, n_init=2, random_state=42
            )
            self.assertEqual(model.get_params()["n_jobs"], jobs)
            model.fit(features)
            self.assertTrue(model.converged_)
            self.assertEqual(model.predict_proba(features).shape, (80, 2))

    def test_model_probabilities_share_scene_order(self):
        rng = np.random.default_rng(13)
        bt = np.concatenate((rng.normal(240, 2, (100, 6)), rng.normal(290, 2, (100, 6))))
        features = spectral_features_from_bt(bt)
        model, order = fit_spectral_model(features, n_components=2)
        labels, probabilities, scores = classify_spectral_features(model, order, features)
        self.assertEqual(probabilities.shape, (200, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(labels, probabilities.argmax(axis=1))
        self.assertLess(labels[:100].mean(), labels[100:].mean())
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_end_to_end_outputs_include_all_unaveraged_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            abi = root / "abi.nc"
            write_abi(abi)
            sunny = root / "Sunny"
            wavelength = np.linspace(2.5, 25, 2251)
            # Deliberately distinct views; the flux column must not be used.
            data = np.column_stack((
                wavelength, np.full(len(wavelength), 999.),
                np.ones((len(wavelength), 18)) * np.linspace(1, 8, 18),
            ))
            for folder in ("radiance_lw_cs", "radiance_lw_cl"):
                (sunny / folder).mkdir(parents=True)
                np.savetxt(sunny / folder / f"sunny_lw_{folder[-2:]}_0000", data)
            output = root / "output"
            args = SimpleNamespace(
                abi_files=[abi], points_per_file=100, components=2,
                chunk_rows=3, pixel_step=1, seed=42, gmm_jobs=1,
                sunny_dir=sunny, filter_dir=ROOT / "data" / "goes_channels",
                output_dir=output, max_roundtrip_error_K=0.1,
            )
            summary = run_comparison(args)
            self.assertEqual(summary["abi"]["n_records"], 100)
            self.assertEqual(summary["sunny"]["n_records"], 36)
            self.assertEqual(len(summary["sunny_by_view"]), 18)
            with (output / "sunny_scenes.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 36)
            self.assertEqual([int(r["view_angle_deg"]) for r in rows[:18]], list(range(0, 90, 5)))
            self.assertLess(float(rows[0]["C14_av"]), float(rows[17]["C14_av"]))
            with np.load(output / "sunny_abi_radiances.npz") as arrays:
                self.assertEqual(arrays["radiance"].shape, (2, 18, 6))
                self.assertEqual(arrays["brightness_temperature"].shape, (2, 18, 6))
            bundle = joblib.load(output / "spectral_gmm.joblib")
            self.assertEqual(bundle["training_source"], "ABI only")
            features = np.array([[float(row[name]) for name in bundle["feature_names"]] for row in rows])
            labels, _, _ = classify_spectral_features(bundle["pipeline"], bundle["scene_order"], features)
            np.testing.assert_array_equal(labels, [int(row["scene_id"]) for row in rows])
            with (output / "summary.json").open() as handle:
                self.assertEqual(json.load(handle)["sunny_spectrum_count"], 2)


if __name__ == "__main__":
    unittest.main()
