"""Checks for shared GOES/ECO cubic N2BC scatter diagnostics."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from plotting import plot_narrowband_to_broadband as plotter


class NarrowbandToBroadbandPlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_dir.cleanup)
        self.directory = Path(self.temporary_dir.name)
        self.residuals = self.directory / "residuals.csv"
        self.columns = [
            "scenario_id", "assessment_stage", "training_mode", "regression_degree",
            "reference_olr_w_m2", "predicted_olr_w_m2",
        ]
        self.rows = [
            ["rfma_goal_6", "n2bc_only", "unstratified", "3", 200, 220],
            ["rfma_goal_6", "n2bc_only", "unstratified", "3", 400, 380],
            ["rfma_goal_6", "end_to_end", "unstratified", "3", 200, 999],
            ["rfma_goal_6", "n2bc_only", "unstratified", "1", 200, 999],
            ["rfma_goal_6", "n2bc_only", "clear_cloud_stratified", "3", 200, 201],
            ["rfma_threshold_6", "n2bc_only", "unstratified", "3", 200, 202],
        ]
        with self.residuals.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.columns)
            writer.writerows(self.rows)

    def test_eco_selects_only_requested_cubic_n2bc_predictions(self):
        for scenario, mode, expected in (
            ("rfma_goal_6", "unstratified", [220, 380]),
            ("rfma_goal_6", "clear_cloud_stratified", [201]),
            ("rfma_threshold_6", "unstratified", [202]),
        ):
            with self.subTest(scenario=scenario, mode=mode):
                reference, prediction = plotter.load_eco_flux_predictions(
                    self.residuals, scenario, mode
                )
                np.testing.assert_array_equal(prediction, expected)
                self.assertEqual(reference.shape, prediction.shape)

    def test_eco_missing_groups_and_columns_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "No cubic N2BC-only"):
            plotter.load_eco_flux_predictions(self.residuals, "missing", "unstratified")
        self.residuals.write_text("scenario_id\nrfma_goal_6\n")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            plotter.load_eco_flux_predictions(self.residuals, "rfma_goal_6", "unstratified")

    def test_shared_plot_preserves_sign_and_mean_centering(self):
        figure, axis = MagicMock(), MagicMock()
        with patch.object(plotter.plt, "subplots", return_value=(figure, axis)):
            with patch.object(plotter.plt, "close"):
                plotter.plot_flux_errors(
                    np.array([200.0, 400.0]), np.array([220.0, 380.0]),
                    self.directory / "plot.png", "test",
                )
        x_values, errors, marker = axis.plot.call_args.args
        np.testing.assert_array_equal(x_values, [220, 380])
        np.testing.assert_allclose(errors, [-7.5, 7.5])
        self.assertEqual(marker, ".")
        figure.savefig.assert_called_once_with(self.directory / "plot.png", dpi=150)

    def test_invalid_fluxes_fail_before_plotting(self):
        for reference, prediction in (
            ([0], [1]), ([np.nan], [1]), ([1], [np.inf]),
            ([1, 2], [1]), ([], []), ([[1]], [[1]]),
        ):
            with self.subTest(reference=reference, prediction=prediction):
                with self.assertRaisesRegex(ValueError, "Flux arrays"):
                    plotter.plot_flux_errors(
                        np.asarray(reference), np.asarray(prediction),
                        self.directory / "invalid.png", "invalid",
                    )
        self.assertFalse((self.directory / "invalid.png").exists())

    def test_goes_prediction_math_is_unchanged(self):
        coefficients = self.directory / "coefficients.json"
        number_of_features = plotter.PolynomialFeatures(
            degree=3, include_bias=False
        ).fit(np.ones((1, len(plotter.CHANNELS)))).n_output_features_
        coefficients.write_text(json.dumps({
            "cubic": {"coefficients": [0.0] * number_of_features, "intercept": 280.0}
        }))
        for name in ("radiance_lw_cs", "radiance_lw_cl"):
            folder = self.directory / name
            folder.mkdir()
            (folder / "scene").touch()
        wavelengths = [np.array([6000.0, 7000.0])] * len(plotter.CHANNELS)
        responses = [np.ones(2)] * len(plotter.CHANNELS)
        with patch.object(plotter, "load_goes_filters", return_value=(wavelengths, responses)):
            with patch.object(
                plotter, "load_sunny_scene",
                return_value=(300.0, np.ones(len(plotter.CHANNELS))),
            ) as load_scene:
                reference, prediction = plotter.load_goes_flux_predictions(
                    coefficients, self.directory, "filters",
                    self.directory / "missing_power_law.json",
                )
        np.testing.assert_array_equal(reference, [300.0, 300.0])
        np.testing.assert_allclose(prediction, plotter.SIGMA * 280.0 ** 4)
        np.testing.assert_array_equal(load_scene.call_args.args[1][0], [6.0, 7.0])

    def test_eco_writes_a_real_png_without_loading_goes(self):
        with patch.object(plotter, "load_goes_flux_predictions") as goes:
            output = plotter.plot_narrowband_to_broadband(
                sensor="eco", residuals_file=self.residuals, output_dir=self.directory
            )
        goes.assert_not_called()
        self.assertEqual(
            output.name,
            "narrowband_to_broadband_cubic_fit_error_eco_rfma_goal_6_unstratified.png",
        )
        with output.open("rb") as handle:
            self.assertEqual(handle.read(8), b"\x89PNG\r\n\x1a\n")

    def test_goes_default_retains_filename_and_uses_shared_plot(self):
        fluxes = (np.array([200.0]), np.array([201.0]))
        with patch.object(plotter, "load_goes_flux_predictions", return_value=fluxes):
            with patch.object(plotter, "plot_flux_errors") as plot:
                plotter.plot_narrowband_to_broadband(output_dir=self.directory)
        self.assertEqual(
            plot.call_args.args[2],
            self.directory / "narrowband_to_broadband_cubic_fit_error.png",
        )

    def test_cli_routes_eco_and_rejects_wrong_sensor_options(self):
        with patch.object(plotter, "plot_narrowband_to_broadband") as plot:
            with contextlib.redirect_stdout(io.StringIO()):
                plotter.main(["--sensor", "eco", "--scenario", "rfma_threshold_6"])
        self.assertEqual(plot.call_args.kwargs["sensor"], "eco")
        self.assertEqual(plot.call_args.kwargs["scenario"], "rfma_threshold_6")
        for arguments in (
            ["--sensor", "eco", "--filter-dir", "filters"],
            ["--sensor", "goes", "--residuals-file", "errors.csv"],
            ["--sensor", "invalid"],
        ):
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        plotter.main(arguments)
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
