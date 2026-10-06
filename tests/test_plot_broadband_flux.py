"""Regression checks for the input-only broadband flux plotting CLI."""

import contextlib
import io
import unittest
from unittest.mock import patch

from plotting import plot_broadband_flux


class BroadbandFluxCLITests(unittest.TestCase):
    def test_input_only_uses_plot_defaults(self):
        input_file = "data/broadband_flux/broadband_flux_271_res2km.nc"
        with patch.object(plot_broadband_flux, "plot_broadband_flux") as plot:
            plot_broadband_flux.main(["--input", input_file])
        plot.assert_called_once_with(input_file, "figures/products/broadband_flux", -106)

    def test_renamed_input_and_plot_options(self):
        with patch.object(plot_broadband_flux, "plot_broadband_flux") as plot:
            plot_broadband_flux.main(
                [
                    "--input", "renamed_product.nc",
                    "--output-dir", "custom_figures",
                    "--lambda-center", "-100",
                ]
            )
        plot.assert_called_once_with("renamed_product.nc", "custom_figures", -100.0)

    def test_input_is_required(self):
        stderr = io.StringIO()
        with patch.object(plot_broadband_flux, "plot_broadband_flux") as plot:
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                plot_broadband_flux.main([])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("required: --input", stderr.getvalue())
        plot.assert_not_called()

    def test_removed_selectors_are_rejected(self):
        for option in ("--day", "-d", "--resolution", "-r"):
            with self.subTest(option=option):
                stderr = io.StringIO()
                with patch.object(plot_broadband_flux, "plot_broadband_flux") as plot:
                    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                        plot_broadband_flux.main(["--input", "product.nc", option, "2"])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("unrecognized arguments", stderr.getvalue())
                plot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
