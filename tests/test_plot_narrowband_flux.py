"""CLI checks for offline narrowband spectral-irradiance maps."""

import contextlib
import io
import unittest
from unittest.mock import patch

from plotting import plot_narrowband_flux


class NarrowbandFluxCLITests(unittest.TestCase):
    def test_input_only_uses_abi_proxy_product_directory(self):
        input_file = "data/abi_proxy/products/narrowband_flux/narrowband_flux_271_res2km_C0.nc"
        with patch.object(plot_narrowband_flux, "plot_narrowband_irradiance") as plot:
            plot_narrowband_flux.main(["--input", input_file])
        plot.assert_called_once_with(
            input_file,
            "figures/abi_proxy/products/narrowband_flux",
            -106,
        )

    def test_custom_output_options(self):
        with patch.object(plot_narrowband_flux, "plot_narrowband_irradiance") as plot:
            plot_narrowband_flux.main(
                ["--input", "product.nc", "--output-dir", "custom", "--lambda-center", "-100"]
            )
        plot.assert_called_once_with("product.nc", "custom", -100.0)

    def test_input_is_required(self):
        stderr = io.StringIO()
        with patch.object(plot_narrowband_flux, "plot_narrowband_irradiance") as plot:
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                plot_narrowband_flux.main([])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("required: --input", stderr.getvalue())
        plot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
