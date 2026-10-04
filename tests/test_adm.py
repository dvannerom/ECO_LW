"""Regression checks for physical hemispheric ADM flux normalization.

Run with PYTHONPATH=src:scripts python -m unittest discover -s tests.
"""

import unittest

import numpy as np
from scipy.integrate import quad

from adm import radiance_integrand, radiance_linear
from adm_fitting import correct_radiance, fit_adm_scene
from evaluate_eco_spectral_reconstruction import retrieve_narrowband_fluxes


def physical_flux(amplitude, shape_parameter):
    """Integrate azimuth-independent radiance over the hemisphere in radians."""
    angular_integral = quad(
        lambda angle: radiance_linear(np.degrees(angle), shape_parameter)
        * np.cos(angle)
        * np.sin(angle),
        0.0,
        np.pi / 2.0,
    )[0]
    return 2.0 * np.pi * amplitude * angular_integral


class ADMNormalizationTests(unittest.TestCase):
    def test_isotropic_normalization_is_unity(self):
        integral = quad(radiance_integrand, 0.0, 90.0, args=(0.0,))[0]
        self.assertAlmostEqual(integral, 1.0, places=12)

    def test_abi_fit_recovers_physical_flux(self):
        angles_g16 = np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0])
        angles_g18 = np.array([65.0, 60.0, 50.0, 40.0, 30.0, 20.0])
        amplitudes = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        for shape_parameter in (0.0, -0.3, 0.3, -0.6, 1.0):
            with self.subTest(shape_parameter=shape_parameter):
                radiances_g16 = amplitudes * radiance_linear(angles_g16, shape_parameter)
                radiances_g18 = amplitudes * radiance_linear(angles_g18, shape_parameter)
                fitted_shape, normalization = fit_adm_scene(
                    angles_g16, angles_g18, radiances_g16, radiances_g18
                )
                self.assertAlmostEqual(fitted_shape, shape_parameter, places=7)
                expected_flux = physical_flux(amplitudes, shape_parameter)
                for radiances, angles in (
                    (radiances_g16, angles_g16),
                    (radiances_g18, angles_g18),
                ):
                    retrieved_flux = np.pi * correct_radiance(
                        radiances, angles, fitted_shape, normalization
                    )
                    np.testing.assert_allclose(retrieved_flux, expected_flux, rtol=1e-8)

    def test_eco_retrieval_recovers_physical_flux(self):
        viewing_angles = np.arange(0.0, 75.0, 5.0)
        shape_parameters = np.array([[0.0, -0.3], [0.3, -0.6], [1.0, 0.0]])
        amplitudes = np.array([[10.0, 20.0], [15.0, 30.0], [5.0, 25.0]])
        radiances = amplitudes[:, None, :] * radiance_linear(
            viewing_angles[None, :, None], shape_parameters[:, None, :]
        )
        retrieved_flux, fitted_shapes, residual_rms = retrieve_narrowband_fluxes(
            radiances, viewing_angles
        )
        expected_flux = np.empty_like(amplitudes)
        for scene_index, channel_index in np.ndindex(amplitudes.shape):
            expected_flux[scene_index, channel_index] = physical_flux(
                amplitudes[scene_index, channel_index],
                shape_parameters[scene_index, channel_index],
            )
        np.testing.assert_allclose(retrieved_flux, expected_flux, rtol=1e-12)
        np.testing.assert_allclose(fitted_shapes, shape_parameters, atol=1e-12)
        np.testing.assert_allclose(residual_rms, 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()