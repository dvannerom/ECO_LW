"""Shared Sunny integration is independent of executable plotting/fit scripts."""

import numpy as np

from channel_emulation import load_sunny_scene


def test_sunny_flux_integration_preserves_band_units_and_channel_order(tmp_path):
    wavelength = np.array([4., 5., 6.])
    source = tmp_path / "sunny.txt"
    np.savetxt(source, np.column_stack([wavelength, [3., 3., 3.]]))
    filters = [wavelength] * 6
    responses = [np.full(3, (channel + 1) / 6) for channel in range(6)]
    broadband, narrowband = load_sunny_scene(source, filters, responses)
    assert broadband == 6
    np.testing.assert_allclose(narrowband, np.arange(1, 7))
