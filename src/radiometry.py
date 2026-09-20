import numpy as np


def radiance_to_brightness_temperature(radiance, planck):
    """Convert ABI radiance to brightness temperature using Planck coefficients."""
    fk1, fk2, bc1, bc2 = planck
    ratio = fk1 / radiance
    return (fk2 / np.log(ratio + 1.0) - bc1) / bc2
