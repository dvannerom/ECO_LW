import numpy as np


def normalize_longitude(longitude):
    """Normalize longitude values to the interval [-180, 180)."""
    return ((longitude + 180.0) % 360.0) - 180.0


def atanh_safe(value):
    """Evaluate atanh while returning zero for values numerically equal to zero."""
    return 0.0 if np.isclose(value, 0.0) else 0.5 * np.log((1.0 + value) / (1.0 - value))
