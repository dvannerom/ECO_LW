import numpy as np


def elmer(viewing_zenith_angle):
    cosine = np.cos(np.radians(viewing_zenith_angle))
    cosine_55 = np.cos(np.radians(55.0))
    return np.log(cosine + 2.0) / np.log(cosine_55 + 2.0)


def radiance_linear(viewing_angle, b):
    return 1.0 + b * (elmer(viewing_angle) - 1.0)


def radiance_integrand(viewing_angle, b):
    """Integrand for F/pi normalization, with viewing_angle in degrees."""
    angle_radians = np.radians(viewing_angle)
    return 2.0 * radiance_linear(viewing_angle, b) * np.cos(angle_radians) * np.sin(angle_radians) * np.radians(1.0)


def radiance_linear_normalized(parameters, a):
    viewing_angle, b = parameters
    return a * radiance_linear(viewing_angle, b)


def radiance_linear_ratio(parameters, b):
    first_angle, second_angle = parameters
    return radiance_linear(first_angle, b) / radiance_linear(second_angle, b)
