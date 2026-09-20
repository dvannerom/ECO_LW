import numpy as np


def elmer(viewing_zenith_angle):
    cosine = np.cos(np.radians(viewing_zenith_angle))
    cosine_55 = np.cos(np.radians(55.0))
    return np.log(cosine + 2.0) / np.log(cosine_55 + 2.0)


def fit_function_difference(viewing_angles, a, b):
    first = a * (elmer(viewing_angles[0]) - 1.0) + b * (elmer(viewing_angles[0]) ** 2 - 1.0)
    second = a * (elmer(viewing_angles[1]) - 1.0) + b * (elmer(viewing_angles[1]) ** 2 - 1.0)
    return first - second


def fit_function_difference_1d(viewing_angle, a, b):
    transformed = elmer(viewing_angle)
    return a * (transformed - 1.0) + b * (transformed ** 2 - 1.0)


def radiance_linear(viewing_angle, b):
    return 1.0 + b * (elmer(viewing_angle) - 1.0)


def radiance_integrand(viewing_angle, b):
    return radiance_linear(viewing_angle, b) * np.sin(np.radians(viewing_angle)) * np.radians(1.0)


def radiance_linear_normalized(parameters, a):
    viewing_angle, b = parameters
    return a * radiance_linear(viewing_angle, b)


def radiance_linear_ratio(parameters, b):
    first_angle, second_angle = parameters
    return radiance_linear(first_angle, b) / radiance_linear(second_angle, b)


def radiance_quadratic(viewing_angle, b, c):
    transformed = elmer(viewing_angle)
    return 1.0 + b * transformed + c * transformed ** 2


def radiance_quadratic_normalized(parameters, a):
    viewing_angle, b, c = parameters
    return a * radiance_quadratic(viewing_angle, b, c)


def radiance_quadratic_ratio(parameters, b, c):
    first_angle, second_angle = parameters
    return radiance_quadratic(first_angle, b, c) / radiance_quadratic(second_angle, b, c)


# Backward-compatible name for the integrands used by the ADM fitting script.
def radiance_norm(parameters, a):
    return radiance_linear_normalized(parameters, a)


def flux_integrand(viewing_angle, a, b):
    return radiance_norm((viewing_angle, b), a) * np.sin(np.radians(viewing_angle))


def flux_upward_integrand(viewing_angle, a, b):
    return radiance_norm((viewing_angle, b), a) * np.cos(np.radians(viewing_angle)) * np.sin(np.radians(viewing_angle))
