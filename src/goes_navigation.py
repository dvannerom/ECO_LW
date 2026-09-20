import math

import numpy as np


def compute_rs(x, y, height, equatorial_radius, polar_radius):
    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (
        np.cos(y) ** 2 + (equatorial_radius / polar_radius * np.sin(y)) ** 2
    )
    b = -2.0 * height * np.cos(x) * np.cos(y)
    c = height ** 2 - equatorial_radius ** 2
    discriminant = b ** 2 - 4.0 * a * c
    return np.where(discriminant >= 0.0, (-b - np.sqrt(np.maximum(discriminant, 0.0))) / (2.0 * a), 0.0)


def compute_sx_1(radius_s, x, y):
    return radius_s * np.cos(x) * np.cos(y)


def compute_sy_1(radius_s, x):
    return -radius_s * np.sin(x)


def compute_sz_1(radius_s, x, y):
    return radius_s * np.cos(x) * np.sin(y)


def compute_latitude(sx, sy, sz, height, equatorial_radius, polar_radius):
    return np.arctan(
        (equatorial_radius / polar_radius) ** 2
        * sz
        / np.sqrt((height - sx) ** 2 + sy ** 2)
    )


def compute_longitude(sx, sy, height, lambda_0):
    return lambda_0 - np.arctan(sy / (height - sx))


def compute_phi_c(polar_radius, equatorial_radius, latitude):
    return np.arctan((polar_radius / equatorial_radius) ** 2 * np.tan(latitude))


def compute_radius_c(polar_radius, eccentricity, phi_c):
    return polar_radius / np.sqrt(1.0 - (eccentricity * np.cos(phi_c)) ** 2)


def compute_sx_2(height, radius_c, phi_c, longitude, lambda_0):
    return height - radius_c * np.cos(phi_c) * np.cos(longitude - lambda_0)


def compute_sy_2(radius_c, phi_c, longitude, lambda_0):
    return -radius_c * np.cos(phi_c) * np.sin(longitude - lambda_0)


def compute_sz_2(radius_c, phi_c):
    return radius_c * np.sin(phi_c)


def compute_fixed_grid_x(sx, sy, sz):
    return np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))


def compute_fixed_grid_y(sx, sz):
    return np.arctan(sz / sx)


def compute_x_sinusoidal(latitude, longitude, lambda_0):
    return (longitude - lambda_0) * np.cos(latitude)


def compute_longitude_sinusoidal(x, latitude, lambda_0):
    return x / np.cos(latitude) + lambda_0


def compute_local_zenith_angle(height, latitude, longitude, lambda_0, equatorial_radius):
    beta = np.arccos(np.cos(latitude) * np.cos(longitude - lambda_0))
    numerator = height * np.sin(beta)
    denominator = np.sqrt(
        height ** 2 + equatorial_radius ** 2
        - 2.0 * height * equatorial_radius * np.cos(beta)
    )
    return np.arcsin(numerator / denominator)


def latlon_to_xy_goes(latitude, longitude, goes_imager_projection):
    """Convert latitude/longitude arrays to GOES fixed-grid coordinates."""
    lambda_0 = math.radians(goes_imager_projection.longitude_of_projection_origin)
    r_eq = goes_imager_projection.semi_major_axis
    r_pol = goes_imager_projection.semi_minor_axis
    height = goes_imager_projection.perspective_point_height + r_eq
    flattening = 1.0 / goes_imager_projection.inverse_flattening
    eccentricity = math.sqrt(flattening * (2.0 - flattening))

    phi_c = compute_phi_c(r_pol, r_eq, latitude)
    radius_c = compute_radius_c(r_pol, eccentricity, phi_c)
    sx_2 = compute_sx_2(height, radius_c, phi_c, longitude, lambda_0)
    sy_2 = compute_sy_2(radius_c, phi_c, longitude, lambda_0)
    sz_2 = compute_sz_2(radius_c, phi_c)
    return compute_fixed_grid_x(sx_2, sy_2, sz_2), compute_fixed_grid_y(sx_2, sz_2)


def parallax_correction(latitude, longitude, cloud_top_height, goes_imager_projection):
    """Correct geolocation for cloud-top height in the ABI viewing geometry."""
    altitude = np.nan_to_num(cloud_top_height)
    lambda_0 = math.radians(goes_imager_projection.longitude_of_projection_origin)
    r_eq = goes_imager_projection.semi_major_axis
    r_pol = goes_imager_projection.semi_minor_axis
    height = goes_imager_projection.perspective_point_height + r_eq

    radius_ratio = r_eq / r_pol
    latitude_geocentric = np.arctan(np.tan(latitude) / radius_ratio ** 2)
    radius_local = r_eq / np.sqrt(
        np.cos(latitude_geocentric) ** 2
        + (radius_ratio * np.sin(latitude_geocentric)) ** 2
    )
    x_goes = radius_local * np.cos(latitude_geocentric) * np.cos(longitude - lambda_0)
    y_goes = radius_local * np.cos(latitude_geocentric) * np.sin(longitude - lambda_0)
    z_goes = radius_local * np.sin(latitude_geocentric)

    x_difference = height - x_goes
    y_difference = -y_goes
    z_difference = -z_goes
    local_radius_ratio = ((r_eq + altitude) / (r_pol + altitude)) ** 2
    quadratic_a = x_difference ** 2 + y_difference ** 2 + local_radius_ratio * z_difference ** 2
    quadratic_b = 2.0 * (x_goes * x_difference + y_goes * y_difference + local_radius_ratio * z_goes * z_difference)
    quadratic_c = x_goes ** 2 + y_goes ** 2 + local_radius_ratio * z_goes ** 2 - (r_eq + altitude) ** 2
    correction = (np.sqrt(quadratic_b ** 2 - 4.0 * quadratic_a * quadratic_c) - quadratic_b) / (2.0 * quadratic_a)

    x_corrected = x_goes + correction * x_difference
    y_corrected = y_goes + correction * y_difference
    z_corrected = z_goes + correction * z_difference
    corrected_latitude = np.arctan2(
        local_radius_ratio * z_corrected,
        np.sqrt(x_corrected ** 2 + y_corrected ** 2),
    )
    corrected_longitude = np.arctan2(y_corrected, x_corrected) + lambda_0
    return corrected_latitude, corrected_longitude


def xy_goes_to_latlon_grid(x_goes, y_goes, goes_imager_projection, cloud_top_height):
    """Convert GOES fixed-grid coordinates to geographic and parallax-corrected grids."""
    altitude = np.nan_to_num(cloud_top_height)
    lambda_0 = math.radians(goes_imager_projection.longitude_of_projection_origin)
    r_eq = goes_imager_projection.semi_major_axis
    r_pol = goes_imager_projection.semi_minor_axis
    height = goes_imager_projection.perspective_point_height + r_eq

    radius_s = compute_rs(x_goes, y_goes, height, r_eq, r_pol)
    sx_1 = compute_sx_1(radius_s, x_goes, y_goes)
    sy_1 = compute_sy_1(radius_s, x_goes)
    sz_1 = compute_sz_1(radius_s, x_goes, y_goes)
    latitude_goes = compute_latitude(sx_1, sy_1, sz_1, height, r_eq, r_pol)
    longitude_goes = compute_longitude(sx_1, sy_1, height, lambda_0)
    local_zenith_angle = compute_local_zenith_angle(height, latitude_goes, longitude_goes, lambda_0, r_eq)

    radius_ratio = r_eq / r_pol
    latitude_geocentric = np.arctan(np.tan(latitude_goes) / radius_ratio ** 2)
    radius_local = r_eq / np.sqrt(
        np.cos(latitude_geocentric) ** 2
        + (radius_ratio * np.sin(latitude_geocentric)) ** 2
    )
    x_goes_cartesian = radius_local * np.cos(latitude_geocentric) * np.cos(longitude_goes - lambda_0)
    y_goes_cartesian = radius_local * np.cos(latitude_geocentric) * np.sin(longitude_goes - lambda_0)
    z_goes_cartesian = radius_local * np.sin(latitude_geocentric)

    x_difference = height - x_goes_cartesian
    y_difference = -y_goes_cartesian
    z_difference = -z_goes_cartesian
    local_radius_ratio = ((r_eq + altitude) / (r_pol + altitude)) ** 2
    quadratic_a = x_difference ** 2 + y_difference ** 2 + local_radius_ratio * z_difference ** 2
    quadratic_b = 2.0 * (x_goes_cartesian * x_difference + y_goes_cartesian * y_difference + local_radius_ratio * z_goes_cartesian * z_difference)
    quadratic_c = x_goes_cartesian ** 2 + y_goes_cartesian ** 2 + local_radius_ratio * z_goes_cartesian ** 2 - (r_eq + altitude) ** 2
    correction = (np.sqrt(quadratic_b ** 2 - 4.0 * quadratic_a * quadratic_c) - quadratic_b) / (2.0 * quadratic_a)

    x_corrected = x_goes_cartesian + correction * x_difference
    y_corrected = y_goes_cartesian + correction * y_difference
    z_corrected = z_goes_cartesian + correction * z_difference
    corrected_latitude = np.arctan2(
        local_radius_ratio * z_corrected,
        np.sqrt(x_corrected ** 2 + y_corrected ** 2),
    )
    corrected_longitude = np.arctan2(y_corrected, x_corrected) + lambda_0

    return (
        np.degrees(latitude_goes),
        np.degrees(longitude_goes),
        np.degrees(corrected_latitude),
        np.degrees(corrected_longitude),
        np.degrees(local_zenith_angle),
    )
