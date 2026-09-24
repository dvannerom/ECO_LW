"""Shared cubic narrowband-to-broadband regression, matching narrowband_to_broadband.py."""

import json

import numpy as np

# GOES channels 08, 11, 12, 14, 15, 16, fitted against reference broadband flux.
COEFF_CUBIC = [
    4.69962675e+00, 1.62183406e+01, 7.61262687e-01, -8.92166533e+00,
    -4.86682683e+00, -1.14231072e+01, -9.45069025e-02, 2.50272570e-01,
    -6.50158635e-02, -2.75717730e-01, -2.60760940e-02, 2.61120108e-01,
    -1.52914394e-01, 1.53894060e-01, -4.17800377e-01, 1.34917427e+00,
    -1.18213776e+00, -3.57863341e-02, 2.01244030e-03, -2.27775285e-01,
    2.00477535e-01, 1.35122229e+00, -3.38625494e+00, 1.45998620e+00,
    1.33664170e+00, -3.39776272e-01, -1.42809106e-01, 1.34300203e-04,
    -5.90140657e-04, 5.23752905e-05, 6.85399660e-04, -7.85379265e-05,
    -8.49371329e-05, -1.12610983e-03, -8.95732635e-05, -9.87184197e-04,
    3.26488292e-03, 1.53438709e-04, 1.54233887e-04, -8.07753393e-04,
    8.23816754e-04, -5.41931708e-05, 5.54989968e-03, -1.15234035e-02,
    2.01203872e-03, 4.83026715e-03, -1.93513714e-03, -5.30714915e-04,
    -5.90870290e-03, -1.42607367e-03, 4.87879796e-02, -3.76451896e-02,
    9.61241946e-03, 3.47400352e-04, 3.55688764e-03, -8.14456940e-05,
    -1.92873252e-03, -1.08799800e-01, 1.45713496e-01, -2.63262126e-02,
    -4.05851615e-02, 1.82810953e-03, 6.20333119e-03, 9.08135558e-05,
    -1.11355453e-03, 9.35698960e-04, -4.23572660e-04, 1.61386994e-03,
    -8.46903041e-03, 4.58617016e-03, 4.59084130e-03, -2.15894325e-03,
    -2.58627281e-04, 6.59718996e-02, -1.07547359e-01, 6.20092029e-03,
    3.84030469e-02, 2.54580565e-02, -1.21409230e-02, 1.93717934e-03,
    -1.79004995e-02, 6.84997236e-03, 1.47793639e-04,
]
INTERCEPT_CUBIC = 384.61181077479637


def cubic_regression(rad, coeff=COEFF_CUBIC, intercept=INTERCEPT_CUBIC):
    """Map six-channel narrowband brightness temperatures to an effective broadband value."""
    x = [rad[:, i] for i in range(6)]
    out = np.full(rad.shape[0], intercept, dtype=rad.dtype)

    k = 0
    for i in range(6):
        out += coeff[k] * x[i]
        k += 1

    for i in range(6):
        for j in range(i, 6):
            out += coeff[k] * x[i] * x[j]
            k += 1

    for i in range(6):
        for j in range(i, 6):
            xij = x[i] * x[j]
            for l in range(j, 6):
                out += coeff[k] * xij * x[l]
                k += 1

    return out


def load_cubic_coefficients(path):
    """Load cubic regression coefficients fitted by scripts/compute_temperature_SBDART.py.

    :param path: (str or Path) JSON file with a "cubic" key holding
        "coefficients" (list of 27 floats) and "intercept" (float).
    :return: (list[float], float) coefficients, intercept, ready for cubic_regression().
    """
    with open(path) as handle:
        data = json.load(handle)
    cubic = data["cubic"]
    return cubic["coefficients"], cubic["intercept"]
