"""Streaming operators for the production-aligned sensitivity study.

Every ABI variant is the production chain (abi_product) with exactly one
model choice changed, evaluated on every native pixel of every configured day
and paired with the recomputed production baseline. Only sufficient
statistics are kept:

* per-group (count, mean, M2) of paired differences d = F_variant - F_baseline
  [W m-2], merged with Chan's parallel update, so memory is O(groups);
* per 10 km cell daily sum(d) and count, so exact monthly mean-difference
  maps follow from summing days without storing any flux field.

No function here reads files; the runner streams row slabs through them.
"""

import numpy as np

from abi_product import broadband_flux, satellite_chain
from scene_features import SPECTRAL_CHANNEL_INDICES, spectral_features_from_bt


class GroupMoments:
    """Exact streamed count/mean/M2 per integer group (float64)."""

    def __init__(self, groups):
        self.count = np.zeros(groups, dtype=np.int64)
        self.mean = np.zeros(groups)
        self.m2 = np.zeros(groups)

    def update(self, groups, values):
        """Add values[n] (finite) belonging to groups[n] in [0, groups)."""
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("Non-finite paired differences")
        size = len(self.count)
        count = np.bincount(groups, minlength=size)
        if len(count) != size:
            raise ValueError("Group index out of range")
        mean = np.zeros(size)
        present = count > 0
        mean[present] = np.bincount(groups, weights=values, minlength=size)[present] / count[present]
        m2 = np.bincount(groups, weights=np.square(values - mean[groups]), minlength=size)
        self.merge(count, mean, m2)

    def merge(self, count, mean, m2):
        count = np.asarray(count, dtype=np.int64)
        mean, m2 = np.asarray(mean, dtype=np.float64), np.asarray(m2, dtype=np.float64)
        total = self.count + count
        present = total > 0
        delta = mean - self.mean
        ratio = np.zeros(len(total))
        ratio[present] = count[present] / total[present]
        self.m2 = self.m2 + m2 + np.square(delta) * self.count * ratio
        self.mean = self.mean + delta * ratio
        self.count = total

    def state(self):
        return {"count": self.count.tolist(), "mean": self.mean.tolist(), "m2": self.m2.tolist()}

    @classmethod
    def from_state(cls, state):
        moments = cls(len(state["count"]))
        moments.merge(state["count"], state["mean"], state["m2"])
        return moments

    def summary(self, reference_mean=None):
        """Per-group bias, population SD and RMSE [W m-2]; percent if reference given."""
        rows = []
        for index, count in enumerate(self.count):
            if not count:
                rows.append({"count": 0})
                continue
            bias = float(self.mean[index])
            sd = float(np.sqrt(self.m2[index] / count))
            row = {"count": int(count), "bias_w_m2": bias, "sd_w_m2": sd,
                   "rmse_w_m2": float(np.hypot(bias, sd))}
            if reference_mean is not None and reference_mean[index] > 0:
                scale = 100. / reference_mean[index]
                row.update(bias_percent=bias * scale, sd_percent=sd * scale,
                           rmse_percent=row["rmse_w_m2"] * scale)
            rows.append(row)
        return rows


def cell_sums(values, valid, block):
    """Sum and count of values[rows, cols] over valid pixels in block x block cells.

    The slab must start on a cell boundary. Partial edge cells keep only
    their in-grid pixels. Returns (sums[cy, cx] float64, counts[cy, cx] int64).
    """
    rows, width = values.shape
    cells_y, cells_x = -(-rows // block), -(-width // block)
    padded = np.zeros((cells_y * block, cells_x * block))
    padded[:rows, :width] = np.where(valid, values, 0.)
    sums = padded.reshape(cells_y, block, cells_x, block).sum(axis=(1, 3))
    padded[:] = 0.
    padded[:rows, :width] = valid
    counts = padded.reshape(cells_y, block, cells_x, block).sum(axis=(1, 3)).astype(np.int64)
    return sums, counts


def vza_bins(angles, edges):
    """Bin index[n] for angles[n] in degrees; values outside edges get len(edges)-1."""
    edges = np.asarray(edges, dtype=np.float64)
    index = np.searchsorted(edges, angles, side="right") - 1
    index[angles == edges[-1]] = len(edges) - 2
    index[(index < 0) | (index > len(edges) - 2) | ~np.isfinite(angles)] = len(edges) - 1
    return index


def production_flux(radiance, angles, labels, planck, library, form, integral, regression,
                    max_vza_deg):
    """Production flux[n] [W m-2] and bt[n, 6] [K] including float32 BT storage.

    Production writes narrowband BTs as float32 and evaluates N2B on them, so
    the BT is rounded to float32 here too; the baseline then closes on the
    written daily product to float32 flux rounding.
    """
    bt, _ = satellite_chain(radiance, angles, labels, planck, library, form, integral,
                            None, max_vza_deg)
    bt = bt.astype(np.float32).astype(np.float64)
    return bt, n2b_flux(bt, regression)


def n2b_flux(bt, regression):
    """Broadband flux[n] [W m-2] from bt[n, 6]; NaN where any BT is NaN."""
    flux = np.full(len(bt), np.nan)
    finite = np.all(np.isfinite(bt), axis=1)
    if finite.any():
        flux[finite] = broadband_flux(bt[finite], regression)
    return flux


def slab_spectral_features(source, rows):
    """Scene-identification features for a row slab, as in scene_id_pca.py.

    Returns (features[m, 10], valid[n]) with n = rows x columns; validity is
    all six G16 and G18 BTs finite and in (0, 1000) K.
    """
    channels = list(SPECTRAL_CHANNEL_INDICES)
    g16 = np.ma.filled(source.variables["BT_G16_interp"][rows, :, channels].astype(np.float32), np.nan)
    g18 = np.ma.filled(source.variables["BT_G18_interp"][rows, :, channels].astype(np.float32), np.nan)
    g16, g18 = g16.reshape(-1, 6), g18.reshape(-1, 6)
    valid = np.all(np.isfinite(g16) & np.isfinite(g18) & (g16 > 0) & (g16 < 1000)
                   & (g18 > 0) & (g18 < 1000), axis=1)
    return spectral_features_from_bt((g16[valid] + g18[valid]) / 2.0), valid


def fit_n2bc_huber(flux, reference, weights, degree, cutoff, max_iterations, tolerance):
    """Population-weighted Huber IRLS N2BC (scaler, poly, regression).

    Starts from the weighted least-squares fit. The Huber scale s = 1.4826 MAD
    of those residuals is then held fixed (re-estimating it each iteration
    collapses it for flexible models), so the convex Huber loss is minimised
    by IRLS with weights = population weight * min(1, c*s/|r|).
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    scaler = StandardScaler().fit(flux, sample_weight=weights)
    poly = PolynomialFeatures(degree, include_bias=False)
    design = poly.fit_transform(scaler.transform(flux))
    regression = LinearRegression().fit(design, reference, sample_weight=weights)
    residual = reference - regression.predict(design)
    scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-8)
    for _ in range(max_iterations):
        previous = regression.predict(design)
        residual = reference - previous
        huber = np.minimum(1., cutoff * scale / np.maximum(np.abs(residual), 1e-12))
        regression = LinearRegression().fit(design, reference, sample_weight=weights * huber)
        if np.max(np.abs(regression.predict(design) - previous)) < tolerance:
            return scaler, poly, regression
    raise RuntimeError("Huber N2BC did not converge")
