"""Canonical scene-classification feature construction."""

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

# Spatial blocks centered on each pixel.
SPATIAL_HALF_WINDOW = 2
SPATIAL_HALF_WINDOW_9X9 = 4

# Channel labels for the 6 averaged fields, in build_scene_features() column order.
CHANNEL_LABELS = ("C08", "C11", "C12", "C14", "C15", "C16")
SPECTRAL_CHANNEL_INDICES = (0, 3, 4, 6, 7, 8)
# BTD labels, in build_scene_features() column order.
DIFFERENCE_LABELS = ("BTD14-11", "BTD14-15", "BTD14-08", "BTD14-16")
# Spatial standard-deviation labels, in build_scene_features() column order.
LOCAL_STD_5X5_LABELS = tuple(f"{label}_std5x5" for label in CHANNEL_LABELS)
LOCAL_STD_9X9_LABELS = tuple(f"{label}_std9x9" for label in CHANNEL_LABELS)

# Column slices of the build_scene_features() feature array, single source of truth
# for anything downstream that needs to name or split feature columns by group.
FEATURE_GROUPS = {
    "averages": slice(0, 6),
    "differences": slice(6, 10),
    "local_std_5x5": slice(10, 16),
    "local_std_9x9": slice(16, 22),
}


def feature_names():
    """Return the 22 column names of build_scene_features(), in column order."""
    return (
        [f"{label}_av" for label in CHANNEL_LABELS]
        + list(DIFFERENCE_LABELS)
        + list(LOCAL_STD_5X5_LABELS)
        + list(LOCAL_STD_9X9_LABELS)
    )


def spectral_feature_names():
    """Return the ten spectral-only feature names in production column order."""
    return feature_names()[:10]


def spectral_features_from_bt(brightness_temperature):
    """Build spectral features from BT[K] with shape ``(..., 6)``.

    ABI supplies the arithmetic mean of G16/G18 BTs. Sunny supplies one
    directional view directly, treated as the equivalent mean-view input.
    The result has shape ``(..., 10)``; no spatial texture is synthesized.
    """
    bt = np.asarray(brightness_temperature)
    if bt.ndim < 1 or bt.shape[-1] != len(CHANNEL_LABELS):
        raise ValueError("Expected six BT channels in C08/C11/C12/C14/C15/C16 order")
    differences = bt[..., 3, None] - bt[..., (1, 4, 0, 5)]
    return np.concatenate((bt, differences), axis=-1)


def iter_abi_spectral_features(input_file, chunk_rows=64, pixel_step=10):
    """Yield ``(features[n, 10], flat_pixel_indices[n])`` from disk-backed ABI.

    Only selected channels and sampled pixels are loaded. No full-grid arrays,
    geographic coordinates, halos, or spatial statistics are materialized.
    """
    if chunk_rows < 1 or pixel_step < 1:
        raise ValueError("chunk_rows and pixel_step must be positive")
    with xr.open_dataset(input_file, engine="netcdf4") as dataset:
        bt16 = dataset["BT_G16_interp"]
        bt18 = dataset["BT_G18_interp"]
        if bt16.shape != bt18.shape or bt16.ndim != 3 or bt16.shape[-1] != 9:
            raise ValueError(f"Expected matching (rows, cols, 9) ABI BT arrays in {input_file}")
        height, width, _ = bt16.shape
        for start in range(0, height, chunk_rows * pixel_step):
            selection = (
                slice(start, min(start + chunk_rows * pixel_step, height), pixel_step),
                slice(None, None, pixel_step),
                list(SPECTRAL_CHANNEL_INDICES),
            )
            g16 = bt16[selection].values.astype(np.float32, copy=False)
            g18 = bt18[selection].values.astype(np.float32, copy=False)
            valid = np.all(
                np.isfinite(g16) & np.isfinite(g18)
                & (g16 > 0) & (g16 < 1000) & (g18 > 0) & (g18 < 1000),
                axis=-1,
            )
            if not np.any(valid):
                continue
            rows, columns = np.nonzero(valid)
            positions = (start + rows * pixel_step) * width + columns * pixel_step
            averages = (g16[valid] + g18[valid]) / 2.0
            yield spectral_features_from_bt(averages), positions


def _local_valid_stats(values, valid, half_window=SPATIAL_HALF_WINDOW):
    """NaN-aware local mean/stddev over a ``(2*half_window+1)^2`` block.

    Parameters
    ----------
    values : numpy.ndarray
        Fields with shape ``(n_fields, rows, cols)``.
    valid : numpy.ndarray
        Boolean validity mask with shape ``(rows, cols)``, shared across
        fields.

    Returns
    -------
    mean, std : numpy.ndarray
        Local statistics with shape ``(n_fields, rows, cols)``. Blocks with
        zero valid neighbors are set to NaN.
    """
    size = 2 * half_window + 1
    valid_f = valid.astype(np.float32)
    # mode="constant", cval=0 treats out-of-image neighbors as invalid (no wraparound).
    count = uniform_filter(valid_f, size=size, mode="constant", cval=0.0) * (size * size)
    safe_count = np.where(count > 0, count, 1.0)

    filled = np.where(valid, values, 0.0).astype(np.float32, copy=False)
    valid_count = np.count_nonzero(valid)
    reference = np.sum(filled, axis=(1, 2), dtype=np.float64, keepdims=True)
    reference /= max(valid_count, 1)
    reference = reference.astype(np.float32, copy=False)
    filled -= reference
    filled[:, ~valid] = 0.0
    window = (1, size, size)
    sum1 = uniform_filter(filled, size=window, mode="constant", cval=0.0) * (size * size)
    sum2 = uniform_filter(filled * filled, size=window, mode="constant", cval=0.0) * (size * size)

    mean_offset = sum1 / safe_count
    variance = np.maximum(sum2 / safe_count - mean_offset * mean_offset, 0.0)
    mean = mean_offset + reference
    std = np.sqrt(variance, dtype=np.float32)

    no_data = count == 0
    mean[:, no_data] = np.nan
    std[:, no_data] = np.nan
    return mean, std


def scene_label_sort_order(means_scaled, method="c14_btd14_08"):
    """Return component order from cold/icy to hot clear-sky land.

    Parameters
    ----------
    means_scaled : numpy.ndarray
        Component means in standardized feature space with shape
        ``(n_components, n_features)``. Indices 3 (C14 average) and 8
        (BTD14-08) are used regardless of how many extra features follow.
    method : str
        ``"c14"`` sorts by standardized C14. ``"c14_btd14_08"`` sorts
        by standardized C14 plus BTD14-08.
    """
    if method == "c14":
        score = means_scaled[:, 3]
    elif method == "c14_btd14_08":
        score = means_scaled[:, 3] + means_scaled[:, 8]
    else:
        raise ValueError(f"Unknown scene label ordering: {method}")
    return np.argsort(score)


def scene_label_mapping(means_scaled, method="c14_btd14_08"):
    """Return a mapping from GMM component IDs to ordered scene IDs."""
    sort_idx = scene_label_sort_order(means_scaled, method)
    return {component: scene_id for scene_id, component in enumerate(sort_idx)}


def build_scene_features(input_file, chunk_rows=512, pixel_step=1, chunk_callback=None):
    """Build valid-pixel scene features using the production feature schema.

    Feature layout (10 spectral + 12 spatial = 22 columns):
    ``[6 channel averages, 4 BT differences, 6 local 5x5 stddevs, 6 local 9x9
    stddevs]``. The local mean/stddev are computed at native (full) resolution
    over the 6 channel-average fields, using 5x5 and 9x9 blocks of valid
    neighbors, then subsampled by ``pixel_step`` together with everything
    else, so the window always covers the true native-pixel neighborhood
    regardless of ``pixel_step``. The local means are used only to build the
    validity mask and are not included in the returned features.

    If ``chunk_callback`` is provided, it is called with each feature block and
    its flattened source-pixel indices. In this streaming mode, feature blocks
    are not accumulated and the function returns ``None``.

    Returns
    -------
    features : numpy.ndarray
        Valid pixel features with shape ``(n_valid, 22)``.
    valid_flat : numpy.ndarray
        Flattened indices of valid pixels in the source grid.
    flat_mask : numpy.ndarray
        Boolean validity mask with the source grid shape.
    lat, lon : numpy.ndarray
        Source latitude and longitude grids.
    """
    if pixel_step < 1:
        raise ValueError("pixel_step must be at least 1")

    # BT_G16_interp/BT_G18_interp are ~3.8 GB each; keep them disk-backed and only
    # materialize one row-chunk (+halo) at a time instead of the full array.
    with xr.open_dataset(input_file, engine="netcdf4") as dataset:
        if chunk_callback is None:
            lat = dataset["lat_interp_grid"].values
            lon = dataset["lon_interp_grid"].values
        else:
            lat = lon = None
        bt_g16 = dataset["BT_G16_interp"]
        bt_g18 = dataset["BT_G18_interp"]
        height, width = dataset["lat_interp_grid"].shape
        flat_mask = np.zeros((height, width), dtype=bool) if chunk_callback is None else None
        valid_features = [] if chunk_callback is None else None
        valid_positions = [] if chunk_callback is None else None

        row_block = chunk_rows * pixel_step
        for row_start in range(0, height, row_block):
            row_end = min(row_start + row_block, height)
            # Halo rows (native resolution) so both spatial windows are correct at chunk edges.
            halo_size = max(SPATIAL_HALF_WINDOW, SPATIAL_HALF_WINDOW_9X9)
            halo_start = max(row_start - halo_size, 0)
            halo_end = min(row_end + halo_size, height)

            bt_g16_halo = bt_g16[halo_start:halo_end].values
            bt_g18_halo = bt_g18[halo_start:halo_end].values

            channels_g16 = [
                bt_g16_halo[:, :, index].astype(np.float32, copy=False)
                for index in (0, 3, 4, 6, 7, 8)
            ]
            channels_g18 = [
                bt_g18_halo[:, :, index].astype(np.float32, copy=False)
                for index in (0, 3, 4, 6, 7, 8)
            ]
            averages_halo = [(g16 + g18) / 2.0 for g16, g18 in zip(channels_g16, channels_g18)]

            validity_fields = channels_g16 + channels_g18
            halo_valid = np.ones(validity_fields[0].shape, dtype=bool)
            for field in validity_fields:
                halo_valid &= (field > 0) & (field < 1e3) & np.isfinite(field)

            averages_stack_halo = np.stack(averages_halo, axis=0)
            local_mean_halo, local_std_halo = _local_valid_stats(averages_stack_halo, halo_valid)
            local_mean_9x9_halo, local_std_9x9_halo = _local_valid_stats(
                averages_stack_halo, halo_valid, half_window=SPATIAL_HALF_WINDOW_9X9
            )

            # Trim the halo back down to this chunk's native rows.
            trim_top = row_start - halo_start
            trim_bottom = halo_end - row_end
            native_rows = row_end - row_start
            row_slice = slice(trim_top, trim_top + native_rows)

            averages = [field[row_slice] for field in averages_halo]
            block_valid = halo_valid[row_slice]
            local_mean = local_mean_halo[:, row_slice]
            local_std = local_std_halo[:, row_slice]
            local_mean_9x9 = local_mean_9x9_halo[:, row_slice]
            local_std_9x9 = local_std_9x9_halo[:, row_slice]

            # Subsample to pixel_step now that the native-resolution window is applied.
            averages = [field[::pixel_step, ::pixel_step] for field in averages]
            block_valid = block_valid[::pixel_step, ::pixel_step]
            local_mean = local_mean[:, ::pixel_step, ::pixel_step]
            local_std = local_std[:, ::pixel_step, ::pixel_step]
            local_mean_9x9 = local_mean_9x9[:, ::pixel_step, ::pixel_step]
            local_std_9x9 = local_std_9x9[:, ::pixel_step, ::pixel_step]

            differences = [
                averages[3] - averages[1],
                averages[3] - averages[4],
                averages[3] - averages[0],
                averages[3] - averages[5],
            ]

            block_mask = block_valid & np.all(np.isfinite(local_mean), axis=0)

            if not np.any(block_mask):
                continue

            rows, columns = np.nonzero(block_mask)
            global_rows = row_start + rows * pixel_step
            global_columns = columns * pixel_step
            flat_indices = global_rows * width + global_columns
            feature_fields = (
                averages
                + differences
                + [local_std[index] for index in range(local_std.shape[0])]
                + [local_std_9x9[index] for index in range(local_std_9x9.shape[0])]
            )
            feature_block = np.stack(
                [field[block_mask] for field in feature_fields], axis=-1
            ).astype(np.float32, copy=False)

            if chunk_callback is None:
                flat_mask[global_rows, global_columns] = True
                valid_features.append(feature_block)
                valid_positions.append(flat_indices)
            else:
                chunk_callback(feature_block, flat_indices)
    if chunk_callback is not None:
        return None
    if not valid_features:
        raise ValueError(f"No valid pixels found in {input_file}")

    return (
        np.concatenate(valid_features, axis=0),
        np.concatenate(valid_positions, axis=0),
        flat_mask,
        lat,
        lon,
    )
