"""Canonical scene-classification feature construction."""

import numpy as np

from netcdf_io import load_data


def build_scene_features(input_file, chunk_rows=2048, pixel_step=1):
    """Build valid-pixel scene features using the production feature schema.

    Returns
    -------
    features : numpy.ndarray
        Valid pixel features with shape ``(n_valid, 10)``.
    valid_flat : numpy.ndarray
        Flattened indices of valid pixels in the source grid.
    flat_mask : numpy.ndarray
        Boolean validity mask with the source grid shape.
    lat, lon : numpy.ndarray
        Source latitude and longitude grids.
    """
    if pixel_step < 1:
        raise ValueError("pixel_step must be at least 1")

    dataset = load_data(
        input_file,
        variable_names=(
            "lat_interp_grid",
            "lon_interp_grid",
            "BT_G16_interp",
            "BT_G18_interp",
        ),
    )
    lat = dataset["lat_interp_grid"]
    lon = dataset["lon_interp_grid"]
    height, width = lat.shape
    flat_mask = np.zeros((height, width), dtype=bool)
    valid_features = []
    valid_positions = []

    for row_start in range(0, height, chunk_rows * pixel_step):
        row_end = min(row_start + chunk_rows * pixel_step, height)
        bt_g16 = dataset["BT_G16_interp"][row_start:row_end:pixel_step, ::pixel_step]
        bt_g18 = dataset["BT_G18_interp"][row_start:row_end:pixel_step, ::pixel_step]

        channels_g16 = [
            bt_g16[:, :, index].astype(np.float32, copy=False)
            for index in (0, 3, 4, 6, 7, 8)
        ]
        channels_g18 = [
            bt_g18[:, :, index].astype(np.float32, copy=False)
            for index in (0, 3, 4, 6, 7, 8)
        ]
        averages = [(g16 + g18) / 2.0 for g16, g18 in zip(channels_g16, channels_g18)]
        differences = [
            averages[3] - averages[1],
            averages[3] - averages[4],
            averages[3] - averages[0],
            averages[3] - averages[5],
        ]

        validity_fields = channels_g16 + channels_g18
        block_mask = np.ones(validity_fields[0].shape, dtype=bool)
        for field in validity_fields:
            block_mask &= (field > 0) & (field < 1e3) & np.isfinite(field)

        if not np.any(block_mask):
            continue

        rows, columns = np.nonzero(block_mask)
        global_rows = row_start + rows * pixel_step
        global_columns = columns * pixel_step
        flat_indices = global_rows * width + global_columns
        feature_fields = averages + differences
        feature_block = np.stack(
            [field[block_mask] for field in feature_fields], axis=-1
        ).astype(np.float32, copy=False)

        flat_mask[global_rows, global_columns] = True
        valid_features.append(feature_block)
        valid_positions.append(flat_indices)

    if not valid_features:
        raise ValueError(f"No valid pixels found in {input_file}")

    return (
        np.concatenate(valid_features, axis=0),
        np.concatenate(valid_positions, axis=0),
        flat_mask,
        lat,
        lon,
    )
