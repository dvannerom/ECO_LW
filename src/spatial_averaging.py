"""Bounded-array spatial averaging shared by production and assessment."""

import math

import numpy as np


def block_average_chunk(chunk, block_size):
	"""Average (rows, cols[, channel]) into (ceil(rows/b), ceil(cols/b)[, channel])."""
	chunk = np.asarray(chunk)
	rows, cols = chunk.shape[:2]
	pad_rows = math.ceil(rows / block_size) * block_size
	pad_cols = math.ceil(cols / block_size) * block_size
	if pad_rows != rows or pad_cols != cols:
		padded = np.full((pad_rows, pad_cols) + chunk.shape[2:], np.nan, dtype=chunk.dtype)
		padded[:rows, :cols] = chunk
		chunk = padded
	reshaped = chunk.reshape(
		pad_rows // block_size, block_size, pad_cols // block_size, block_size, *chunk.shape[2:]
	)
	reshaped = reshaped.swapaxes(1, 2)
	return np.nanmean(reshaped, axis=(2, 3))
