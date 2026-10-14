from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset


def load_data(path, variable_names=None):
    """Load selected NetCDF variables into a NumPy-backed mapping."""
    with xr.open_dataset(path, engine="netcdf4") as dataset:
        names = dataset.variables if variable_names is None else variable_names
        return {name: dataset[name].values for name in names}


def write_dataset(path, variables, dimensions, attrs=None):
    """Write named arrays as a structured NetCDF product."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data_vars = {
        name: (dimensions[name], np.asarray(value))
        for name, value in variables.items()
    }
    dataset = xr.Dataset(data_vars=data_vars, attrs=attrs or {})
    encoding = {
        name: {"zlib": True, "complevel": 1, "shuffle": True}
        for name, (_, value) in data_vars.items()
        if np.asarray(value).ndim > 0
    }
    dataset.to_netcdf(path, engine="netcdf4", format="NETCDF4", encoding=encoding)


def write_preprocessed(path, variables):
    """Write the gridded preprocessing product with explicit dimensions."""
    dimensions = {}
    for name, value in variables.items():
        array = np.asarray(value)
        if name.startswith("planck_"):
            dimensions[name] = ("channel", "planck_parameter")
        elif array.ndim == 0:
            dimensions[name] = ()
        elif array.ndim == 2:
            dimensions[name] = ("y", "x")
        elif array.ndim == 3:
            dimensions[name] = ("y", "x", "channel")
        else:
            raise ValueError(f"Unsupported preprocessing variable shape: {name} {array.shape}")
    write_dataset(path, variables, dimensions, attrs={"product": "ECO ABI preprocessed data"})


class PreprocessedWriter:
	"""Write the preprocessing product while arrays are still short-lived."""

	def __init__(self, path, height, width):
		Path(path).parent.mkdir(parents=True, exist_ok=True)
		self.dataset = Dataset(path, "w", format="NETCDF4")
		self.dataset.setncattr("product", "ECO ABI preprocessed data")
		self.dataset.createDimension("y", height)
		self.dataset.createDimension("x", width)
		self.dataset.createDimension("channel", 9)
		self.dataset.createDimension("planck_parameter", 4)
		self.variables = {}

	def _create(self, name, values, dimensions):
		array = np.asarray(values)
		chunksizes = None
		if dimensions == ("y", "x", "channel"):
			chunksizes = (
				min(self.dataset.dimensions["y"].size, 256),
				min(self.dataset.dimensions["x"].size, 256),
				1,
			)
		variable = self.dataset.createVariable(
			name,
			"f4" if array.dtype.kind == "f" else array.dtype,
			dimensions,
			zlib=True,
			complevel=1,
			shuffle=True,
			chunksizes=chunksizes,
		)
		self.variables[name] = variable
		return variable

	def write_scalar(self, name, value):
		self._create(name, np.asarray(value), ())[:] = value

	def write_2d(self, name, values):
		self._create(name, values, ("y", "x"))[:] = values

	def write_planck(self, name, values):
		self._create(name, values, ("channel", "planck_parameter"))[:] = values

	def write_channel(self, name, channel, values):
		if name not in self.variables:
			self._create(name, values, ("y", "x", "channel"))
		self.variables[name][:, :, channel] = values

	def close(self):
		self.dataset.close()
