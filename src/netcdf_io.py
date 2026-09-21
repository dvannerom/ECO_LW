from pathlib import Path

import numpy as np
import xarray as xr


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
