# ECO_LW Project Context

## Scientific context

This repository supports the ECO Earth Explorer candidate mission.

The objective is to estimate TOA longwave fluxes from geostationary satellite radiances.

Input data:
- GOES-16
- GOES-18

Potentially generalizable to other satellites (EarthCare, 3MI)

## Coding principles

- Python only
- NumPy-first solutions
- Avoid unnecessary copies
- Minimize RAM consumption
- Prefer memory mapping for large files
- Avoid loops when vectorization is possible
- Prioritize scientific correctness over abstraction

## Typical array sizes

Examples:

rad:
(9024, 12475, 9)

Approximate memory:
3.8 GB per variable

Therefore:
- avoid flatten()+copy operations
- avoid temporary arrays
- chunk processing where possible

## Performance requirements

Target machine:
- Linux HPC
- Large RAM nodes
- Multicore CPUs

Preferred:
- vectorization
- Welford statistics
- incremental processing
- memory-mapped arrays

## Coding style

- Explicit variable names
- Minimal class usage unless justified
- Functional approach preferred
- Document units and dimensions
- Include array shapes in docstrings
