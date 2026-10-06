# ECO_LW Project Context

## Scientific context

This repository supports the ECO Earth Explorer candidate mission.

The objective is to estimate TOA longwave fluxes from geostationary satellite radiances.

Input data:
- GOES-16
- GOES-18

Potentially generalizable to other satellites (EarthCare, 3MI)

## Mission facts

Authoritative ECO / GOES-proxy / CERES mission parameters (objectives,
requirements, orbit, instruments, LW channel table, ADM retrieval regime,
uncertainty framework) live in `docs/mission_context.md`, curated from the ECO
Report for Mission Assessment (ESA-EOPSM-ECO-RP-5024) with section citations.

Read that file before any discussion of uncertainty, instrument requirements,
mission design trades, or proxy-to-ECO transfer. Whenever the user supplies new
mission information, append it there, tagged [RfMA] / [user] / [derived] /
[assumed], and promote [assumed] entries once confirmed.

The RfMA PDF and its extracted-text sidecar sit in `docs/` and are git-ignored;
consult them directly when a fact needs checking at source.

## Coding principles

- Python-first solutions
- Use Snakemake-aware syntax highlighting/validation for Snakefile and config.yaml
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

## Chat equation formatting

- Use `$$ ... $$` for display equations in chat, with the delimiters on
  separate lines. This syntax renders correctly in the user's chat interface.
- Do not use `\[ ... \]` for display equations; it appears as raw LaTeX.
