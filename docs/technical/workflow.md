# Workflow guide

The [README](../../README.md) is the quick-start guide and records the current
workflow entry points. The main Snakemake file is
[`workflow/Snakefile`](../../workflow/Snakefile).

## Main targets

Run from the repository root. Preview a target with `-n` before starting an
expensive run.

```bash
# Integrated ABI production and ECO/Sunny assessment
snakemake -s workflow/Snakefile --configfile config.yaml -n
snakemake -s workflow/Snakefile --configfile config.yaml --cores 8

# Focused branches in the same DAG
snakemake -s workflow/Snakefile --configfile config.yaml --cores 8 all_goes_proxy
snakemake -s workflow/Snakefile --configfile config.yaml --cores 8 uncertainty_assessment

# K-independent component-selection diagnostics
snakemake -s workflow/Snakefile --configfile config.yaml --cores 8 \
    gmm_selection_diagnostics \
    --config nominal_budget_settings=config/nominal_budget_full.yaml
```

`all` requests the configured daily and monthly GOES products, nominal
ECO/Sunny budget, production-aligned sensitivity assessment, ECO
spectral/angular assessment, and uncertainty registry. Focused targets request
their outputs and upstream dependencies; they are not separate implementations.
See the [scene taxonomy guide](scene-taxonomy.md) before running any
K-dependent target if the reviewed taxonomy or its diagnostics have changed.

## Inputs and outputs

Large source inputs are not included in the repository. The configured Sunny
spectra, channel responses, and preprocessed ABI files must be available at the
paths in `config.yaml`. ABI preprocessing also uses source paths specific to
the project HPC filesystem; review them before running preprocessing elsewhere.

Products are written primarily under `data/` and `figures/`. The
[plotting guide](plotting.md) documents optional figures from saved products.
For scientific mission assumptions and requirement values, use the
[mission context](../mission_context.md); for the scoped readiness assessment,
see the [SRL4 evidence assessment](../srl4_lw_readiness_assessment.md).

## Storage and compute notes

The full-resolution ABI population can be very large. The nominal analysis
processes bounded row chunks, while GMM sufficient statistics and model state
scale with the number of components and squared feature dimension. Set
`--cores` and Snakemake memory resources for the target machine, and avoid
running duplicate full-population jobs concurrently. Product maps are separate
offline plotting steps and subsample grid cells for rendering; the saved
NetCDF products are not downsampled.
