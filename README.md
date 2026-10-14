# ECO_LW

ECO_LW develops methods to estimate top-of-atmosphere (TOA) longwave (LW)
fluxes from geostationary satellite observations and assess the uncertainties
in those estimates. It uses GOES-16/18 ABI data as proxy observations and Sunny
simulation spectra for ECO channel studies. Its two main goals are to:

1. Produce daily and monthly GOES ABI flux products as a demonstration of the
   radiance-to-flux processing chain.
2. Assess uncertainty in the processing and in ECO-relevant flux estimates,
   combining proxy observations and simulations where appropriate, and report
   the evidence, assumptions, and unquantified gaps.

GOES results are proxy demonstrations, not ECO mission performance claims.
Some ECO uncertainty results are conditional on simulation and transfer
assumptions; consult the reports before interpreting them.

For more than this quick start, see the [technical documentation index](TECHNICAL_DETAILS.md).
It links to focused guides for the [workflow](docs/technical/workflow.md),
[uncertainty assessment](docs/technical/uncertainty.md),
[scene-taxonomy selection](docs/technical/scene-taxonomy.md), and
[plotting](docs/technical/plotting.md).

## Before running

- Run commands from the repository root.
- Install Snakemake and the Python packages used by the workflow in your
  environment. This repository currently has no dependency lockfile or
  requirements manifest.
- Obtain the input data separately. Sunny spectra are expected under
  `data/Sunny/`, channel response files under `data/goes_channels/`, and ABI
  preprocessed files under `data/preprocessed_files/`. These large data are not
  included in the repository.
- ABI preprocessing currently looks for raw GOES files at paths hardcoded in
  `scripts/preprocess_data_ABI.py` on the project HPC system. If you are not
  using that filesystem, update those paths before running preprocessing.
- Before running `all` or a K-dependent focused target, select and record the
  shared GMM component count as described below.

The workflow configuration is in [config.yaml](config.yaml). Its `days`,
`resolution_km`, and `preprocess_step` settings control the ABI period and
spatial sampling. Workflow products are written under `data/` and `figures/`;
large inputs and generated products are generally not tracked by Git.

## Select the GMM component count

The GOES processing and nominal ECO/Sunny uncertainty analysis use the same
scene-classification GMM, so they use the reviewed component count, K. K is
not selected automatically: the explicit diagnostics target compares
candidate counts, but their criteria do not define one universally best
choice. The default production workflow fits the selected K; the configured
K-1/K+1 candidates are fitted only for the one-at-a-time component-count
sensitivity. Both use one reproducible `gmm_seed` (42) and retain two internal
starts per K.

For a new or changed taxonomy, run the K-independent diagnostics first. They
analyze the configured full ABI population across the full 2-15 candidate
range and can be run before choosing K:

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores N \
    gmm_selection_diagnostics \
    --config nominal_budget_settings=config/nominal_budget_full.yaml
PYTHONPATH=src python scripts/check_scene_taxonomy.py --summary
```

Review the diagnostic table and figures in
`figures/gmm_component_selection/diagnostics/`. Then edit `config/scene_taxonomy.yaml`:
set `selection.n_components` and document `selection.rationale`,
`selection.selected_by`, and `selection.date`; copy the reported diagnostics
SHA-256 into `diagnostics.csv_sha256`. The gate checks that this review is
complete and still matches the diagnostics before allowing K-dependent runs.

New diagnostic tables include full held-out-day likelihood, full-training ICL,
individual bootstrap agreements and assignment ambiguity, paired per-day
flux SD/bias/RMSE changes, and scene/paired-angle support. The overview shows
aggregate likelihood/ICL, bootstrap ARI and scene Jaccard agreement, mean
physical-performance changes, and scene occupancy. Individual-day overlays,
likelihood-gain, ambiguity/entropy and angular-support plots are omitted;
their underlying statistics remain saved. ICL uses the nominal pixel count
without correcting spatial dependence. Refreshing these diagnostics requires evaluating saved candidates
(one training pass and one physical validation pass per K), plus the configured
bootstrap refits, not additional full-population GMM fits when upstream
candidates are current. Older tables retain the legacy overview; missing new
statistics are not inferred from the old curves.

This is a required pre-run and review step when the taxonomy is unset or its
diagnostics have changed. Running `all`, `all_goes_proxy`, or
`uncertainty_assessment` with an unset or stale selection will not choose K for
you; the workflow stops at the taxonomy gate. Once the selection is valid, run
the integrated workflow below. The explicit `gmm_selection_diagnostics`
target retains the full candidate sweep; it is not a dependency of `all`.
The [scene taxonomy guide](docs/technical/scene-taxonomy.md) describes the
review procedure and effects of changing K.

## Run the integrated workflow

GOES flux production and uncertainty assessment are parts of one integrated
Snakemake workflow. The default `all` target builds the configured GOES daily
and monthly products and the associated ECO/Sunny uncertainty budget,
production-aligned sensitivity analyses, and uncertainty evidence reports.
These are connected stages in the same DAG, not separate production and
nominal-uncertainty workflows.
The shared selected scene-classification GMM is stored at
`data/models/spectral_gmm.joblib` (with its provenance sidecar); uncertainty
products and fitting diagnostics remain under `data/uncertainty/` and
`data/diagnostics/`.

Preview the full workflow, then run it with the number of CPU cores
appropriate for your machine:

```bash
snakemake -s workflow/Snakefile --configfile config.yaml -n
snakemake -s workflow/Snakefile --configfile config.yaml --cores 8
```

The full workflow can be computationally and storage intensive. Start with a
dry run, and adjust `--cores` and the configured days to suit your machine.
Snakemake runs only outputs that are missing or out of date. The full target
requires all configured inputs and the reviewed GMM selection above. Its
production and nominal GMM use the selected K=7; K=6 and K=8 are additionally
fitted for the configured component-count sensitivity. It does not fit the
full 2-15 selection sweep. See the [workflow guide](docs/technical/workflow.md)
for target details, or the [technical documentation index](TECHNICAL_DETAILS.md)
for the focused guides.

### Focused targets in the same workflow

You can request a subset of the integrated DAG when you do not need every
output. `all_goes_proxy` requests the GOES daily and monthly product branch.
`uncertainty_assessment` requests the uncertainty registry, gap report,
variance-decomposition diagnostic and unified budget CSV, along with their
upstream dependencies. The nominal budget shares the reviewed scene taxonomy
and model inputs with the wider workflow;
these targets are convenient entry points, not separate scientific pipelines.
The uncertainty assessment evaluates evidence from the ABI chain, Sunny-based
ECO simulations, and production-choice sensitivities. Its ECO budget remains
conditional, not a complete mission uncertainty or a requirement-compliance
claim.

```bash
snakemake -s workflow/Snakefile --configfile config.yaml -n all_goes_proxy
snakemake -s workflow/Snakefile --configfile config.yaml --cores 8 all_goes_proxy

snakemake -s workflow/Snakefile --configfile config.yaml -n uncertainty_assessment
snakemake -s workflow/Snakefile --configfile config.yaml --cores 8 uncertainty_assessment
```

## Main outputs

| Product | Location |
| --- | --- |
| Daily GOES broadband flux NetCDF files | `data/broadband_flux/` |
| Monthly GOES flux NetCDF file | `data/monthly/` |
| ECO/Sunny assessment metrics and uncertainty registry | `data/uncertainty/` |
| ABI proxy diagnostics and product-map figures | `figures/abi_proxy/{diagnostics,products}/` |
| Nominal budget and uncertainty evidence | `figures/nominal_production/{diagnostics,uncertainties}/` |
| Full component-selection review figures | `figures/gmm_component_selection/diagnostics/` |
| Uncertainty budget table (CSV) | `docs/Uncertainty budget and propagation for ECO_LW - Sheet1.csv` |
| Uncertainty evidence registry and gap report | `data/uncertainty/eco_uncertainty_registry.json`, `data/uncertainty/eco_uncertainty_gaps.md` |

ABI and monthly maps are optional offline plots of saved products; `all` does
not request them. Narrowband NetCDF files retain ADM-corrected brightness
temperatures for the narrow-to-broadband conversion and also store
ADM-corrected spectral irradiance for mapping. The irradiance is pi times the
isotropic-equivalent corrected ABI radiance, converted from mW to W; it is
spectral (W m-2 per cm-1), not integrated over each channel response:

```bash
python plotting/plot_narrowband_flux.py \
    --input data/narrowband_flux/narrowband_flux_271_res2km_C0.nc

python plotting/plot_broadband_flux.py \
    --input data/broadband_flux/broadband_flux_271_res2km.nc

python plotting/plot_monthly_flux.py \
    --input data/monthly/monthly_flux_res2km.nc
```

Each narrowband map is written to
`figures/abi_proxy/products/narrowband_flux/`, each daily broadband map to
`figures/abi_proxy/products/broadband_flux/`, and the monthly map to
`figures/abi_proxy/products/monthly_flux/`. Maps sample every fourth grid cell
for display to bound memory; the saved NetCDF product remains at its original
resolution. Run the narrowband command once for each channel and configured
day to create that product’s map. The example day and resolution must match
the products generated from your configuration. Regenerate narrowband products
with the workflow before plotting if they predate the `spectral_irradiance_*`
fields.
ADM-fit and ADM-curve plots default to
`figures/abi_proxy/diagnostics/adm/`. See the [plotting guide](docs/technical/plotting.md)
for the plotter catalogue and the [technical documentation index](TECHNICAL_DETAILS.md)
for all technical guides.
