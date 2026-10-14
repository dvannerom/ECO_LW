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
    --config nominal_budget_settings=config/gmm_component_selection.yaml
```

`all` requests the configured daily and monthly GOES products, nominal
ECO/Sunny held-out flux products and budget, production-aligned sensitivity assessment, ECO
spectral/angular assessment, and uncertainty registry. Focused targets request
their outputs and upstream dependencies; they are not separate implementations.
See the [scene taxonomy guide](scene-taxonomy.md) before running any
K-dependent target if the reviewed taxonomy or its diagnostics have changed.

## Inputs and outputs

Large source inputs are not included in the repository. The configured Sunny
spectra, channel responses, and preprocessed ABI files must be available at the
paths in `config.yaml`. Its raw ABI source locations are specific to the project
HPC filesystem; review them before running preprocessing elsewhere.

Products are written primarily under `data/` and `figures/`. The
[plotting guide](plotting.md) documents optional figures from saved products.
For scientific mission assumptions and requirement values, use the
[mission context](../mission_context.md); for the scoped readiness assessment,
see the [SRL4 evidence assessment](../srl4_lw_readiness_assessment.md).

## Artifact ownership

Directories describe producing stages, not the set of downstream consumers.
An artifact has one canonical location; branches reference it rather than copy it.

```text
data/
  inputs/                     Sunny library and ABI response files
  preprocessing/
    abi/                      Prepared native observations
    features/                 Reusable row-chunked ABI feature caches
    training_pca/             Memory-mapped held-in PCA scores
  scene_classification/
    models/                   Promoted reviewed classifier
    scene_taxonomy_gate.json  Reviewed-K gate
  gmm_component_selection/
    models/                   Candidate fits, starts and candidate scores
    diagnostics/              Full-range review evidence
    work/                     Other full-profile intermediates, when requested
  nominal_production/
    models/                   Nominal and grouped-CV ECO estimators
    products/                 Clean held-out narrowband/broadband simulation fluxes
    diagnostics/              Population support and spectral-stage metrics
    uncertainties/            Nominal budget, sensitivity and geometry outputs
    reports/                  Registry, gap report and unified numerical table
  abi_proxy/
    models/                   ABI ADM library and N2B coefficients
    products/                 Scene labels, narrowband, daily and monthly fluxes
    diagnostics/              ABI proxy evidence
```

The three purpose branches have corresponding figure roots:
`figures/gmm_component_selection/`, `figures/nominal_production/` and
`figures/abi_proxy/`. Data-only preparation/model directories do not need
artificial figure counterparts. Generated CSV/JSON diagnostics belong with data;
PNG visualizations belong with figures.

`nominal_production/products/spectral_assessment/` contains the separate
spectral-scenario assessment's retrievals. The nominal estimator's products
live directly under `products/narrowband_flux/` and `products/broadband_flux/`.
Do not combine these different estimators as though they were one result.

## Workflow modules and configuration

[`Snakefile`](../../workflow/Snakefile) assembles one DAG:

| Module | Responsibility |
| --- | --- |
| [`settings.smk`](../../workflow/settings.smk) | Configuration composition and canonical artifact names |
| [`scene_classification.smk`](../../workflow/scene_classification.smk) | Shared feature preparation, candidate fitting, model promotion and ABI ADM libraries |
| [`gmm_component_selection.smk`](../../workflow/gmm_component_selection.smk) | K-independent review diagnostics |
| [`abi_proxy.smk`](../../workflow/abi_proxy.smk) | ABI preprocessing and flux products |
| [`nominal_budget.smk`](../../workflow/nominal_budget.smk) | Nominal ECO/Sunny flux products and uncertainty decomposition |
| [`eco_assessment.smk`](../../workflow/eco_assessment.smk) | Separate ECO spectral/angular scenario assessment |
| [`uncertainty_sensitivity.smk`](../../workflow/uncertainty_sensitivity.smk), [`uncertainty_extensions.smk`](../../workflow/uncertainty_extensions.smk) | Paired sensitivities and optional geometry diagnostics |
| [`reporting.smk`](../../workflow/reporting.smk) | Evidence synthesis, numerical export and report figures |
| [`exploratory.smk`](../../workflow/exploratory.smk) | Optional sampled ABI/Sunny comparison; not the production classifier |

The production and selection profiles extend
[`processing_common.yaml`](../../config/processing_common.yaml).
`extends` names one file relative to the profile; overrides replace whole
top-level values, without recursive merging. Every inherited file is tracked
for downstream provenance. Stage-specific GMM identities continue to exclude
unrelated downstream settings.

The production profile requests `components: reviewed_and_adjacent`: the reviewed
K plus the enabled K-1/K+1 sensitivities inside the diagnostic range.
The selection profile explicitly requests the full candidate range. Use the
full-profile command above for component-selection review, not the smaller
production candidate set. Both reference the same canonical candidate fits,
feature cache and PCA scores. Do not run the two profiles concurrently: their
sweep manifest and promoted model locations are shared.

The full-population assessment requires native 2 km, step-1 ABI inputs.
Unsupported resolution/step combinations fail explicitly rather than silently
selecting different files. ABI scripts accept explicit input/output paths, and
the DAG passes them from `config.yaml`'s `paths` mapping.

## Existing-checkout migration

Stop workflow writers first. Preview the rename plan, then apply it:

```bash
python scripts/migrate_artifact_layout.py
python scripts/migrate_artifact_layout.py --apply
```

Migration uses same-filesystem renames, not payload copies. All destination
conflicts are checked before moving anything; no existing destination is
overwritten. Duplicate full-profile artifacts are preserved under `legacy/`.
The move manifest is written under `logs/`. Historical evidence and its original
provenance are retained, including the previously tracked budget spreadsheet.

**Relocation does not certify freshness.** Original metadata is not rewritten
to look current. Paths, configuration names and implementation hashes have
changed, so existing derived artifacts may need rebuilding; inspect a dry run
before allocating a production job. The reviewed diagnostic CSV bytes and its
pinned SHA-256 are unchanged by a rename. Recomputed diagnostic bytes still
require the normal human review.

Archived sampled classifiers and texture diagnostics are under `legacy/`.
`scripts/fit_ADM.py` and its plotters remain optional older-format diagnostics;
the production branch uses the pooled scene ADM libraries instead.

## Storage and compute notes

The full-resolution ABI population can be very large. The nominal analysis
processes bounded row chunks, while GMM sufficient statistics and model state
scale with the number of components and squared feature dimension. Set
`--cores` and Snakemake memory resources for the target machine, and avoid
running duplicate full-population jobs concurrently. Product maps are separate
offline plotting steps and subsample grid cells for rendering; the saved
NetCDF products are not downsampled.
