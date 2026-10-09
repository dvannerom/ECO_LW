Framework for demonstrating the full GOES ABI narrowband longwave
radiance-to-broadband-flux workflow and assessing what that demonstration,
together with ECO-specific simulations, establishes about ECO processing
feasibility and uncertainty.

The complete ABI chain is a core deliverable. ECO's expected improvement from
per-pixel multi-view sampling and overlapping channels must be tested explicitly.
Uncertainty estimates must identify their source, RfMA requirement, and any
ABI-to-ECO transfer or rescaling assumptions; missing or non-transferable terms
must remain explicit gaps, not zero uncertainty.

This repository contributes evidence toward a scoped SRL4 case for ECO MO2's
LW radiance-to-flux data flow; it is not a standalone assessment of mission-wide
SRL. See the [SRL4 readiness assessment](docs/srl4_lw_readiness_assessment.md)
for the handbook criteria, evidence currently available, and remaining gaps.
Its dated workflow-purpose audit records scientific blockers, including an ADM
flux-normalization defect now corrected in code. Any products predating that
correction must be regenerated. The newer nominal-budget and sensitivity
reports have producer provenance; their freshness can be checked independently
of older production products.

The default Snakemake target combines two evidence branches: ECO longwave
spectral reconstruction against ObsReq 16 using GERB/Clerbaux Sunny spectra and
configured ECO channel scenarios, plus GOES ABI proxy processing through scene
identification, ADM fitting, and ADM-corrected narrowband BT inputs. The full GOES ABI
narrowband-to-broadband and monthly production chain remains the explicit
`all_goes_proxy` target; its coefficients are not used as ECO performance
estimates.

ECO observations are not available; the assessment therefore uses ABI as
empirical proxy evidence and SBDART/GERB simulations with ECO channel scenarios
to estimate expected ECO performance.

## Current uncertainty status (2026-10-06)

The current source-based assessment is the explicit
`uncertainty_nominal_budget` target, **not** the default `all` target or the
historical sensitivity table. It uses grouped held-out Sunny scenes, an
ABI-trained spectral-only six-component GMM, ECO goal channels, 15 views,
and a 4-100 um reference. See the
[nominal-budget details](#scene-conditioned-nominal-uncertainty-budget).

The saved conditional combined budget has bias +0.6423 W/m2,
scene-dependent SD 1.8484 W/m2, random SD 0.3780 W/m2, and RMSE
1.9930 W/m2. It includes noise-mediated assignment under an assumed ABI-to-ECO
transfer and spatial processing under an independence hypothesis. It is
**not total ECO uncertainty or a requirement-compliance result**: calibration,
flight SRFs, registration, temporal evolution, training/library discrepancy,
and spatial-noise interactions remain unquantified.

Selected small nominal reports and diagnostics are versioned as evidence
snapshots; large caches, residual arrays, fitted models and generated figures
remain local. These snapshots do not replace the input data needed to reproduce
the workflow. Historical sensitivity experiments remain separately documented.

## GOES ABI proxy processing chain

The following seven steps describe the existing GOES ABI proxy production chain,
run with the `all_goes_proxy` target; they are not the default ECO assessment.

1. Read proxy data and generate smaller files with relevant data: preprocess_data_ABI.py
2. Train a GMM clustering algorithm on the data to assign a scene identification label to all pixels: train_GMM.py
3. Apply the GMM model on the data: scene_id_pca.py
4. For each scene, fit a function and determine the ADM parameters: fit_ADM.py
5. Convert radiance to irradiance (flux) using the derived ADMs: radiance_to_flux.py
6. convert narrowband fluxes to broadband flux: narrowband_to_broadband.py
7. Compute monthly mean flux: monthly_flux.py

## Running the workflow

The original scripts remain available for individual experiments. For production
runs, use the declarative workflow in `workflow/Snakefile`:

```bash
# Preview the default mixed ECO/ABI assessment workflow.
snakemake -s workflow/Snakefile --configfile config.yaml -n

# Generate spectral/angular metrics, the evidence registry, and ABI N2BC inputs.
snakemake -s workflow/Snakefile --configfile config.yaml --cores 1

# Run the existing GOES ABI proxy production chain.
snakemake -s workflow/Snakefile --configfile config.yaml --cores 8 all_goes_proxy
```

### Spectral-only ABI / Sunny scene comparison

The optional `abi_sunny_spectral_comparison` target fits a **separate** GMM
on ABI data only, then applies that same scaler/PCA/GMM to every supplied Sunny
view. It does not change the production 22-feature texture classifier, its
selected component count, scene IDs, or ADM products.

Both inputs use six BTs (C08, C11, C12, C14, C15, C16) and four differences
(C14-C11, C14-C15, C14-C08, C14-C16). ABI uses the arithmetic mean of G16/G18
BTs before computing differences. Sunny uses each provided directional view
directly in place of that mean: no VZA-pair emulation, no angular averaging,
and no invented spatial texture.

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores 1 \
    abi_sunny_spectral_comparison

# Smaller independent experiment on one existing ABI file.
PYTHONPATH=src python scripts/compare_abi_sunny_scenes.py \
    --abi-files data/preprocessed_files/abi_245_res2km_step1.nc \
    --points-per-file 10000 --pixel-step 20 --components 7 \
    --output-dir data/abi_sunny_comparison_small
```

Settings live in `abi_sunny_comparison` in `config.yaml`; the default comparison
component count is 7, independently configured rather than selected by the
production texture-GMM diagnostics. PCA retains 98% of standardized spectral
variance and removes exact BT/BTD dependencies. Scene IDs are ordered by the
component's reconstructed standardized C14 mean, cold to warm.

Sunny directional spectra are convolved with the available GOES-R PFM SRFs.
To match operational ABI radiance units, the band integral is divided by
the response integral in wavenumber and converted to
`mW m-2 sr-1 (cm-1)-1`. BT conversion uses `planck_G16` from the first
lexicographically sorted ABI input as the reference calibration, **not** the
approximate N2BC power-law conversion. Each distinct Sunny wavelength grid
passes a 180-330 K blackbody round-trip check (maximum absolute error <=0.1 K
by default); incompatible SRFs/calibration or unresolved channels fail
explicitly. PFM responses plus G16 reference calibration are a sensor-emulation
approximation, not separate flight-response reproductions for both satellites.

The configured output directory contains:

- `spectral_gmm.joblib`: pipeline, scene ordering, feature names, reference
  calibration and ABI density threshold.
- `abi_sample_scenes.csv`: sampled ABI source/flat-pixel IDs, ten features,
  ordered scene labels, probabilities and PCA-space log density.
- `sunny_scenes.csv`: the same classification fields for every spectrum/view,
  retaining shared scene index, clear/cloud regime, and viewing angle.
- `sunny_abi_radiances.npz`: synthetic channel radiance and BT arrays with shape
  `(n_spectra, 18, 6)`, plus source, channel, angle and unit metadata.
- `summary.json`: scene occupancy, feature means/sample SDs and confidence/
  density diagnostics, including Sunny results by angle and clear/cloud regime,
  sampling settings and radiometric assumptions.

ABI sampling scans strided pixels in row chunks and retains a uniform
random-priority sample capped per file. Memory scales with the training sample
and one chunk, not a full ABI disk. Convolution processes one Sunny spectrum at
a time; retained channel data scale as `n_spectra * n_views * n_channels`.
Sampling is linear in strided pixel count; convolution is linear in spectra,
wavelength samples, views and channels. Full-covariance GMM fitting additionally
scales with EM iterations, training rows, components and squared PCA dimension.

The ABI summary is **in-sample** and covers only the retained training subset,
not all ABI pixels or held-out classification accuracy. Equal caps per file
are not a pixel-population weighting. Sunny angles and clear/cloud records
sharing an index are not independent validation samples; retain the shared
index for any later validation split. A Sunny view replacing a dual-view BT
mean is an explicit comparison convention, not angular equivalence.
Posterior probabilities and low PCA-space density flag ambiguity/coverage,
not calibrated correctness or complete out-of-domain detection. This target
compares scene coverage, not flux-retrieval performance or total ECO uncertainty.

#### Sunny file population and angular scene consistency

The comparison target also generates `sunny_scene_population.png`, with one
bin per GMM scene (including empty scenes). Each Sunny **file** contributes
exactly once, using its most frequent scene across all 18 provided VZAs.
Clear/cloud files with a shared index remain separate files.

The additional `sunny_scene_population_all_views.png` counts every view
assignment, also with one bin per scene. A file with 12 views in scene 0 and
6 in scene 1 contributes 12 and 6 to those bins. Counts sum to
`18 * n_files`, rather than `n_files`; these are not independent files.
The aggregate JSON records these as `scene_view_counts` and
`n_file_view_assignments`, preserving the original `scene_file_counts`.

For each file, angular consistency is
`100 * (number of views assigned to the main scene) / 18`.
Count ties are flagged and resolved by the largest mean posterior over all
views among the tied scenes; an exact posterior tie uses the lowest scene ID.
`sunny_file_scene_consistency.csv` records the main scene, its view count and
percentage, all per-scene view counts, distinct-scene count, and tie flag.
`sunny_scene_population.json` summarizes file counts and consistency;
`sunny_scene_consistency.png` visualizes the distribution of these percentages.
Outputs go to the configured `figure_dir`.

Existing classification outputs can be plotted without refitting the GMM:

```bash
python plotting/plot_sunny_scene_population.py \
    --input data/abi_sunny_comparison_smoke/sunny_scenes.csv \
    --output-dir figures/diagnostics/abi_sunny/smoke
```

Every file must contain all 18 unique VZAs; missing/duplicate views and invalid
posteriors fail explicitly rather than silently changing the denominator.
The CSV is streamed and only per-file counts/posterior sums are retained:
time O(files * views * scenes), memory O(files * scenes).
Changing labels across VZA identifies angular sensitivity and possible
scene-ID instability. It does not alone establish bias or incorrect physical
classification: unsupervised GMM components are spectral regimes rather than
ground-truth scene labels, and spectra physically change with viewing angle.

The shared ADM normalization uses twice the cosine-weighted hemispheric integral
with the degree-to-radian conversion, so corrected radiance represents flux/pi.
Regression checks for isotropic and limb-darkened profiles can be run with:

```bash
PYTHONPATH=src:scripts python -m unittest discover -s tests -p test_adm.py -v
```

The default workflow writes ABI scene-ID and ADM-corrected narrowband BT input
NetCDF products for configured days, plus a traceable summary of ABI ADM
between-day variability. Its ECO simulation branch convolves Sunny
directional radiances with ECO SRFs, fits the repository ADM form per simulated
scene/channel and retrieves narrowband fluxes. Independently, it fits N2BC
coefficients using true band-integrated fluxes. It then applies those held-out
coefficients to both true and ADM-retrieved band fluxes to report N2BC-only and
end-to-end OLR errors. Metrics and per-spectrum out-of-fold residuals are written under
`data/uncertainty/`, with four diagnostic figures:
`figures/uncertainty/spectral_angular/eco_spectral_k2_scatter.png` compares spectral-only
N2BC evidence for ObsReq 16 with angular-plus-spectral evidence for ObsReq 15,
and
`figures/uncertainty/spectral_angular/eco_goal_residuals_by_regime.png` shows residuals for the
RfMA goal-band scenario, and
`figures/uncertainty/spectral_angular/eco_adm_narrowband_flux_error.png` shows the preceding
angular-retrieval error by channel, and
`figures/uncertainty/spectral_angular/abi_adm_proxy_variability.png` shows empirical ABI ADM
shape spread across days by channel and proxy scene. This spread is not an ECO
sigma or a curve-fit covariance; the final JSON carries its ObsReq 12/15/17
traceability and transfer caveats separately. The simulation uses 15 noise-free Sunny
views from 0° to 70°; this is an idealized case, not the mission's full N=1–20
geometry and instrument-noise distribution. The plotted k=2 scatter is
provisional and is not a formal compliance result until its metric convention
and flight SRFs are confirmed.
The two branches are complementary evidence, not a serial sensor substitution:
ECO channel scenarios and coefficients are not applied to GOES ABI band values.
The default workflow does not claim a combined per-pixel ECO broadband flux or
total ECO uncertainty.

The default `all` target also writes
`data/uncertainty/eco_uncertainty_registry.json` and
`data/uncertainty/eco_uncertainty_gaps.md`. The registry records code and
configuration hashes, source-artifact hashes, input-tree inventory identities,
scenario freshness, conditional residual statistics, and unquantified terms.
For paired Sunny cross-validation residuals it reports the observed covariance
between spectral-only error and the angular-plus-interaction increment; it does
not assume stage independence or apply an aggregation reduction. Old products
without producer provenance are labelled unverified even if their scenario
names match the current catalog. Input tree identities use path/size/mtime
metadata rather than reading and hashing large source-data contents.

Three additional registry-driven figures summarize evidence coverage, signed
bias/sample SD/RMSE by scenario/model/regime, and the covariance-aware variance
decomposition. They are linked from the gap report and written to the configured
uncertainty figure directory as `eco_uncertainty_evidence_map.png`,
`eco_uncertainty_residual_summary.png`, and
`eco_uncertainty_variance_decomposition.png`.
Hatching identifies missing evidence or a negative covariance contribution;
freshness labels distinguish current, stale and unverified numerical evidence.
Record counts in the evidence map are not independent sources or budget shares.
The variance waterfall uses all regimes pooled, keeps the signed `2*Cov` term,
and excludes bias; the residual summary separately shows bias, sample SD and
RMSE for all/clear/cloud scenes. Increment RMSE is derived using the exact
sample-count correction from its bias and sample SD.

```bash
# Build the registry, report, six figures and unified CSV (also included in all).
snakemake -s workflow/Snakefile --configfile config.yaml --cores 1 \
    uncertainty_assessment
```

Plotting reads only the small registry: cost and input memory scale with the
number of registered terms, not ABI pixel count. If the registry already exists,
the figures can be generated directly without scheduling upstream processing:

```bash
python plotting/plot_uncertainty_registry.py \
    --registry data/uncertainty/eco_uncertainty_registry.json \
    --evidence-map figures/uncertainty/registry/eco_uncertainty_evidence_map.png \
    --residual-summary figures/uncertainty/registry/eco_uncertainty_residual_summary.png \
    --variance-decomposition figures/uncertainty/registry/eco_uncertainty_variance_decomposition.png
```

This direct command preserves the registry's recorded freshness labels; it does
not recheck producer identities against subsequently changed source files.

The same `uncertainty_assessment` target also generates three numerical panels,
embedded in the gap report:

- `eco_uncertainty_numerical_budget.png`: non-double-counted output-space
  ECO assessment with contributor, uncertainty sources/experiment, and
  final broadband-flux impact columns. Proposed impacts remain pending;
  supporting Sunny bias, sample SD and RMSE use pooled held-out scenes.
- `eco_uncertainty_scenario_budgets.png`: detailed all/clear/cloud evidence
  for every registered channel scenario, using identical N2BC settings.
- `eco_uncertainty_channel_diagnostics.png`: per-channel ADM bias/SD/RMSE.
  These percentages refer to true channel flux, not broadband flux, and are
  not additional contributions to the combined broadband row.

The illustrative selection is `rfma_goal_6`, clear/cloud-stratified N2BC,
degree 2. It is fixed, not chosen by minimum held-out error. Optional
`uncertainty_panel` configuration keys `scenario`, `training_mode`, and `degree`
override this selection; direct plotting also accepts `--budget-scenario`,
`--training-mode` and `--degree`.

The main figure and
[uncertainty CSV](<docs/Uncertainty budget and propagation for ECO_LW - Sheet1.csv>)
share the rows defined in
[the experiment specification](config/uncertainty_experiments.yaml).
The workflow exports the CSV alongside the figures; edit the specification,
not the generated CSV. To refresh only the main panel and CSV from an existing
registry, add these arguments to the direct plotting command above:

```bash
    --numerical-budget figures/uncertainty/historical/first_draft/eco_uncertainty_numerical_budget.png \
    --experiment-spec config/uncertainty_experiments.yaml \
    --budget-csv "docs/Uncertainty budget and propagation for ECO_LW - Sheet1.csv"
```

The proposed experiments are not yet executed: RfMA NEdT-derived Gaussian
radiance noise; 2 km processing followed by flux aggregation versus 10 km
radiance aggregation followed by processing; GMM component counts 6/7;
second-choice assignment for an assumed random 10% of valid held-out pixels;
normalized ADM functional-form comparison; and Sunny library reweighting,
physical-screening and robust-fit sensitivities. Noise is injected before
classification and propagated through the complete chain. The noise scenario
uses radiance SD from the channel-integrated Planck derivative at 255 K,
not constant NEdT at every scene temperature. Independent noise is an explicit
initial assumption; realization counts are unset until execution is configured.
The spatial reference is finer-resolution ABI processing on matched 10 km
footprints, not absolute truth. Component-count changes require consistent
scene ADM libraries. A higher-order per-pixel ADM is not identifiable from
two views without additional constraints. Library sensitivity does not establish
real-world representativity or RT accuracy. Existing spectral/angular residuals
are supporting evidence, not estimates of these proposed sensitivity impacts.

Only OLR/LW RfMA values are shown: ObsReq 16 beside spectral reconstruction,
ObsReq 15 beside the combined instantaneous residual, and ObsReq 10/11 in a
separate unquantified aggregation/stability section. They are contextual
references, not statistical thresholds or compliance tests. No requirement
value is assigned to individual channel errors or the angular increment.
Measured spectral/increment covariance is retained; unknown contributions
remain hatched and are not assigned zero. The combined row is neither total
ECO uncertainty nor a guaranteed lower bound.

### Figure organization

Generated figures are grouped by purpose:

```text
figures/
  products/
    broadband_flux/                 Retrieved instantaneous maps
    monthly_flux/                   Retrieved monthly maps
  diagnostics/
    nominal/                        Current budget GMM, assignment and spatial diagnostics
    gmm/production/                 Historical production texture-GMM diagnostics
    abi_sunny/comparison/            ABI-to-Sunny transfer studies
    abi_sunny/smoke/                 Small exploratory transfer run
    scene_id/                       Scene maps, centroids and PCA
    adm/                            ADM fits and profiles
    n2bc/                           Channel and broadband regression fits
    instrument/                     Spectral response plots
    radiance_difference/
    parallax/
    convergence/                    Training/evaluation convergence studies
    extensions/                     Geometry and assignment stress curves
    sensitivity/                    First-draft sensitivity experiments
  uncertainty/
    nominal/                        Current numerical source/error-class budget
    spectral_angular/               Quantified spectral/angular assessments
    registry/                       Evidence map and decomposition summaries
    historical/first_draft/          Superseded numerical and scenario budgets
```

Start with the budget in `figures/uncertainty/nominal/` and its supporting
diagnostics in `figures/diagnostics/nominal/`. Smoke tests and historical
budgets are not alternative current budget estimates.

Existing files were relocated without changing their content. Script defaults
and Snakemake output declarations follow the new layout.
`src/figure_layout.py` redirects legacy paths stored in scientific configs
and saved run reports at plotting time. Those original config strings are
retained for provenance compatibility, avoiding invalidation of large ABI
caches merely because figure destinations changed. Data/model paths are
unchanged; explicit custom output locations remain supported.

### Scene-conditioned nominal uncertainty budget

The canonical source-based uncertainty table is now generated separately from
the historical sensitivity tables:

```bash
snakemake uncertainty_nominal_budget \
    -s workflow/Snakefile --configfile config.yaml --cores 2 \
    --resources mem_mb=4096
```

#### Full-data and fast-test profiles

The command above preserves the historical sampled configuration. Select a
different nominal settings file explicitly; profiles have separate caches,
reports, models and figures, so the smoke run cannot overwrite a full budget.

**Full data: all 25 available days, no ABI fitting/evaluation sampling**

```bash
conda activate tf-gpu
snakemake uncertainty_nominal_budget \
    -s workflow/Snakefile --configfile config.yaml \
    --config nominal_budget_settings=config/nominal_budget_full.yaml \
    --cores 48 --resources mem_mb=98304 nominal_io=4 \
    --rerun-incomplete
```

Use `--dry-run` first to inspect the jobs. The full profile in
[nominal_budget_full.yaml](config/nominal_budget_full.yaml) uses the existing
20-day production training split and held-out days **246/252/259/265/271**.
It checks that these disjoint sets partition all days in the main configuration.
The output table is
`data/uncertainty/nominal_full/numerical_budget.csv`, with its figure at
`figures/uncertainty/nominal_full/eco_uncertainty_numerical_budget.png`.
All 25 native 2 km preprocessed files are workflow dependencies.
Day feature preparation and each component-count/seed/initialization GMM fit
are separate Snakemake jobs; a short candidate-aggregation job selects the
highest-likelihood converged start. EM workers report likelihood progress every
10 iterations and at convergence. A resumed run reuses completed caches and
initialization fits only while their code/configuration/input provenance remains
current.

The full profile also exposes candidate scoring (count/seed), noise assignment
(held-out day), spatial assessment (held-out day/block size), and diagnostics
(component count) as separate jobs, followed by deterministic aggregation.
Assignment keeps all 30 realization streams together within each day, preserving
the original seeds and native pixel order without rereading a day 30 times.
Each count's positive scene ADM is fitted once; the nominal-count library is
shared by spatial assessment and diagnostics.

Performance controls in the full settings are:

- `gmm_workers: 8`: maximum processes per initialization, declared as Snakemake
  `threads`. Snakemake scales this down when fewer cores are available; the CLI
  receives the actual allocation through `--workers`. Each process uses one
  BLAS thread. Work and centered-moment reductions are bounded and merged in
  input order; component occupancy is checked globally, not per worker shard.
- `gmm_chunk_points: 65536`: bounded internal EM chunks, independent of the
  native ABI `chunk_rows`. RAM scales with worker count and chunk size, not the
  full population. Independent starts still use the original random seeds.
- `netcdf_chunk_cache_mb: 128`: per-variable compressed-chunk cache for the four
  radiance and two angle variables. This avoids repeated decompression with
  32-row processing slabs, without changing eligibility or footprint geometry.
  The six caches have up to 768 MiB combined capacity per reader; the workflow
  requests 3072 MiB for these reader jobs.

Full-data scaler/PCA moments are computed once and float64 transformed training
features are persisted under `output_dir/training_pca`. Every EM pass still
uses every training pixel. Disk storage is `8 * N * d` bytes, approximately
8.7 GiB for 291 million pixels and four retained dimensions, in addition to
the raw feature cache. Writes and reads remain chunked/memory-mapped.
Likelihood and labels share one Gaussian evaluation; ADM channel reductions
use weighted `bincount` rather than unbuffered repeated-index updates.

The command above is an **example allocation**, not a request to occupy a whole
shared node: choose cores and memory within your scheduler allocation.
`nominal_io=4` caps concurrent native-NetCDF reader jobs at four; tune this
separately from compute concurrency to avoid filesystem/decompression
contention. EM jobs request `1024 + 384 * threads` MiB. For a one-process
reference run, add `--set-threads fit_nominal_full_initialization=1`.

On a bounded actual-data benchmark (524,280 rows across all 20 training days,
15 components, four retained dimensions), the implemented persistent-process
EM kernel took 0.620 s with one worker, 0.194 s with four, and 0.143 s with
eight after startup. Statistics were identical in that test. Process startup
took roughly 1.5-1.6 s for the parallel first pass, so pools are reused across
iterations. These are kernel measurements, not a promised full-run speedup.

These optimizations do not change the population, physics, covariance model,
noise realization count, convergence tolerance, or manual component choice.
NetCDF caching returns identical values. Parallel and larger-chunk moment
reductions can change floating-point roundoff, convergence iteration, or
near-tied labels/model selection; numerical equivalence is tested, not bitwise
identity of every complete scientific run. Do not edit code/configuration
while a run is active. Upgrading from the previous implementation invalidates
strict provenance: expect caches and fits to rebuild, rather than bypassing
checks or relabelling old outputs as current.

Full mode visits every native grid pixel in row chunks. Eligibility requires
finite positive raw/corrected six-channel radiances, physical Planck-derived
BTs and both corrected VZAs in 0-70 degrees; the inherited corrected-radiance
upper bound of 1000 is retained. It does **not** apply historical texture,
tile acceptance, or complete-10-km-block filters to native classifier inputs.
Consequently its population is not just an expansion of the historical tile
sample: it also removes those historical cache eligibility restrictions.

Every eligible held-in pixel contributes to scaler/PCA moments and every
full-covariance GMM EM iteration. A bounded random set initializes the fit
only; it is not a fitting reservoir. All eligible training pairs enter the
positive-hemisphere, paired-ratio scene-ADM objective. Every eligible held-out
pixel enters population counts and each added-noise transition experiment.
The full held-out density percentile uses a disk-backed in-place partition.
Spatial passes align chunks to each footprint size independently, retain all
complete eligible footprints, and exclude only incomplete disk-grid edges,
not chunk edges. Added-noise streams follow day/realization and native pixel
order, independently of chunk size.

Full-mode component diagnostics retain likelihood/ICL, physical ADM scores,
angular coverage and exact ABI cubic-flux impact using all eligible records.
The reliability panel now uses the same **bootstrap assignment-stability**
metric as the production diagnostic: a bounded, per-training-day sample is
split 80/20, then three bootstrap refits are compared with the sample baseline
on its fixed held-out fifth. The diagnostic refits are separate from the
full-population nominal models; this estimates fitting-sample sensitivity, not
full-population robustness or classification accuracy. The
`stability_max_points`, `stability_repeats` and `stability_seed` settings
control its 250,000-point cap, three refits and reproducibility by default.
The full-mode ADM scorer uses the same positive paired-ratio fit as its
spatial stage, and per-day Planck coefficients throughout. Component choice
remains manual.

**Fast partial smoke test**

```bash
snakemake uncertainty_nominal_budget \
    -s workflow/Snakefile --configfile config.yaml \
    --config nominal_budget_settings=config/nominal_budget_fast.yaml \
    --cores 2 --resources mem_mb=4096
```

[nominal_budget_fast.yaml](config/nominal_budget_fast.yaml) uses two held-in
and two held-out days, four existing sampled tiles/day, 5000-row reservoirs,
six components, seed 73, two noise realizations and identity/10 km spatial
checks. The full Sunny library and five grouped folds remain to exercise the
complete join. Seed 73 is retained because this small fitting pool supports
all six Sunny scenes in every training fold; the small scene populations are
not production representativeness evidence. The target completed in about
32 seconds on the current machine with existing convergence caches.
Missing caches are generated by the convergence preparation rules, which
still prepare their original 192-tile pools, so a cold-cache run costs more.
Its table is `data/uncertainty/nominal_fast/numerical_budget.csv`; it is for
execution checks, **not a usable uncertainty estimate**.

The original [nominal_budget.yaml](config/nominal_budget.yaml) remains available
as the default historical eight-day/192-tile/50,000-row profile.
The script also accepts `--settings` and `--config` for individual stages.

**Compute and storage:** full-mode working arrays scale with
`chunk_rows * grid_width` or `batch_points`, not the number of days/pixels.
GMM sufficient statistics scale with components times squared PCA dimension.
Each EM iteration is approximately O(N K D^2), repeated for every configured
component count, seed and initialization; this is an expensive full-data
assessment, not a 32-second run. The feature cache uses **40 bytes per eligible
native pixel** (ten float32 features). ADM fitting temporarily uses another
**36 bytes per held-in pixel per concurrently running fitting job**; the
spatial and diagnostic jobs have separate scratch files. The exact held-out
density percentile temporarily uses eight bytes per held-out pixel.
Reserve disk space accordingly. Spatial residual storage in full mode contains
`block_sizes`, `count`, `mean`, `m2` sufficient statistics rather than individual
footprint residuals; Sunny paired residual products remain unchanged.
`--resources mem_mb` controls Snakemake concurrency, not a process memory limit.

[nominal_budget.yaml](config/nominal_budget.yaml) specifies instantaneous
4-100 um broadband OLR, RfMA goal channels, fifteen 0-70 degree views, the
original repository modified-log angular basis, and a 10 km **product target**.
Sunny spectra are atmospheric columns. Spatial processing is assessed
separately on matched ABI footprints, not by inventing spatial structure in
Sunny. The ABI proxy does not establish ECO optical-response performance.
No CERES is used and no monthly averaging reduction is assumed.
This target must be requested explicitly; `all` does not generate the nominal
budget. The default registry and its historical numerical panel remain separate
from this report and should not be interpreted as a merged current budget.

The workflow in [nominal_budget.smk](workflow/nominal_budget.smk) writes:

- **Uncertainty calculations:** `data/uncertainty/nominal/budget_report.json`,
  paired `residuals.npz`, `numerical_budget.csv`, and grouped CV models.
- **Canonical numerical-budget figure:**
  `figures/uncertainty/nominal/eco_uncertainty_numerical_budget.png`.
- **Supporting diagnostics:** `data/diagnostics/nominal` and
  `figures/diagnostics/nominal`. Component selection, support, population
  weighting and assignment transitions are not additional error allocations.
- **Application models:** `nominal_models.joblib` fits all supported Sunny
  files; it is separate from the CV models used to estimate errors.
- **Spatial proxy:** `spatial_report.json`, `spatial_residuals.npz`, and
  `abi_spatial_adm.joblib`; diagnostic figure `spatial_processing.png`.

Earlier `uncertainty_sensitivity_assessment`, convergence and extension
outputs are under `figures/diagnostics/sensitivity`, `convergence` and
`extensions`; they are not the new source-based budget. The older table at
`figures/uncertainty/historical/first_draft/eco_uncertainty_numerical_budget.png` remains
the first-draft sensitivity table. Use the **nominal** table above for the
new bias / scene-dependent SD / random SD budget.

The new classifier is a spectral-only ABI GMM, with a manually configured
`nominal_components: 6` rather than inheriting the production texture-GMM's
component count. It uses 50,000
training rows from days 245/247/254/261 and 50,000 validation rows from
246/252/259/271, sampled from the existing 192-tile/day eligible overlap
caches. Candidate counts 2 through 15 inclusive use three seeds and two initializations.
The best held-out initialization at the configured count is used; seed
variability is supplementary. The one-SE
likelihood recommendation is diagnostic only and never overrides the count.
Inspect `figures/diagnostics/nominal/gmm_diagnostics.png` for this texture-free
classifier, then edit `nominal_components` manually. The historical production
`figures/diagnostics/gmm/production` describes a different feature/model configuration.
The production and nominal workflows use the same five-initialization
diagnostic fit, 80/20 diagnostic train/test split, bootstrap-refit ARI
calculation, four-panel plotting function, and diagnostic metric definitions.
Thus the plots and metric meanings match; numerical results are not expected
to match because production uses texture features and a sampled population,
while nominal uses spectral-only features and its profile's diagnostic sample.
Nominal's full-profile ADM/flux scores additionally use all eligible pixels
and the positive paired-ratio ADM fit. The four panels show likelihood/ICL,
bootstrap reliability, physical ADM quality, and angular coverage/exact
production flux impact. The diagnostic CSV is
`data/diagnostics/nominal/gmm_component_diagnostics.csv`; all tested counts
receive the exact stage and a criteria heatmap. Seed-specific held-out
likelihood and occupancy are grouped with bootstrap ARI in `gmm_stability.png`;
the four-panel physical/selection diagnostic remains separate. Diagnostic fits
use the existing five-initialization fitter and three bootstrap refits,
separately from the nominal two-initialization models. The original nine-bin
angular coverage metric is preserved despite the cached population's <=70
degree eligibility.
The same validation days are used for initialization selection and cannot
certify independent classification accuracy.

The cache is a **spatial sample**, not a full-overlap evaluation. For each day,
the preparer constructs a non-overlapping 100x100-native-pixel candidate grid,
shuffles it reproducibly, screens up to 6,000 candidates, and retains 192
eligible tiles. At nominal 2 km spacing, each retained core is approximately
200x200 km. Both satellites must have valid data and viewing angles <=70
degrees. The inherited cache additionally requires valid texture features and
complete 5x5 footprints, even though the nominal GMM uses only spectral features.
The 20-pixel halo supports the original native/coarse texture calculations;
only tile cores enter the nominal assessment.

Tiles bound memory/I/O, preserve neighboring pixels for spatial averaging,
and let all studies reuse identical locations. Within retained tiles the
population counting uses all eligible pixels; GMM fitting uses a 50,000-row
reservoir. The spatial stage processes all complete eligible footprints in
the retained tiles. Chunking alone would not require sampling: a future
full-overlap stage could stream every eligible region with boundary handling.
Current sampling leaves geographic/population representativeness and
tile-selection convergence limitations.

Scene populations are counted across complete eligible native footprints on
the held-out days, not just the classifier fitting sample. They describe the
sampled overlap population, not global climatology. ABI and Sunny share six
BTs and four BTDs, without texture. Sunny is convolved with ABI responses for
classification, but with ECO goal responses for physical retrieval. Its
file label is the majority of fifteen view labels; ties use mean posterior.
Mean G16/G18 ABI BT versus single Sunny-view transfer and angular label
consistency remain explicit limitations.

For every shared-index Sunny fold, each scene/channel ADM is trained with
one shared shape and separate file amplitudes, minimizing equal relative
radiance residuals per file. It enforces positive hemispheric profiles.
Held-out files never contribute to their scene ADM or N2BC fit. Weighted
degree-two N2BC training uses true ECO band flux, not retrieved band flux,
so angular errors are not absorbed into its coefficients. Each scene's total
training weight follows the ABI frequency; within-scene files are equally
weighted. Missing scene mass is reported and results are conditional on
covered support. Weighting does not repair within-scene representativity,
unsupported physics or simulation realism.

On identical held-out files, the spectral residual, angular increment,
fixed-label noise increment and assignment-mediated increment sum exactly
to the directly scored subtotal.
The report retains their signed covariance. Thirty noise realizations yield
noise-averaged file errors: their weighted mean is bias, their weighted
variance is scene-dependent spread, and mean within-file variance is random
spread. A finite-Monte-Carlo correction removes estimated noise in file means
from the scene variance; unresolved negative corrected estimates are flagged
as **below Monte Carlo resolution**, not reported as established zero.
Zero random columns for deterministic spectral/angular experiments do not
include unestimated training/library variability.

Assignment-change probabilities are measured on 50,000 uniformly sampled
eligible **held-out ABI pixels** in sampled mode, or every eligible held-out
pixel in full mode, not on Sunny spectra. The baseline-trained
GMM remains fixed. Observed ABI is treated as truth despite existing instrument
noise. Since 2026-10-07, both modes perturb uncorrected radiances with
independent **mean-preserving lognormal noise**, replacing additive Gaussian
noise that could make dim-channel radiances non-positive. The radiance SD
still uses each satellite's Planck derivative at 255 K times 0.4 K; BT and the
mean-G16/G18 spectral features are recomputed for 30 realizations. Noise is
independent across channels, satellites and pixels; averaging the two satellite
BTs changes the effective feature noise. `abi_assignment.json` contains counts,
scene-conditional probabilities, per-day transition counts, and
`noise_model: mean_preserving_lognormal`.

For clean radiance $L>0$, radiance-space SD $\sigma_L$, and independent
$Z\sim N(0,1)$, the implementation uses:

$$
v=\ln(1+(\sigma_L/L)^2),\qquad
L'=\exp(\ln L+\sqrt{v}Z-v/2).
$$

Thus $E[L']=L$ and $\operatorname{Var}(L')=\sigma_L^2$, with strictly positive
radiance mathematically. Zero SD returns the clean radiance exactly. At high
signal-to-noise ratio this approaches additive Gaussian noise; dim channels
have asymmetric, right-tailed perturbations. This distribution is an explicit
modelling assumption, not an RfMA-specified or instrument-validated noise law.
No clipping, rejection sampling, or pixel exclusion is applied. Floating-point
overflow/underflow and invalid BTs still fail explicitly. Log-space evaluation
avoids squaring a potentially overflowing relative SD. Work is linear in
pixel/channel count per realization, with temporary arrays bounded by the
current chunk in full mode. Existing day/realization RNG streams retain
chunk-size invariance.

The separate ECO/Sunny radiometric-noise experiment remains Gaussian.
Previously reported assignment/budget numbers below used the former Gaussian
ABI scenario; rerun assignment and downstream budget products before treating
them as lognormal results.

Sunny is used only to translate these probabilities into broadband error:
each clean file label samples the ABI conditional destination distribution,
and retrieval uses that destination's grouped training ADM on the same noisy
ECO radiances. The assignment increment is the changed-label prediction minus
the fixed-label prediction, so direct radiometric error is not counted twice.
**This contribution is included under the user-assumed similar ECO response.**
Scene-only transfer ignores within-scene spectral dependence and is independent
of ECO-channel noise; physical joint noise covariance is not established.
Unsupported destinations with nonzero probability fail explicitly, without a
fallback ADM or silently restricted population.

Execution uses the configured six components (seed 73), with initialization adjusted
Rand agreement >0.99. All six ABI scene masses were covered in Sunny, but
effective weighted sample size is only about 343 of 4,620 files, and 13.5%
of weighted Sunny views fall below the ABI validation 1st-percentile density.
One scene accounts for 33.3% of ABI mass but has only 42 Sunny files:
nominal weighting does not eliminate this representativity risk.
The revised conditional joint RMSE is 1.956 W/m2: bias +0.622 W/m2,
scene-dependent SD 1.816 W/m2, random SD 0.378 W/m2. ABI pixel-label changes
are 9.03% under added noise; the assignment-mediated RMSE is 0.314 W/m2.
This is noise sensitivity, not a validated physical misclassification rate.
It supersedes the previous Sunny-noise-based probability and excluded row.

The spatial stage reuses `average_resolution.block_average_chunk` on bounded
cached tile locations, rather than writing full coarsened disks. It evaluates
2, 4, 6, 8, 10, 12, 14, 16, 18 and 20 km using native block sizes 1 through
10; the budget still takes the 10 km point (block size 5). At each resolution
it compares 2 km retrieval averaged to complete footprints against retrieval
after radiance averaging, recomputing BT and scene labels. Identity (block 1)
must close to numerical precision. Both routes use the fixed spectral GMM,
ABI scene ADMs fitted only on training days, and the existing ABI cubic N2BC.
Uncorrected radiance defines classifier BT; corrected radiance/angles define
retrieval. The two satellite broadband estimates are averaged in each route.
Cached eligibility remains conditional on overlap and complete valid blocks;
partial/NaN footprints never enter the comparison.
Coarse blocks align to the full-grid origin. Incomplete blocks crossing a
cached tile edge are excluded, never padded or treated as complete. Thus
footprint counts differ across resolutions; the curve uses the same tile
sample, not exactly identical geographic support at every resolution.

The mean coarse-minus-fine difference is the signed spatial bias and its
population SD is scene-dependent spread. There is no stochastic spatial
perturbation here, so the random SD is zero; spatial/noise interactions are
unquantified. This is an ABI two-view processing proxy, not absolute truth or
an ECO fifteen-view/optical PSF simulation. Its existing broadband coefficients
also have a different spectral reference from the ECO 4-100 um fit.

**Under the user's independence work hypothesis**, the spatial variance is
added to the existing paired Sunny subtotal variance. Signed biases are added,
not combined in quadrature. The table retains the directly evaluated Sunny
subtotal and adds a clearly labelled combined budget. Its RMSE is computed
from independent second moments; no artificial pairing of ABI footprints and
Sunny columns is used. Existing Sunny-source covariance is retained, while
spatial cross-covariance is assumed zero.

The executed 2-to-10 km comparison contains 306,644 matched held-out footprints:
spatial bias +0.0200 W/m2, scene SD 0.3471 W/m2, random SD 0 (deterministic),
RMSE 0.3477 W/m2. The independent combined budget is bias +0.6423 W/m2,
scene SD 1.8484 W/m2, random SD 0.3780 W/m2, RMSE 1.9930 W/m2.
These values remain conditional estimates, not a complete mission total.

Memory scales with the bounded ABI reservoir and small Sunny
file/view/channel arrays plus file x realization residuals, not full disks.
Spatial processing reads one tile at a time and stores one scalar residual
per matched footprint/resolution (including the identity check). Its cost is
linear in evaluated pixels per resolution; diagnostics add bootstrap EM fits
and the existing physical ADM/flux scoring.

### Executable first-draft sensitivity assessment

The opt-in rules in
[uncertainty_sensitivity.smk](workflow/uncertainty_sensitivity.smk) execute the
experiment branches and populate the main budget figure and CSV:

```bash
snakemake uncertainty_sensitivity_assessment \
    -s workflow/Snakefile --configfile config.yaml --cores 2 \
    --resources mem_mb=4096 --config uncertainty_sensitivity=true
```

Activate the flag on the first run so the existing plotting rule depends on
the sensitivity report. Once the report exists, subsequent budget plotting
uses it automatically and rejects stale provenance. A dry run (`--dry-run`)
inspects the DAG without executing processing. Each numerical stage writes
an isolated product and a tracked provenance sidecar; failures stop the DAG.
Production radiance, scene and flux products are never overwritten.
Use `--config uncertainty_sensitivity=false` to plot only the original
conditional evidence rather than attach the sensitivity report.

[sensitivity_run.yaml](config/sensitivity_run.yaml) defines a reproducible
bounded assessment: ABI training days 245/247, evaluation days 246/252,
12 non-overlapping 100x100-pixel tiles per day, 20-pixel texture halos,
20,000 sampled training records, GMMs with 6/7 components, and five assignment,
noise and joint realizations. Tiles are randomly drawn then screened for
complete valid footprints and VZA <=70 degrees; this is a temporal holdout,
not geographic independence or a global climatological sample. The full
4,620-spectrum Sunny library uses five shared-index folds, six goal channels
and the configured idealized 15-view geometry. Change `sensitivity_settings`
in the main config to select another run specification; give each assessment
its own `output_dir`. This draft supports bounded tiles, not full-disk streaming
execution, and supports clear/cloud-stratified ECO N2BC only.

The stages cache ABI native/coarse radiances and production 22-feature
spectral/texture vectors, train reusable GMMs, fit pooled scene/channel ADMs
on training days only, and apply fixed models on held-out inputs. Spatial
comparisons share complete 5x5 equal-area footprints and preserve the two
satellite fluxes separately. Second-choice switching changes
`floor(fraction * eligible_pixels)` distinct pixels per evaluation day;
both posterior component IDs remain in the fitted model's native ordering.
Only component-count and ADM-form changes refit the scene-ADM library.
Saved outputs are 10 km footprint fluxes, not full-resolution cubes.

Sunny is convolved one spectrum at a time and cached once. Noise uses the
SRF-integrated Planck derivative at 255 K and RfMA NEdT, with independent
channel/view Gaussian errors as an explicit assumption. The ECO/Sunny branch
has no classifier; its noise score excludes scene-selection effects.
N2BC robust-fit sensitivity uses converged Huber iteratively reweighted
least squares on each training fold, with the same unfiltered held-out cases.
It does not establish library representativity or physical realism; those
subterms remain blocked until justified weights and independent screening
evidence exist.

Both ADM forms enforce positive hemispheric profiles on a 0.1-degree grid.
The regularized baseline is refitted with a positive constraint when needed;
the quadratic extension uses the additional
`cos(theta)^2 - cos(55 deg)^2` basis and constrained fitting. Metadata records
the affected scene/channel counts. This assessment baseline is therefore not
identical to the earlier unconstrained simulation or production library.
Normalization uses the physical cosine-weighted hemispheric integral.

Results live under `data/uncertainty_runs/first_draft`, including
`sensitivity_report.json`, paired flux products, comparison summaries and
producer metadata. The diagnostic figure is
`figures/diagnostics/sensitivity/broadband_sensitivity.png`.
Table metrics show signed bias, sample SD and RMSE of paired flux changes;
repeated-experiment entries are means of realization metrics, not confidence
intervals. ABI relative changes use the baseline flux; Sunny uses simulated
truth. The last row evaluates a **conditional joint Sunny scenario** (noise,
quadratic ADM and robust N2BC) against truth rather than summing source RMSEs.
It is not total mission uncertainty. ABI sensitivities are not added to it.
The covariance row also reports the joint interaction residual: the joint
flux change minus the sum of the matched noise-only, ADM-only and N2BC-only
changes, using shared noise realizations and identical held-out cases.

All branches reuse trained/prepared inputs. Working memory scales with bounded
tile/sample size and Sunny scene count, not full ABI disk size. Evaluation
work scales approximately with sampled pixels x channels x affected variants;
GMM fitting adds the usual sample/component/feature-dependent EM cost.
The first execution uses roughly 100 MB of assessment products; increasing
tile counts or realizations increases storage and compute accordingly.

### ABI regime diagnostics and convergence study

The independent opt-in target in
[uncertainty_convergence.smk](workflow/uncertainty_convergence.smk) implements
the follow-up study without replacing the first-draft budget or its report:

```bash
snakemake uncertainty_convergence_assessment \
    -s workflow/Snakefile --configfile config.yaml --cores 2 \
    --resources mem_mb=8192
```

[sensitivity_convergence.yaml](config/sensitivity_convergence.yaml) specifies
training days 245/247/254/261, held-out days 246/252/259/271, nested tile
prefixes of 12/48/192 per day, and seeds 42/73/109. The 42-job DAG separates
four controls:

- **Evaluation sampling:** three seeded 192-tile pools, evaluated at each
  nested prefix with the same fixed 192-training-tile, 20,000-point models.
- **Training geographic coverage:** 12/48/192 training tiles per day, keeping
  the GMM sample at 20,000 points and the evaluation cases fixed.
- **GMM training sample count:** nested 20,000/50,000/100,000-point samples
  from identical tiles, keeping initialization and evaluation fixed.
- **GMM initialization:** seeds 42/73/109 on identical training rows and
  evaluation cases, with two EM initializations per fit. This probes
  seed-dependent local-optimum selection, not a misclassification rate.

Each model uses a bounded random-priority reservoir of 100,000 paired native
training records for its pooled scene/channel ADM fits. GMM point ladders use
prefixes of the same priority ordering; no evaluation observations enter
training. Each 7-component model is paired with its separately fitted
6-component alternative. Assignment stress switches 10% of valid native
pixels per tile using reproducible day/tile seeds. This is a single
assignment realization per model, distinct from the first draft's five.

The report under `data/uncertainty_runs/convergence/convergence_report.json`
contains paired spatial-order, ADM-form, component-count and assignment
sensitivities, plus changes in the baseline flux itself as training varies.
Regime summaries retain satellite-specific bias/SD/RMSE, sample counts and
shares of squared residuals by model-local scene, day, tile, view angle,
angle separation, native C14 brightness-temperature heterogeneity, modal-scene
fraction, positivity-boundary membership and normalized ADM conditioning.
Scene centroid features and occupancy counts are saved in model sidecars;
component IDs are not mapped across different models or treated as truth labels.
The quadratic-fit condition number comes from a column-normalized ratio
Jacobian: it diagnoses shape identifiability, not radiometric uncertainty.

Three figures under `figures/diagnostics/convergence` show convergence curves,
regime diagnostics and scene/channel ADM profiles. Profile plots mark fits
touching the imposed positivity boundary. Bootstrap intervals resample days
and then tiles within days, keeping all native footprint aggregates and both
satellites together. These 95% intervals are **exploratory and conditional on
four selected held-out days and eligibility-screened tiles**. Repeated
locations across days and unknown spatial correlation between tiles prevent
interpreting them as globally representative mission confidence intervals.
Nested prefixes are correlated. Interval overlap alone does not establish
convergence; the report records a predeclared pragmatic change threshold of
max(0.10 W/m2, 10% of the previous RMSE), without claiming a formal test.

Preparation reuses the existing bounded tile/production-feature operator,
then stores uncompressed per-day NumPy arrays for memory-mapped tile-wise
training and evaluation. It never copies full ABI disks. Preparation memory
scales with 192 selected tiles (including a temporary compressed cache);
training memory scales with the fixed 100,000-record reservoir; evaluation
works tile-wise and saves only 10 km footprint outputs/diagnostics.
Storage scales with the 16 prepared day/seed pools. Bootstrap computation is
O(records + replicates x tiles), using sufficient statistics instead of
materializing resampled pixel arrays. Failures and stale provenance stop
the study; rerunning the same target resumes its completed stages.

The first follow-up execution completed all 42 jobs in approximately 21
minutes using two workers, with about 9.1 GB of isolated products. At the
largest fixed-baseline evaluation (306,644 matched footprints, two
satellites), paired RMSE was 3.028 W/m2 for spatial processing order,
1.248 W/m2 for ADM form, 0.857 W/m2 for 6-versus-7 components, and
0.280 W/m2 for the assignment stress. Across the three largest evaluation
pools, spatial RMSE ranged from 3.020 to 3.154 W/m2 and ADM RMSE from
1.241 to 1.255 W/m2. These values do not replace the first-draft table:
both the training and held-out population have changed.

Training-point sensitivity scores passed the predeclared pragmatic ladder
threshold, but training-coverage convergence did not: spatial RMSE changed
by 0.437 W/m2 between 48 and 192 training tiles/day. Changing training
coverage from 12 to 192 tiles/day changed the baseline flux itself by
2.050 W/m2 RMSE on identical evaluation cases. GMM initialization seed 73
changed the baseline by 0.587 W/m2 relative to seed 42 (seed 109:
0.039 W/m2). Stable source-sensitivity scores therefore do not by themselves
prove retrieval stability.

No fixed-baseline scene/channel fit touched the positivity boundary;
the largest column-normalized quadratic ratio-Jacobian condition number was
24.85. Baseline components 2 and 4 contributed 68.8% of the spatial squared
residual. Spatial sensitivity was not concentrated in high native C14 BT
heterogeneity: 62.6% of its squared residual came from the <1 K footprint-SD
bin. This does not isolate a physical cause; coarse texture/classification
changes are part of the spatial experiment and should be separated from
nonlinear averaging before interpreting the result as a pure resolution
allocation.

### ECO geometry and assignment-response diagnostics

Run the isolated follow-up target:

```bash
snakemake uncertainty_extended_diagnostics \
    -s workflow/Snakefile --configfile config.yaml --cores 2 \
    --resources mem_mb=4096
```

The rules in [uncertainty_extensions.smk](workflow/uncertainty_extensions.smk)
reuse verified first-draft Sunny caches/grouped held-out N2BC models and the
largest fixed-baseline convergence ABI evaluation. Existing numerical-budget
and convergence reports/figures are preserved. New reports are written to
`data/uncertainty_runs/extensions`, and the figures `eco_geometry.png` and
`assignment_curve.png` to `figures/diagnostics/extensions`.
The settings live in
[sensitivity_extensions.yaml](config/sensitivity_extensions.yaml).

The geometry study fits directional radiance as amplitude plus amplitude
times the regularized/quadratic angular basis. Thus it jointly estimates
amplitude and shape with equal observation weights, without requiring an
observed 55-degree view. Flux is the cosine-weighted hemispheric integral
of that fitted profile, constrained positive over a 0.1-degree grid. Sparse
wide and clustered sets of 2/3/5 views, 8 wide views and the original 15
available views are compared, all at VZA <=70 degrees. Two-view quadratic
cases are explicitly reported as unidentifiable, not silently omitted or
filled from scene priors. Minimum-view fits have zero residual degrees of
freedom; design conditioning and constrained scene/channel counts accompany
the broadband errors.

Every geometry is run noise-free and with five shared full-angle Gaussian
noise realizations using the cached ECO goal-channel NEdT-derived radiance
SD at 255 K. The same noise realization is reused between geometries/forms
for overlapping angles. Reports separate truth error, angular increment from
fixed spectral-only held-out predictions, band errors and changes from the
noise-free dense regularized retrieval. The plotted noisy RMSE is the mean
of five realization RMSEs; bars show their range, not confidence intervals.
These are idealized geometry sensitivities, not the actual ECO orbit, and
provide no single-view fallback, angular weighting or ECO scene classifier.
Known clear/cloud regime labels remain an explicit conditional assumption.

The assignment curve uses fractions 0/1/5/10/20/30/40/50 percent and three
seeds on the same fixed native GMM/ADM and held-out footprints. Within each
tile, one seeded permutation is reused across fractions: switched pixels
at lower fractions are strict subsets of those at higher fractions. Exactly
floor(fraction x eligible native pixels) are assigned their second most
probable component; the remainder retain the most probable component.
Zero stress closes exactly to zero broadband change. Actual switch counts,
satellite-specific scores, day/tile bootstrap intervals and regime summaries
are saved. This is a response to assumed assignment stress, not a measured
misclassification probability. The new nested permutation differs from the
historical 10% selector, so its realization is not bitwise identical.

Assignment processing computes native baseline and second-choice flux once
per tile, then aggregates their selected differences to matched 10 km
footprints for each fraction. Compute scales with pixels x channels for
retrieval plus pixels x fractions for aggregation; memory scales with tile
size and compact footprint residuals, never full-disk cubes. Geometry fits
use small batched least-squares arrays and constrained fits only where needed.

The executed study covers all 4,620 Sunny spectra (2,310 paired groups) and
306,644 matched ABI footprints on the four held-out days. Mean assignment
RMSE across three seeds at 0/1/5/10/20/30/40/50 percent is
0.000/0.062/0.167/0.280/0.497/0.711/0.924/1.137 W/m2.
For the quadratic fit, three clustered views at 45/50/60 degrees yield
0.987 W/m2 noise-free truth RMSE but 7.538 W/m2 mean noisy truth RMSE;
three wide views at 0/35/70 degrees yield 0.962 and 1.242 W/m2.
The clustered geometry's noise-only broadband change is 7.457 W/m2,
versus 0.785 W/m2 for the wide geometry. This is a concrete example of
nominal identifiability without sufficient noise robustness.

More views reduce noise amplification in the tested wide geometry ladder,
but total truth error is not necessarily monotonic: ADM approximation bias,
spectral residuals and their covariance also change with the fit geometry.
The direct amplitude/shape estimator is different from the historical
observed-55-degree-normalized estimator, so even its dense-15 result is
not expected to reproduce the historical angular score. Reports include
noise-only changes, raw/column-normalized design conditioning and the
unconstrained linear band-flux noise gain to distinguish these effects.
The latter is flux SD per unit independent equal radiance SD, not a
complete broadband uncertainty or a constrained-estimator confidence bound.

## Individual default workflow stages

The default DAG has independent ECO N2BC-fit and ADM-retrieval branches. They
join only to score the end-to-end chain, alongside ABI proxy scene-ID,
ADM-corrected N2BC inputs, and an ABI ADM variability summary. The two ADM
methods remain separate evidence. Either ECO branch can run first or in parallel.
Run an output target to execute a stage and its upstream dependencies.

### 1. `fit_eco_n2bc_coefficients`

Fits grouped out-of-fold N2BC models using true ECO-band fluxes integrated from
the Sunny spectra. This is the spectral-only N2BC estimate; it does not consume
the ADM retrieval. The fold-specific models are saved for the later end-to-end
stage.

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores 1 \
	data/uncertainty/eco_n2bc_cv_models.joblib
```

This rule also writes `data/uncertainty/eco_n2bc_stage_metrics.json`.

### 2. `retrieve_eco_adm_flux`

Convolves Sunny directional radiances with each ECO channel scenario, fits the
repository ADM shape independently per simulated scene/channel, and saves
ADM-retrieved narrowband fluxes plus angular-retrieval diagnostics. The current
case uses 15 noise-free views from 0° to 70°; it does not yet span ECO's full
N=1–20 geometry or instrument-noise distribution.

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores 1 \
	data/uncertainty/eco_adm_stage_metrics.json
```

### 3. `evaluate_eco_spectral_reconstruction`

Joins the two independent branches: applies each fold's N2BC model to both true
and ADM-retrieved narrowband fluxes, then reports N2BC-only and end-to-end errors
with spectral-only evidence mapped to ObsReq 16 and angular-plus-spectral
evidence mapped to ObsReq 15. The residual CSV labels the two assessment stages
separately.

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores 1 \
	data/uncertainty/eco_spectral_reconstruction.json
```

### 4. `plot_eco_spectral_assessment`

Reads the joined metrics and residuals and creates three diagnostic figures:

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores 1 \
	figures/uncertainty/spectral_angular/eco_spectral_k2_scatter.png
```

This target also produces the goal-band residual and ADM narrowband-flux figures.

### ABI ADM proxy variability summary

Summarizes between-day spread in the existing ABI fitted ADM shape and angular
profiles by ABI channel and proxy scene. It is empirical evidence linked to
ObsReq 15 and 17; it does not provide curve-fit covariance or directly estimate
ECO ADM uncertainty. ObsReq 12 is listed as not directly assessed because ABI's
two fixed views do not reproduce ECO's multi-angle sequence.

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores 1 \
	data/uncertainty/abi_adm_proxy_summary.json
```

### ECO uncertainty registry and gap report

The registry pairs the two Sunny OLR residual stages by scenario, regression,
shared clear/cloud scene index, and validation fold. Its angular-plus-interaction
increment is the paired end-to-end error minus the spectral-only error; its
covariance with the spectral-only error is retained when propagating sample
spread. It also registers ADM narrowband diagnostics and ABI between-day
scene-average variability without transferring or combining ABI terms as ECO
uncertainty.

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores 1 \
    data/uncertainty/eco_uncertainty_registry.json
```

The companion Markdown report lists evidence freshness and unquantified terms.
This older spectral/angular assessment integrates 2.5-500 um, while ObsReq 8
defines 4-100 um. The newer nominal budget uses the latter domain explicitly;
the two assessments are not interchangeable. The registry's measured covariance
is conditional on the Sunny library, idealized SRFs and 15-view noise-free
geometry. It is not total ECO uncertainty, requirement compliance, or
monthly/seasonal uncertainty.

### ABI proxy scene-ID development

The ABI branch trains the configured GMM model and classifies all configured
days. The current selection is 7 components. Run the scene-ID stage on its own
with:

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores 8 \
	abi_scene_id_development
```

These ABI scene labels support development of the ECO scene-ID algorithm; they
remain proxy evidence, not ECO classifications.

### ABI ADM-corrected narrowband inputs

Fits the ABI-proxy ADMs and writes the corrected ABI-channel brightness
temperatures consumed by the GOES-specific N2BC. These are proxy inputs for
scene/ADM development, not ECO narrowband fluxes or ECO-channel measurements.

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores 8 \
	abi_proxy_adm_corrected_narrowband_inputs
```

The ECO simulation and ABI proxy branches are complementary but remain separate
at the N2BC boundary. ECO regression coefficients are not applied to ABI
brightness temperatures. A per-pixel ECO broadband product and total propagated
uncertainty require compatible ECO observations and an explicit uncertainty-
transfer model.

The full-resolution diagnostic map is optional because rendering a disk-sized
scene grid can require substantial memory. Generate the configured diagnostic-
day map with:

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores 1 \
	figures/diagnostics/scene_id/scene_id_271_res2km_7comp.png
```

## GOES ABI proxy target

The additional GOES ABI narrowband-to-broadband conversion and monthly products
are retained under `all_goes_proxy`. Run that target for the complete GOES ABI
proxy production chain:

```bash
# Preview the proxy workflow.
snakemake -s workflow/Snakefile --configfile config.yaml -n all_goes_proxy

# Run the proxy workflow.
snakemake -s workflow/Snakefile --configfile config.yaml --cores 8 all_goes_proxy
```

This target tracks preprocessing, scene classification, ADM fitting, narrowband
conversion, broadband conversion, and monthly aggregation. Its GOES-derived
coefficients are proxy products, not ECO performance estimates. The plotting
section below contains optional GOES proxy diagnostics.

## Offline plotting

Use these standalone scripts for GOES proxy diagnostics after the proxy products
have been generated:

```bash
# Plot an already-generated scene classification output
python plotting/plot_scene_id.py --input data/scene_id/scene_id_271_res2km_7comp.nc

# Plot BT, BTD, and 5x5/9x9 spatial BT stddev centroids from an existing GMM and its training files
PYTHONPATH=src python plotting/plot_scene_centroids.py \
	--model data/models/gmm_pipeline_merged_20files_res2km_7comp.joblib \
	--input-file data/preprocessed_files/abi_245_res2km_step1.nc \
	--label-order c14_btd14_08

# Plot PCA variance and loading diagnostics for the saved model
PYTHONPATH=src python plotting/plot_gmm_diagnostics.py \
	--input-file data/preprocessed_files/abi_245_res2km_step1.nc \
	data/preprocessed_files/abi_271_res2km_step1.nc \
	--model data/models/gmm_pipeline_merged_20files_res2km_7comp.joblib

# Plot an ADM fit for a specific day/channel/scene set
python plotting/plot_fit_ADM.py --day 271 --resolution 2 --n-components 7 --channel 0 --scene 0

# Compare the fitted ADM curves of all scenes for one day/channel
PYTHONPATH=src python plotting/plot_ADMs.py --day 271 --resolution 2 --channel 0 --n-components 7

# Compare G16-G18 radiance differences before and after ADM correction
python plotting/plot_radiance_difference.py --day 271 --resolution 2 --channel 0

# Plot G16, G18, and G16-G18 broadband flux from a saved NetCDF product
python plotting/plot_broadband_flux.py \
	--input data/broadband_flux/broadband_flux_271_res2km.nc

# Plot GOES narrowband-to-broadband cubic-fit diagnostic from saved coefficients
PYTHONPATH=src python plotting/plot_narrowband_to_broadband.py --sensor goes \
	--coefficients-file data/models/narrowband_to_broadband_coeffs.json \
	--sunny-dir data/Sunny \
	--filter-dir data/goes_channels

# Plot ECO held-out, N2BC-only cubic errors from the saved spectral assessment
python plotting/plot_narrowband_to_broadband.py --sensor eco \
	--residuals-file data/uncertainty/eco_spectral_residuals.csv \
	--scenario rfma_goal_6 --training-mode unstratified

# Plot the channel radiance power-law fit used in the broadband conversion
PYTHONPATH=src python plotting/plot_irradiance_fit.py \
	--model data/models/channel_radiance_power_law.json \
	--filter-dir data/goes_channels

# Plot monthly mean G16 and G18 flux and their difference variability
python plotting/plot_monthly_flux.py --input data/monthly/monthly_flux_res2km.nc

# Bin the G16-G18 flux difference by viewing zenith angle and plot its mean
# and standard deviation (pass several --inputs for a multi-day aggregate)
PYTHONPATH=src python plotting/plot_broadband_flux_lza_binned.py \
	--inputs data/broadband_flux/broadband_flux_271_res2km.nc \
	--reference-data data/preprocessed_files/abi_271_res2km_step1.nc \
	--tag 271
```

These scripts read the saved NetCDF products and model artifacts and create PNG
outputs under the `figures/` tree without rerunning the expensive
data-processing steps.

For the broadband flux map, `--input` is the only product selector; no separate
day or resolution argument is needed. The output PNG name is derived from the
input filename, so renamed NetCDF products are also supported.

The cubic N2BC scatter plot defaults to `--sensor goes` for compatibility.
Both sensor modes plot mean-centered `(reference - prediction) / reference`
errors against predicted broadband flux. GOES uses its saved coefficients on
Sunny scenes; ECO streams degree-3, `n2bc_only` grouped out-of-fold predictions
from the assessment CSV, not ADM-inclusive predictions. These are different
validation designs, so their scatter is not a like-for-like sensor comparison.
ECO responses are defined by `config/eco_channel_scenarios.yaml` during the
assessment, not loaded from `--filter-dir` during plotting. Select another saved
scenario with `--scenario`, or separate clear/cloud fits with
`--training-mode clear_cloud_stratified`. Run the
`data/uncertainty/eco_spectral_reconstruction.json` Snakemake target first if
the CSV is missing or needs regeneration after a channel-definition change.
GOES retains its existing PNG name; ECO adds its scenario and training mode
to the filename under `figures/diagnostics/broadband_flux/`. Mean bias is removed across all
selected scenes, not separately by regime. This is not a total uncertainty or
compliance plot.

## GMM component selection

Use `find_nComponents.py` to compare candidate scene counts before changing
`scene_components` in `config.yaml`:

```bash
PYTHONPATH=src python scripts/find_nComponents.py \
	--input-file data/preprocessed_files/abi_245_res2km_step1.nc \
	--validation-inputs data/preprocessed_files/abi_271_res2km_step1.nc \
	--use-pca --min-components 6 --max-components 16 --max-points 100000 \
	--jobs 11 --load-jobs 25
```

The script writes a CSV and a JSON selection file. Plot the saved CSV separately
with:

```bash
python plotting/plot_n_components.py \
	--input-csv figures/diagnostics/gmm/production/gmm_component_diagnostics.csv
```

This creates a 4-panel diagnostic figure and a criteria heatmap alongside the
CSV. The plotter accepts `--output-plot` to override the main figure path.
Cheap diagnostics run for every candidate; a shortlist
(smallest `--exact-shortlist-size` candidates passing the cheap criteria,
default 5) is then re-evaluated with the exact production ADM fit
(`src/adm_fitting.py`, matching `fit_ADM.py`/`radiance_to_flux.py`) and the
exact cubic narrowband-to-broadband regression (`src/broadband.py`, matching
`narrowband_to_broadband.py`).

`--jobs` evaluates that many `n_components` candidates concurrently in
separate worker processes (default 1, sequential); it is capped automatically
to the number of candidates and to the machine's CPU count, so passing a large
value (e.g. the candidate count) is safe. Each worker caps its own BLAS thread
usage so the workers don't oversubscribe cores.

`--load-jobs` similarly parallelizes the upstream step that reads and
featurizes every `--input-file`/`--validation-inputs` file (also capped to the
file count and CPU count). Loading is otherwise the dominant wall-clock cost
with many days: `build_scene_features` keeps the large `BT_G16_interp`/
`BT_G18_interp` variables disk-backed and reads them one row-chunk at a time
rather than materializing the full ~3.8 GB array per variable up front, which
also keeps concurrent workers' peak memory bounded to their current chunk
rather than the whole file. `--load-jobs` is additionally capped using the
node's currently available RAM (`/proc/meminfo`'s `MemAvailable`, Linux-only)
divided by a conservative per-worker memory budget, printing a message when it
downgrades the requested value; a worker OOM-killed by the kernel breaks the
*entire* process pool (`BrokenProcessPool`), not just its own file, so this cap
exists to prevent that failure mode on machines with many files but limited
RAM per core. Per-file sampling uses an independent RNG substream
(derived once from `--seed`, not one shared advancing generator), so results
are identical regardless of `--load-jobs`/execution order, but the specific
sampled points (and therefore exact metric values) differ from runs made
before `--load-jobs` was introduced, even at the same `--seed`.

Selection criteria:

- ADM residual plateau: relative reduction in held-out ADM radiance-ratio RMSE
	between successive component counts; below 1% is the default plateau.
- Scene occupancy: minimum held-out scene fraction; scenes below 0.5% are
	flagged as sparse.
- Cross-day stability: Jensen-Shannon divergence of scene occupancies, the
	standardized feature-centroid shift on validation days, and the minimum
	adjusted Rand index (`ari_min`) across `--stability-repeats` bootstrap
	refits (default 3).
- Angular coverage: minimum fraction of occupied 10-degree viewing-angle bins
	per scene.
- ADM coverage: worst-scene held-out ADM ratio RMSE stays below a threshold;
	shortlisted candidates are checked again with the exact production ADM
	parameters, and their exact global, worst-channel, and worst-scene RMSEs are
	written to the CSV and plotted as exact-stage markers.
- Exact correction benefit: fractional reduction in the *exact* broadband
	G16-G18 flux spread, evaluated only for shortlisted candidates.

A candidate is marked `all_criteria` only when it is shortlisted and passes
all six criteria. The smallest such candidate is written to
`--selection-output` (default `data/models/selected_n_components.json`), which
`train_GMM.py --n_components_file <path>` reads to override `-n` automatically:

```bash
PYTHONPATH=src python scripts/train_GMM.py --input_file ... \
	--n_components_file data/models/selected_n_components.json --use_pca
```

AIC, BIC, silhouette, ARI mean, the training-set minimum scene fraction, and
raw/corrected brightness-temperature spreads were dropped as redundant or
misleading for this decision (see `--help` for remaining thresholds). ICL is
computed on the training set (matching its standard definition), not the
held-out set.

For a SLURM cluster, add a cluster profile or use Snakemake's executor plugin,
for example:

```bash
snakemake -s workflow/Snakefile --configfile config.yaml \
	--executor slurm --jobs 20 --latency-wait 60 all_goes_proxy
```

Within the GOES proxy chain, the GMM and ADM parameters are calibration
products. The Snakefile derives the model artifact path from the training-file
count, resolution, and selected component count; explicit plotting commands
should use that same artifact. Monthly aggregation uses streaming sums and
Welford statistics, so it does not stack all daily scenes in memory.

Production intermediates use structured NetCDF4 files with named variables,
dimensions, units-ready metadata, and a self-describing layout. The numerical
stages still operate on NumPy arrays after loading a product, so this migration
changes the storage contract without changing the scientific calculations.
Products are written with lightweight NetCDF4 compression; the tradeoff is a
small amount of CPU during I/O in exchange for lower storage use and clearer
metadata.

Every spatial product stores `lat(y, x)` and `lon(y, x)` grids next to its data
variables. The `y` and `x` dimensions identify array rows and columns; the
latitude and longitude grids provide the geographic location for each pixel.
