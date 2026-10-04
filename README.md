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
Its workflow-purpose audit records the current coverage and scientific blockers,
including an ADM flux-normalization defect now corrected in code. Existing ABI
ADMs and downstream products, ECO ADM retrievals, and angular/end-to-end scores
must be regenerated before interpreting them with the corrected normalization.

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

# Generate ECO spectral metrics/figures and ABI proxy N2BC inputs.
snakemake -s workflow/Snakefile --configfile config.yaml --cores 1

# Run the existing GOES ABI proxy production chain.
snakemake -s workflow/Snakefile --configfile config.yaml --cores 8 all_goes_proxy
```

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
`figures/uncertainty/eco_spectral_k2_scatter.png` compares N2BC-only and
end-to-end scores across channel scenarios against ObsReq 16, and
`figures/uncertainty/eco_goal_residuals_by_regime.png` shows residuals for the
RfMA goal-band scenario, and
`figures/uncertainty/eco_adm_narrowband_flux_error.png` shows the preceding
angular-retrieval error by channel, and
`figures/uncertainty/abi_adm_proxy_variability.png` shows empirical ABI ADM
shape spread across days by channel and proxy scene. This spread is not an ECO
sigma or a curve-fit covariance; the final JSON carries its ObsReq 12/15/17
traceability and transfer caveats separately. The simulation uses 15 noise-free Sunny
views from 0° to 70°; this is an idealized case, not the mission's full N=1–20
geometry and instrument-noise distribution. The plotted k=2 scatter is
provisional and is not a formal compliance result until its metric convention
and flight SRFs are confirmed.
The two branches are complementary evidence, not a serial sensor substitution:
ECO channel scenarios and coefficients are not applied to GOES ABI band values.
The default workflow does not yet claim a combined per-pixel ECO broadband flux
or total propagated uncertainty.

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
against ObsReq 16. The residual CSV labels the two assessment stages separately.

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores 1 \
	data/uncertainty/eco_spectral_reconstruction.json
```

### 4. `plot_eco_spectral_assessment`

Reads the joined metrics and residuals and creates three diagnostic figures:

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores 1 \
	figures/uncertainty/eco_spectral_k2_scatter.png
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
	figures/scene_id/scene_id_271_res2km_7comp.png
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
python plotting/plot_broadband_flux.py --day 271 --resolution 2 \
	--input data/broadband_flux/broadband_flux_271_res2km.nc

# Plot narrowband-to-broadband cubic-fit diagnostic from the saved coefficients
PYTHONPATH=src python plotting/plot_narrowband_to_broadband.py \
	--coefficients-file data/models/narrowband_to_broadband_coeffs.json \
	--sunny-dir data/Sunny \
	--filter-dir data/goes_channels

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
	--input-csv figures/gmm_diagnostics/gmm_component_diagnostics.csv
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
