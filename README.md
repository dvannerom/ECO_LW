Software to develop ECO tools using proxy data.

The current workflow goes like this:

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
# Edit days, training files, resolution, and model settings first.
snakemake -s workflow/Snakefile --configfile config.yaml -n

# Run independent daily jobs in parallel.
snakemake -s workflow/Snakefile --configfile config.yaml --cores 8
```

### Run an individual step

The following direct script commands match the current `config.yaml` example
(day 245, 2 km resolution, step 1, 5 scene components, and lambda center -106).
Run them from the repository root and adjust the arguments and paths when using
different data. Unlike Snakemake, these commands do not run missing upstream
steps for you, so run the steps in order and make sure their input files exist.

```bash
# Preprocess the training day (and each day to be processed).
PYTHONPATH=src python scripts/preprocess_data_ABI.py \
	--day 245 --step 1 --res_km 2 --lambda_center -106

# Train the scene model using the preprocessed training file.
PYTHONPATH=src python scripts/train_GMM.py \
	--input_file data/preprocessed_files/abi_245_res2km_step1.nc \
	--n_components 5 \
	--n_components_file data/models/selected_n_components.json \
	--use_pca

# Classify the preprocessed day with the trained model.
PYTHONPATH=src python scripts/scene_id_pca.py \
	--input_file data/preprocessed_files/abi_245_res2km_step1.nc \
	--model data/models/gmm_pipeline_merged_1files_res2km_5comp.joblib \
	--lambda_center -106

# Fit ADMs for all channels and scenes for the day.
PYTHONPATH=src python scripts/fit_ADM.py \
	--day 245 --resolution 2 --n-components 5

# Convert radiances to narrowband fluxes for the day.
PYTHONPATH=src python scripts/radiance_to_flux.py \
	--day 245 --resolution 2 --n-components 5 --lambda_center -106

# Fit the channel radiance power-law used by the broadband calibration.
PYTHONPATH=src python scripts/fit_irradiance.py \
	--filter-dir data/goes_channels \
	--output data/models/channel_radiance_power_law.json

# Fit the narrowband-to-broadband coefficients.
PYTHONPATH=src python scripts/compute_temperature_SBDART.py \
	--sunny-dir data/Sunny --filter-dir data/goes_channels \
	--power-law-file data/models/channel_radiance_power_law.json \
	--output data/models/narrowband_to_broadband_coeffs.json

# Convert the day's narrowband fluxes to broadband flux.
PYTHONPATH=src python scripts/narrowband_to_broadband.py \
	--day 245 --resolution 2 --lambda_center -106 \
	--coefficients-file data/models/narrowband_to_broadband_coeffs.json

# Optional: degrade an existing preprocessed file to a coarser resolution
# (e.g. 2 km -> 6 km with block size 3) to study the impact of resolution
# on the rest of the workflow. This writes a new preprocessed file that can
# be used as a drop-in input to the resolution's downstream steps above.
PYTHONPATH=src python scripts/average_resolution.py \
	--input-file data/preprocessed_files/abi_245_res2km_step1.nc \
	--block-size 3

# Aggregate daily broadband flux files (add one --inputs path per day).
PYTHONPATH=src python scripts/monthly_flux.py \
	--inputs data/broadband_flux/broadband_flux_245_res2km.nc \
	--reference-data data/preprocessed_files/abi_245_res2km_step1.nc \
	--output data/monthly/monthly_flux_res2km.nc \
	--resolution 2 --lambda-center -106
```

The model filename includes the number of training files and the selected
component count. Update its path if either changes. For monthly aggregation,
include every configured daily broadband file after `--inputs`; use the
preprocessed file for the first configured day as `--reference-data`.

Snakemake tracks preprocessing, scene classification, ADM fitting, narrowband
conversion, broadband conversion, and monthly aggregation as separate stages.
Completed daily products are reused automatically, and a failed stage can be
resumed without restarting earlier stages. The legacy ADM and radiance scripts
write several products in one invocation; the workflow records per-day
completion markers for those batch operations.

Plotting is kept as an explicit offline step after the products have been written to disk.

## Offline plotting

Use the standalone plotting scripts after the computational stages have finished:

```bash
# Plot an already-generated scene classification output
python plotting/plot_scene_id.py --input data/scene_id/scene_id_271_res2km_5comp.nc

# Plot BT and BTD centroids from an existing GMM and its training files
PYTHONPATH=src python plotting/plot_scene_centroids.py \
	--model data/models/gmm_pipeline_merged_1files_res2km_5comp.joblib \
	--input-file data/preprocessed_files/abi_245_res2km_step1.nc \
	--label-order c14_btd14_08

# Plot PCA variance and loading diagnostics for the saved model
PYTHONPATH=src python plotting/plot_gmm_diagnostics.py \
	--input-file data/preprocessed_files/abi_245_res2km_step1.nc \
	data/preprocessed_files/abi_271_res2km_step1.nc \
	--model data/models/gmm_pipeline_merged_1files_res2km_5comp.joblib

# Plot an ADM fit for a specific day/channel/scene set
python plotting/plot_fit_ADM.py --day 271 --resolution 2 --n-components 5 --channel 0 --scene 0

# Compare the fitted ADM curves of all scenes for one day/channel
PYTHONPATH=src python plotting/plot_ADMs.py --day 271 --resolution 2 --channel 0 --n-components 5

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
	--executor slurm --jobs 20 --latency-wait 60
```

The GMM and ADM parameters are calibration products. Train and inspect those
products before running the production days, then set `model` in `config.yaml`
to the approved model artifact. Monthly aggregation uses streaming sums and
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
