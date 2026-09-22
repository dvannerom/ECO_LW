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

Snakemake tracks preprocessing, scene classification, ADM fitting, narrowband
conversion, broadband conversion, and monthly aggregation as separate stages.
Completed daily products are reused automatically, and a failed stage can be
resumed without restarting earlier stages. The legacy ADM and radiance scripts
write several products in one invocation; the workflow records per-day
completion markers for those batch operations.

For the production workflow, plotting is explicitly disabled in the compute steps:

```bash
python scene_id_pca.py --input_file data/preprocessed_files/abi_271_res2km_step1.nc \
  --model data/models/gmm_pipeline_merged_res2km_10comp.joblib --lambda_center -106 --no-plot

python fit_ADM.py --day 271 --resolution 2 --no-plot
```

This keeps the computational pipeline lighter and lets the figures be generated
later, offline, from the saved NetCDF products.

## Offline plotting

Use the standalone plotting scripts after the computational stages have finished:

```bash
# Plot an already-generated scene classification output
python plot_scene_id.py --input data/scene_id/scene_id_271_res2km_10comp.nc

# Plot an ADM fit for a specific day/channel/scene pair
python plot_fit_ADM.py --day 271 --resolution 2 --channel 0 --scene 0

# Compare G16-G18 radiance differences before and after ADM correction
python plot_radiance_difference.py --day 271 --resolution 2 --channel 0

# Plot G16, G18, and G16-G18 broadband flux
python plot_broadband_flux.py --day 271 --resolution 2

# Plot monthly mean G16 and G18 flux and their difference variability
python plot_monthly_flux.py --input data/monthly/monthly_flux_res2km.nc
```

These scripts read the saved NetCDF products and create PNG outputs in the
`figures/` tree without rerunning the expensive data-processing steps.

## GMM component selection

Use `find_nComponents.py` to compare candidate scene counts before changing
`scene_components` in `config.yaml`:

```bash
PYTHONPATH=src python scripts/find_nComponents.py \
	--input-file data/preprocessed_files/abi_245_res2km_step1.nc \
	--validation-inputs data/preprocessed_files/abi_271_res2km_step1.nc \
	--use-pca --min-components 6 --max-components 16 --max-points 100000
```

The script writes a CSV, a 4-panel diagnostic figure, a criteria heatmap, and a
JSON selection file. Cheap diagnostics run for every candidate; a shortlist
(smallest `--exact-shortlist-size` candidates passing the cheap criteria,
default 5) is then re-evaluated with the exact production ADM fit
(`src/adm_fitting.py`, matching `fit_ADM.py`/`radiance_to_flux.py`) and the
exact cubic narrowband-to-broadband regression (`src/broadband.py`, matching
`narrowband_to_broadband.py`).

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
- ADM coverage: worst-scene held-out ADM ratio RMSE stays below a threshold.
- Exact correction benefit: fractional reduction in the *exact* broadband
	G16-G18 flux spread, evaluated only for shortlisted candidates.

A candidate is marked `all_criteria` only when it is shortlisted and passes
all six criteria. The smallest such candidate is written to
`--selection-output` (default `data/models/selected_n_components.json`), which
`train_GMM.py --n-components-file <path>` reads to override `-n` automatically:

```bash
PYTHONPATH=src python scripts/train_GMM.py --input_file ... \
	--n-components-file data/models/selected_n_components.json --use_pca
```

AIC, BIC, silhouette, ARI mean, the training-set minimum scene fraction, and
raw/corrected brightness-temperature spreads were dropped as redundant or
misleading for this decision (see `--help` for remaining thresholds). ICL is
computed on the training set (matching its standard definition), not the
held-out set.

A candidate is marked `all_criteria` only when all four thresholds pass. AIC,
BIC, held-out likelihood, ICL, posterior entropy, and the raw/corrected BT and
flux spreads are also retained in the CSV for inspection. Candidate outputs do
not overwrite production models, labels, ADMs, or flux products.

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
