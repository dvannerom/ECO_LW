import json
from pathlib import Path
import yaml
import sys
sys.path.insert(0, str(Path("src").resolve()))
from figure_layout import figure_path
from scene_taxonomy import load_scene_taxonomy
from workflow_config import load_settings, settings_paths, resolve_components
from product_paths import preprocessed_path, scene_id_path, narrowband_path, broadband_path, nominal_product_path


# One shared spectral-only taxonomy (config/scene_taxonomy.yaml). Its K is a
# reviewed manual choice; until it is recorded, only K-independent targets
# (e.g. gmm_selection_diagnostics) can run and SCENE_TAXONOMY_GATE fails.
SCENE_TAXONOMY_FILE = config.get("scene_taxonomy_file", "config/scene_taxonomy.yaml")
SCENE_TAXONOMY = load_scene_taxonomy(SCENE_TAXONOMY_FILE)
SCENE_TAXONOMY_SELECTED = SCENE_TAXONOMY["selection"]["n_components"] is not None
# Placeholder K only names paths; every K-dependent rule requires the gate.
SCENE_COMPONENTS = int(SCENE_TAXONOMY["selection"]["n_components"] or 2)
SCENE_TAXONOMY_GATE = "data/scene_classification/scene_taxonomy_gate.json"

DAYS = [int(day) for day in config["days"]]
RESOLUTION = int(config["resolution_km"])
STEP = int(config["preprocess_step"])

SUNNY_DIR = config["sunny_dir"]
GOES_FILTER_DIR = config["goes_filter_dir"]
ECO_CHANNEL_SCENARIOS_FILE = config["eco_channel_scenarios_file"]
ECO_SPECTRAL_CV_FOLDS = config.get("eco_spectral_cv_folds", 5)
ECO_SPECTRAL_METRICS = config["output"]["eco_spectral_metrics"]
ECO_SPECTRAL_RESIDUALS = config["output"]["eco_spectral_residuals"]
ECO_N2BC_MODELS = config["output"]["eco_n2bc_models"]
ECO_N2BC_METRICS = config["output"]["eco_n2bc_metrics"]
ECO_ADM_DIR = config["output"]["eco_adm_dir"]
ECO_ADM_METRICS = config["output"]["eco_adm_metrics"]
ABI_ADM_PROXY_SUMMARY = config["output"]["abi_adm_proxy_summary"]
UNCERTAINTY_REGISTRY = config["output"]["uncertainty_registry"]
UNCERTAINTY_GAP_REPORT = config["output"]["uncertainty_gap_report"]
SENS_SETTINGS = config.get("sensitivity_settings", "config/sensitivity_run.yaml")
with open(SENS_SETTINGS) as sensitivity_handle:
    SENS = yaml.safe_load(sensitivity_handle)
SENS_DIR = SENS["output_dir"]
SENS_REPORT = f"{SENS_DIR}/sensitivity_report.json"
# Keep model-selection diagnostics opt-in. The default production profile
# contains only the reviewed K and the configured K-1/K+1 sensitivity fits.
NOMINAL_SETTINGS_PATH = config.get(
    "nominal_budget_settings", "config/nominal_production.yaml"
)
NOMINAL_SETTINGS_CONFIG = resolve_components(load_settings(NOMINAL_SETTINGS_PATH), SCENE_TAXONOMY, SENS)
PATHS = config.get("paths", {})
PREPROCESSED_ROOT = PATHS.get("preprocessed", "data/preprocessing/abi")
SCENE_ROOT = PATHS.get("scene_id", "data/abi_proxy/products/scene_id")
NARROWBAND_ROOT = PATHS.get("narrowband", "data/abi_proxy/products/narrowband_flux")
BROADBAND_ROOT = PATHS.get("broadband", "data/abi_proxy/products/broadband_flux")
TRAINING_FILES = config.get("training_files", [
    preprocessed_path(day, RESOLUTION, STEP, PREPROCESSED_ROOT)
    for day in NOMINAL_SETTINGS_CONFIG["training_days"]
])
NOMINAL_PRODUCTS = [
    nominal_product_path(
        NOMINAL_SETTINGS_CONFIG.get("product_dir", NOMINAL_SETTINGS_CONFIG["output_dir"] + "/products"),
        NOMINAL_SETTINGS_CONFIG.get("scenario", "rfma_goal_6"), kind
    ) + suffix
    for kind in ("narrowband_flux", "broadband_flux") for suffix in ("", ".json")
]
BUDGET_CSV = config["output"].get(
    "budget_csv", "data/nominal_production/reports/uncertainty_budget.csv"
)

# Shared scene model: {"model": scaler/PCA/GMM Pipeline on 10 spectral features}.
MODEL = NOMINAL_SETTINGS_CONFIG.get(
    "model_path", f"{NOMINAL_SETTINGS_CONFIG['output_dir']}/spectral_gmm.joblib"
)
# Production ADM: pooled held-in full-population library adm_k{K} (K, 6, 1),
# fit with the same taxonomy as MODEL (rules in nominal_budget.smk).
PRODUCTION_ADM_DIR = NOMINAL_SETTINGS_CONFIG.get("adm_dir", f"{NOMINAL_SETTINGS_CONFIG['output_dir']}/component_adms")
PRODUCTION_MAX_VZA_DEG = 70.0
PRODUCTION_CODE = ["src/abi_product.py", "src/broadband.py", "src/sensitivity.py", "src/product_paths.py"]
NOMINAL_BUDGET_REPORT = (
    f"{NOMINAL_SETTINGS_CONFIG['output_dir']}/budget_report.json"
)
NOMINAL_BUDGET_FIGURE = (
    f"{NOMINAL_SETTINGS_CONFIG['figure_dir']}/eco_uncertainty_numerical_budget.png"
)
ECO_SPECTRAL_FIGURE_DIR = str(figure_path(config["output"]["eco_spectral_figures"][0]).parent)
ECO_SPECTRAL_FIGURES = [str(figure_path(path)) for path in config["output"]["eco_spectral_figures"]]
UNCERTAINTY_FIGURES = [
    str(figure_path(
        "figures/nominal_production/diagnostics/uncertainty/"
        "eco_uncertainty_variance_decomposition.png"
    ))
]
with open(ECO_CHANNEL_SCENARIOS_FILE) as scenario_handle:
    ECO_SCENARIOS = list(yaml.safe_load(scenario_handle)["scenarios"])
ECO_ADM_OUTPUTS = [f"{ECO_ADM_DIR}/{scenario}.npz" for scenario in ECO_SCENARIOS]
NARROWBAND_TO_BROADBAND_COEFFS_FILE = config["narrowband_to_broadband_coeffs_file"]
CHANNEL_POWER_LAW_FILE = config["channel_power_law_file"]
ABI_SUNNY_COMPARISON = config.get("abi_sunny_comparison", {})
ABI_SUNNY_COMPARISON_DIR = ABI_SUNNY_COMPARISON.get(
    "output_dir", "data/nominal_production/diagnostics/abi_sunny/comparison"
)
ABI_SUNNY_COMPARISON_OUTPUTS = [
    f"{ABI_SUNNY_COMPARISON_DIR}/{name}"
    for name in (
        "spectral_gmm.joblib", "abi_sample_scenes.csv", "sunny_scenes.csv",
        "sunny_abi_radiances.npz", "summary.json",
    )
]
ABI_SUNNY_POPULATION_DIR = ABI_SUNNY_COMPARISON.get(
    "figure_dir", "figures/abi_sunny_comparison"
)
ABI_SUNNY_POPULATION_DIR = str(figure_path(ABI_SUNNY_POPULATION_DIR))
SENS["figure_dir"] = str(figure_path(SENS["figure_dir"]))
ABI_SUNNY_POPULATION_OUTPUTS = [
    f"{ABI_SUNNY_POPULATION_DIR}/{name}"
    for name in (
        "sunny_scene_population.png", "sunny_scene_population_all_views.png",
        "sunny_scene_consistency.png",
    )
]
ABI_SUNNY_POPULATION_SUMMARY = [
    f"{ABI_SUNNY_COMPARISON_DIR}/{name}"
    for name in ("sunny_file_scene_consistency.csv", "sunny_scene_population.json")
]


def preprocessed(day):
    return preprocessed_path(day, RESOLUTION, STEP, PREPROCESSED_ROOT)


def scene_id(day):
    return scene_id_path(day, RESOLUTION, SCENE_COMPONENTS, SCENE_ROOT)


def narrowband(day, channel):
    return narrowband_path(day, RESOLUTION, channel, NARROWBAND_ROOT)


def broadband(day):
    return broadband_path(day, RESOLUTION, BROADBAND_ROOT)


BROADBAND_OUTPUTS = [broadband(day) for day in DAYS]
SCENE_ID_OUTPUTS = [scene_id(day) for day in DAYS]
NARROWBAND_OUTPUTS = [narrowband(day, channel) for day in DAYS for channel in range(6)]
ABI_PRODUCTION_OUTPUTS = [
    config["output"]["monthly"],
    MODEL,
    CHANNEL_POWER_LAW_FILE,
    NARROWBAND_TO_BROADBAND_COEFFS_FILE,
    *BROADBAND_OUTPUTS,
]
PRODUCTION_ADM = f"{PRODUCTION_ADM_DIR}/adm_k{SCENE_COMPONENTS}.joblib"
DAILY_ADM_OUTPUTS = [
    f"{PRODUCTION_ADM_DIR}/daily/adm_k{SCENE_COMPONENTS}_day{day}.joblib" for day in DAYS
]
SCENE_ID_DIAGNOSTIC_DAY = int(config.get("scene_id_diagnostic_day", DAYS[-1]))
if SCENE_ID_DIAGNOSTIC_DAY not in DAYS:
    raise ValueError("scene_id_diagnostic_day must be included in config['days']")
SCENE_ID_DIAGNOSTIC_FIGURE = (
    f"figures/abi_proxy/diagnostics/scene_id/scene_id_{SCENE_ID_DIAGNOSTIC_DAY}_res{RESOLUTION}km_"
    f"{SCENE_COMPONENTS}comp.png"
)


import shlex
from nominal_provenance import stage_code_paths, stage_signature as nominal_stage_signature

NOM_SETTINGS = NOMINAL_SETTINGS_PATH
NOMINAL = dict(NOMINAL_SETTINGS_CONFIG)

if "nominal_components" in NOMINAL:
    raise ValueError("nominal_components moved to config/scene_taxonomy.yaml")
# Reviewed K from the shared taxonomy; a placeholder only names unrunnable
# K-dependent paths until the gate (SCENE_TAXONOMY_GATE) has been passed.
NOMINAL["nominal_components"] = SCENE_COMPONENTS
if NOMINAL.get("abi_mode", "full") != "full":
    raise ValueError("Only abi_mode: full is supported; the sampled tile-cache mode is retired")
if "gmm_workers" in NOMINAL:
    raise ValueError("Remove gmm_workers: full GMM scheduling now uses --cores automatically")
if "seeds" in NOMINAL:
    raise ValueError("Remove seeds: the component sweep now uses one gmm_seed")
NOM_GMM_SEED = NOMINAL["gmm_seed"]
if (isinstance(NOM_GMM_SEED, bool) or not isinstance(NOM_GMM_SEED, int)
        or not 0 <= NOM_GMM_SEED < 2**32):
    raise ValueError("gmm_seed must be an integer in [0, 2**32)")
NOM_DIR = NOMINAL["output_dir"]
NOM_MODEL_DIR = NOMINAL.get("nominal_model_dir", NOM_DIR)
NOM_CANDIDATE_DIR = NOMINAL.get("candidate_dir", NOM_DIR)
NOM_TRAINING_DIR = NOMINAL.get("training_dir", f"{NOM_DIR}/training_pca")
NOM_MODEL = NOMINAL.get("model_path", f"{NOM_DIR}/spectral_gmm.joblib")
NOM_CODE = [
    "scripts/run_nominal_budget.py", "src/nominal_processing.py", "src/nominal_budget.py",
    *[str(path) for path in settings_paths(NOM_SETTINGS)], "config.yaml", ECO_CHANNEL_SCENARIOS_FILE,
    "src/nominal_streaming.py", "src/streaming_gmm.py", "src/scene_taxonomy.py",
    "src/gmm_scheduler.py",
    "src/gmm_stability.py",
    "src/abi_sunny_comparison.py", "src/geometry_sensitivity.py",
    "src/spectral_response.py",
    "src/spatial_averaging.py", "src/workflow_config.py",
    "src/product_paths.py", "src/nominal_products.py", "src/broadband.py",
    "src/sensitivity.py", "src/scene_features.py", "src/radiometry.py",
    "src/uncertainty.py", "src/eco_spectral_response.py",
]
NOM_FEATURE_CODE = stage_code_paths("features")
NOM_PCA_CODE = stage_code_paths("pca")
NOM_GMM_CODE = stage_code_paths("gmm")
NOM_COMMAND = (
    "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    f"python scripts/run_nominal_budget.py --settings {shlex.quote(str(NOM_SETTINGS))} "
    f"--taxonomy {shlex.quote(str(SCENE_TAXONOMY_FILE))} "
)
NOM_CACHE = [f"{NOMINAL['cache_root']}/{day}"
             for day in NOMINAL["training_days"] + NOMINAL["validation_days"]]
NOM_CANDIDATES = (
    [f"{NOM_CANDIDATE_DIR}/candidate_k{count}_s{NOM_GMM_SEED}.joblib"
     for count in NOMINAL["components"]]
)
NOM_SCORES = (
    [f"{NOM_CANDIDATE_DIR}/candidate_scores/k{count}_s{NOM_GMM_SEED}.json"
     for count in NOMINAL["components"]]
)
NOM_ASSIGNMENT_PARTS = (
    [f"{NOM_DIR}/assignment_parts/day{day}.json"
     for day in NOMINAL["validation_days"]]
)
NOM_SPATIAL_PARTS = (
    [f"{NOM_DIR}/spatial_parts/day{day}_b{block}.json"
     for day in NOMINAL["validation_days"]
     for block in NOMINAL["spatial"]["block_sizes"]]
)
NOM_DIAGNOSTIC_PARTS = (
    [f"{NOMINAL['diagnostic_dir']}/component_parts/k{count}.json"
     for count in NOMINAL["components"]]
)
NOM_CANDIDATE_SUMMARY = f"{NOMINAL['diagnostic_dir']}/gmm_candidates.json"
NOM_ADM_DIR = NOMINAL.get("adm_dir", f"{NOM_DIR}/component_adms")
NOM_NOMINAL_ADM = [
    f"{NOM_ADM_DIR}/adm_k{NOMINAL['nominal_components']}.joblib",
    f"{NOM_ADM_DIR}/adm_k{NOMINAL['nominal_components']}.joblib.json",
]
NOM_ALL_DAYS = NOMINAL["training_days"] + NOMINAL["validation_days"]


wildcard_constraints:
    components=r"\d+",
    seed=str(NOM_GMM_SEED),
    day=r"\d+",
    block=r"\d+"



GMM_SELECTION_FIGURES = [
    "figures/gmm_component_selection/diagnostics/gmm_selection_overview.png",
    "figures/gmm_component_selection/diagnostics/gmm_selection_overview_criteria.png",
    "figures/gmm_component_selection/diagnostics/gmm_selection_stability.png",
]


def nominal_abi_sources(wildcards):
    return [preprocessed(day)
            for day in NOMINAL["validation_days"]]
