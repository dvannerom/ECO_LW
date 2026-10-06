with open("config/nominal_budget.yaml") as handle:
    NOMINAL = yaml.safe_load(handle)
NOM_DIR = NOMINAL["output_dir"]
NOM_CODE = [
    "scripts/run_nominal_budget.py", "src/nominal_budget.py",
    "config/nominal_budget.yaml", "config.yaml", ECO_CHANNEL_SCENARIOS_FILE,
    "src/abi_sunny_comparison.py", "src/geometry_sensitivity.py",
    "src/spectral_response.py",
    "scripts/find_nComponents.py", "scripts/average_resolution.py",
    "scripts/preprocess_data_ABI.py", "src/broadband.py",
    "scripts/run_sensitivity_convergence.py", *SENS_CODE,
]
NOM_COMMAND = (
    "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    "python scripts/run_nominal_budget.py "
)
NOM_CACHE = [f"{NOMINAL['cache_root']}/{day}"
             for day in NOMINAL["training_days"] + NOMINAL["validation_days"]]


rule uncertainty_nominal_budget:
    input:
        f"{NOM_DIR}/budget_report.json",
        f"{NOM_DIR}/numerical_budget.csv",
        f"{NOMINAL['figure_dir']}/eco_uncertainty_numerical_budget.png",
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_diagnostics.png",
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_seed_variability.png",
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_diagnostics_criteria.png",
        f"{NOMINAL['diagnostic_figure_dir']}/spatial_processing.png",
        f"{NOMINAL['diagnostic_figure_dir']}/population_and_assignment.png",
        f"{NOMINAL['diagnostic_dir']}/gmm_selection.json",
        f"{NOMINAL['diagnostic_dir']}/scene_support.json"


rule train_nominal_spectral_gmm:
    input:
        caches=NOM_CACHE,
        code=NOM_CODE
    output:
        f"{NOM_DIR}/spectral_gmm.joblib",
        f"{NOM_DIR}/spectral_gmm.joblib.json",
        f"{NOMINAL['diagnostic_dir']}/gmm_selection.json"
    resources:
        mem_mb=2048
    shell:
        NOM_COMMAND + "train"


rule prepare_nominal_sunny:
    input:
        spectra=sorted(str(path) for path in Path(SUNNY_DIR).glob("radiance_lw_*/sunny_lw_*")),
        filters=sorted(str(path) for path in Path(GOES_FILTER_DIR).glob("goes-r_abi_*") if path.is_file()),
        cache=NOM_CACHE[0],
        code=NOM_CODE
    output:
        f"{NOM_DIR}/sunny.npz",
        f"{NOM_DIR}/sunny.npz.json"
    resources:
        mem_mb=1024
    shell:
        NOM_COMMAND + "prepare"


def nominal_abi_sources(wildcards):
    sources = []
    for day in NOMINAL["validation_days"]:
        with open(f"{NOMINAL['cache_root']}/{day}/index.json") as handle:
            index = json.load(handle)
        sources.extend(entry["path"] for entry in index["provenance"]["inputs"].values())
    return sources


rule assess_nominal_abi_assignment:
    input:
        model=f"{NOM_DIR}/spectral_gmm.joblib",
        metadata=f"{NOM_DIR}/spectral_gmm.joblib.json",
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["validation_days"]],
        abi=nominal_abi_sources,
        code=NOM_CODE
    output:
        f"{NOM_DIR}/abi_assignment.json"
    resources:
        mem_mb=2048
    shell:
        NOM_COMMAND + "assignment"


rule diagnose_nominal_gmm:
    input:
        caches=NOM_CACHE,
        coefficients=config["narrowband_to_broadband_coeffs_file"],
        code=NOM_CODE
    output:
        f"{NOMINAL['diagnostic_dir']}/gmm_component_diagnostics.csv",
        f"{NOMINAL['diagnostic_dir']}/gmm_component_diagnostics.csv.json"
    resources:
        mem_mb=2048
    shell:
        NOM_COMMAND + "diagnostics"


rule assess_nominal_spatial:
    input:
        model=f"{NOM_DIR}/spectral_gmm.joblib",
        metadata=f"{NOM_DIR}/spectral_gmm.joblib.json",
        caches=NOM_CACHE,
        abi=nominal_abi_sources,
        coefficients=config["narrowband_to_broadband_coeffs_file"],
        code=NOM_CODE
    output:
        f"{NOM_DIR}/spatial_report.json",
        f"{NOM_DIR}/spatial_residuals.npz",
        f"{NOM_DIR}/abi_spatial_adm.joblib"
    resources:
        mem_mb=2048
    shell:
        NOM_COMMAND + "spatial"


rule calculate_nominal_budget:
    input:
        f"{NOM_DIR}/sunny.npz", f"{NOM_DIR}/sunny.npz.json",
        f"{NOM_DIR}/spectral_gmm.joblib", f"{NOM_DIR}/spectral_gmm.joblib.json",
        f"{NOM_DIR}/abi_assignment.json",
        f"{NOM_DIR}/spatial_report.json",
        code=NOM_CODE
    output:
        f"{NOM_DIR}/budget_report.json",
        f"{NOM_DIR}/residuals.npz",
        f"{NOM_DIR}/cv_models.joblib",
        f"{NOM_DIR}/nominal_models.joblib",
        f"{NOM_DIR}/nominal_models.joblib.json",
        f"{NOMINAL['diagnostic_dir']}/scene_support.json"
    resources:
        mem_mb=2048
    shell:
        NOM_COMMAND + "calculate"


rule plot_nominal_budget:
    input:
        report=f"{NOM_DIR}/budget_report.json",
        selection=f"{NOMINAL['diagnostic_dir']}/gmm_selection.json",
        diagnostics=f"{NOMINAL['diagnostic_dir']}/gmm_component_diagnostics.csv",
        diagnostics_metadata=f"{NOMINAL['diagnostic_dir']}/gmm_component_diagnostics.csv.json",
        plotter="plotting/plot_n_components.py",
        code="plotting/plot_nominal_budget.py"
    output:
        f"{NOM_DIR}/numerical_budget.csv",
        f"{NOMINAL['figure_dir']}/eco_uncertainty_numerical_budget.png",
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_diagnostics.png",
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_seed_variability.png",
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_diagnostics_criteria.png",
        f"{NOMINAL['diagnostic_figure_dir']}/spatial_processing.png",
        f"{NOMINAL['diagnostic_figure_dir']}/population_and_assignment.png"
    resources:
        mem_mb=1024
    shell:
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python plotting/plot_nominal_budget.py "
        "--report {input.report:q}"
