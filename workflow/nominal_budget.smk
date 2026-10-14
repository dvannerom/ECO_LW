import shlex
from nominal_provenance import stage_code_paths, stage_signature as nominal_stage_signature

NOM_SETTINGS = NOMINAL_SETTINGS_PATH
with open(NOM_SETTINGS) as handle:
    NOMINAL = yaml.safe_load(handle)
if Path(NOM_SETTINGS).resolve() == Path("config/nominal_budget_production.yaml").resolve():
    with open(SCENE_TAXONOMY["diagnostics"]["settings"]) as handle:
        DIAGNOSTIC_COMPONENTS = set(yaml.safe_load(handle)["components"])
    active_variants = set(SENS["abi_variants"]) | set(SENS["sunny_variants"])
    expected_components = {SCENE_COMPONENTS}
    for name, shift in (("components_minus_1", -1), ("components_plus_1", 1)):
        count = SCENE_COMPONENTS + shift
        if name in active_variants and count in DIAGNOSTIC_COMPONENTS:
            expected_components.add(count)
    if set(NOMINAL["components"]) != expected_components:
        raise ValueError(
            "nominal_budget_production.yaml components must match the reviewed "
            f"K and configured adjacent sensitivities: {sorted(expected_components)}"
        )
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
NOM_MODEL = NOMINAL.get("model_path", f"{NOM_DIR}/spectral_gmm.joblib")
NOM_CODE = [
    "scripts/run_nominal_budget.py", "src/nominal_budget.py",
    NOM_SETTINGS, "config.yaml", ECO_CHANNEL_SCENARIOS_FILE,
    "src/nominal_streaming.py", "src/streaming_gmm.py", "src/scene_taxonomy.py",
    "src/gmm_scheduler.py",
    "src/gmm_stability.py",
    "src/abi_sunny_comparison.py", "src/geometry_sensitivity.py",
    "src/spectral_response.py",
    "scripts/find_nComponents.py", "scripts/average_resolution.py",
    "scripts/preprocess_data_ABI.py", "src/broadband.py",
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
    [f"{NOM_DIR}/candidate_k{count}_s{NOM_GMM_SEED}.joblib"
     for count in NOMINAL["components"]]
)
NOM_SCORES = (
    [f"{NOM_DIR}/candidate_scores/k{count}_s{NOM_GMM_SEED}.json"
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
NOM_ADM_DIR = f"{NOM_DIR}/component_adms"
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


rule prepare_nominal_full_abi:
    input:
        data=lambda wc: f"data/preprocessed_files/abi_{wc.day}_res2km_step1.nc",
        code=NOM_FEATURE_CODE
    output:
        directory(f"{NOMINAL['cache_root']}/{{day}}")
    params:
        stage_signature=lambda wc: nominal_stage_signature(
            "features", NOMINAL, {"day": int(wc.day)}
        )
    resources:
        mem_mb=3072,
        nominal_io=1
    shell:
        NOM_COMMAND + "cache --day {wildcards.day}"

rule prepare_nominal_full_training:
    input:
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["training_days"]],
        code=NOM_PCA_CODE
    output:
        directory(f"{NOM_DIR}/training_pca")
    params:
        stage_signature=nominal_stage_signature("pca", NOMINAL)
    resources:
        mem_mb=1024
    shell:
        NOM_COMMAND + "preprocessing"

rule fit_nominal_full_sweep:
    input:
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["training_days"]],
        preprocessing=f"{NOM_DIR}/training_pca",
        code=NOM_GMM_CODE
    output:
        f"{NOM_DIR}/initialization_sweep.json"
    params:
        # Queue membership affects the manifest but not already fitted starts.
        stage_signature=nominal_stage_signature("sweep", NOMINAL),
        requested_jobs_signature=nominal_stage_signature(
            "sweep", NOMINAL,
            {
                "components": NOMINAL["components"],
                "seed": NOM_GMM_SEED,
                "initializations": list(range(NOMINAL["gmm_initializations"])),
            },
        )
    threads:
        workflow.cores
    resources:
        mem_mb=lambda wildcards, threads: 1024 + 384 * threads
    shell:
        NOM_COMMAND + "candidate-sweep --workers {threads}"

rule fit_nominal_full_candidate:
    input:
        sweep=ancient(f"{NOM_DIR}/initialization_sweep.json"),
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["training_days"]],
        preprocessing=f"{NOM_DIR}/training_pca",
        code=NOM_GMM_CODE
    output:
        f"{NOM_DIR}/candidate_k{{components}}_s{{seed}}.joblib",
        f"{NOM_DIR}/candidate_k{{components}}_s{{seed}}.joblib.json"
    params:
        stage_signature=lambda wc: nominal_stage_signature(
            "candidate", NOMINAL,
            {
                "components": int(wc.components),
                "seed": int(wc.seed),
                "initializations": list(range(NOMINAL["gmm_initializations"])),
            },
        )
    resources:
        mem_mb=512
    shell:
        NOM_COMMAND + "candidate-combine --components {wildcards.components} --seed {wildcards.seed}"

rule score_nominal_full_candidate:
    input:
        model=f"{NOM_DIR}/candidate_k{{components}}_s{{seed}}.joblib",
        metadata=f"{NOM_DIR}/candidate_k{{components}}_s{{seed}}.joblib.json",
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["validation_days"]],
        code=NOM_CODE
    output:
        f"{NOM_DIR}/candidate_scores/k{{components}}_s{{seed}}.json"
    resources:
        mem_mb=1024
    shell:
        NOM_COMMAND + "score --components {wildcards.components} --seed {wildcards.seed}"

rule summarize_nominal_full_candidates:
    input:
        candidates=NOM_CANDIDATES,
        candidate_metadata=[path + ".json" for path in NOM_CANDIDATES],
        scores=NOM_SCORES,
        code=NOM_CODE
    output:
        NOM_CANDIDATE_SUMMARY
    resources:
        mem_mb=512
    shell:
        NOM_COMMAND + "candidates"

rule fit_nominal_full_adm:
    input:
        selection=NOM_CANDIDATE_SUMMARY,
        candidates=lambda wc: f"{NOM_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib",
        metadata=lambda wc: f"{NOM_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib.json",
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["training_days"]],
        abi=[f"data/preprocessed_files/abi_{day}_res2km_step1.nc"
             for day in NOMINAL["training_days"]],
        code=NOM_CODE
    output:
        f"{NOM_DIR}/component_adms/adm_k{{components}}.joblib",
        f"{NOM_DIR}/component_adms/adm_k{{components}}.joblib.json"
    resources:
        mem_mb=3072,
        nominal_io=1
    shell:
        NOM_COMMAND + "adm --components {wildcards.components}"

# Model-choice variant for the sensitivity study: quadratic angular form,
# same held-in days, scene taxonomy and paired-ratio objective.
rule fit_nominal_full_adm_quadratic:
    input:
        selection=NOM_CANDIDATE_SUMMARY,
        candidates=lambda wc: f"{NOM_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib",
        metadata=lambda wc: f"{NOM_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib.json",
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["training_days"]],
        abi=[f"data/preprocessed_files/abi_{day}_res2km_step1.nc"
             for day in NOMINAL["training_days"]],
        code=NOM_CODE
    output:
        f"{NOM_ADM_DIR}/adm_k{{components}}_quadratic.joblib",
        f"{NOM_ADM_DIR}/adm_k{{components}}_quadratic.joblib.json"
    resources:
        mem_mb=3072,
        nominal_io=1
    shell:
        NOM_COMMAND + "adm --components {wildcards.components} --form quadratic"

# Single-day in-sample refits: between-day ADM variability (proxy summary)
# and the per-day ADM sensitivity variant.
rule fit_nominal_full_adm_day:
    input:
        selection=NOM_CANDIDATE_SUMMARY,
        candidates=lambda wc: f"{NOM_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib",
        metadata=lambda wc: f"{NOM_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib.json",
        cache=f"{NOMINAL['cache_root']}/{{day}}",
        abi="data/preprocessed_files/abi_{day}_res2km_step1.nc",
        code=NOM_CODE
    output:
        f"{NOM_ADM_DIR}/daily/adm_k{{components}}_day{{day}}.joblib",
        f"{NOM_ADM_DIR}/daily/adm_k{{components}}_day{{day}}.joblib.json"
    resources:
        mem_mb=2048,
        nominal_io=1
    shell:
        NOM_COMMAND + "adm --components {wildcards.components} --day {wildcards.day}"

rule assess_nominal_full_assignment_day:
    input:
        model=NOM_MODEL,
        metadata=NOM_MODEL + ".json",
        cache=f"{NOMINAL['cache_root']}/{{day}}",
        abi="data/preprocessed_files/abi_{day}_res2km_step1.nc",
        code=NOM_CODE
    output:
        f"{NOM_DIR}/assignment_parts/day{{day}}.json"
    resources:
        mem_mb=3072,
        nominal_io=1
    shell:
        NOM_COMMAND + "assignment-day --day {wildcards.day}"

rule assess_nominal_full_spatial_part:
    input:
        model=NOM_MODEL,
        metadata=NOM_MODEL + ".json",
        selection=f"{NOMINAL['diagnostic_dir']}/gmm_selection.json",
        library=NOM_NOMINAL_ADM,
        abi="data/preprocessed_files/abi_{day}_res2km_step1.nc",
        coefficients=config["narrowband_to_broadband_coeffs_file"],
        code=NOM_CODE
    output:
        f"{NOM_DIR}/spatial_parts/day{{day}}_b{{block}}.json"
    resources:
        mem_mb=3072,
        nominal_io=1
    shell:
        NOM_COMMAND + "spatial-part --day {wildcards.day} --block {wildcards.block}"

rule prepare_nominal_full_stability_sample:
    input:
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["training_days"]],
        code=NOM_CODE
    output:
        sample=f"{NOM_DIR}/stability_sample.npz",
        metadata=f"{NOM_DIR}/stability_sample.npz.json"
    resources:
        mem_mb=1024
    shell:
        NOM_COMMAND + "stability-sample"

rule diagnose_nominal_full_component:
    input:
        selection=NOM_CANDIDATE_SUMMARY,
        training_pca=f"{NOM_DIR}/training_pca",
        stability_sample=f"{NOM_DIR}/stability_sample.npz",
        stability_metadata=f"{NOM_DIR}/stability_sample.npz.json",
        library=f"{NOM_DIR}/component_adms/adm_k{{components}}.joblib",
        library_metadata=f"{NOM_DIR}/component_adms/adm_k{{components}}.joblib.json",
        candidates=lambda wc: f"{NOM_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib",
        metadata=lambda wc: f"{NOM_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib.json",
        abi=[f"data/preprocessed_files/abi_{day}_res2km_step1.nc"
             for day in NOMINAL["validation_days"]],
        coefficients=config["narrowband_to_broadband_coeffs_file"],
        code=NOM_CODE
    output:
        f"{NOMINAL['diagnostic_dir']}/component_parts/k{{components}}.json"
    resources:
        mem_mb=3072,
        nominal_io=1
    shell:
        NOM_COMMAND + "diagnostics-component --components {wildcards.components}"


GMM_SELECTION_FIGURES = [
    "figures/gmm_component_selection/diagnostics/gmm_selection_overview.png",
    "figures/gmm_component_selection/diagnostics/gmm_selection_overview_criteria.png",
    "figures/gmm_component_selection/diagnostics/gmm_selection_stability.png",
]


# K-independent review package for the manual component-count choice.
rule gmm_selection_diagnostics:
    input:
        f"{NOMINAL['diagnostic_dir']}/gmm_component_diagnostics.csv",
        f"{NOMINAL['diagnostic_dir']}/gmm_component_diagnostics.csv.json",
        GMM_SELECTION_FIGURES


rule plot_gmm_selection_diagnostics:
    input:
        diagnostics=f"{NOMINAL['diagnostic_dir']}/gmm_component_diagnostics.csv",
        diagnostics_metadata=f"{NOMINAL['diagnostic_dir']}/gmm_component_diagnostics.csv.json",
        candidates=NOM_CANDIDATE_SUMMARY,
        plotter="plotting/plot_n_components.py",
        code=["plotting/plot_gmm_selection.py", "plotting/plot_nominal_budget.py"]
    output:
        GMM_SELECTION_FIGURES
    params:
        output_dir="figures/gmm_component_selection/diagnostics"
    resources:
        mem_mb=1024
    shell:
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python plotting/plot_gmm_selection.py "
        "--diagnostics {input.diagnostics:q} --candidates {input.candidates:q} "
        "--output-dir {params.output_dir:q}"


rule uncertainty_nominal_budget:
    input:
        f"{NOM_DIR}/budget_report.json",
        f"{NOM_DIR}/numerical_budget.csv",
        f"{NOMINAL['figure_dir']}/eco_uncertainty_numerical_budget.png",
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_diagnostics.png",
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_stability.png",
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_diagnostics_criteria.png",
        f"{NOMINAL['diagnostic_figure_dir']}/spatial_processing.png",
        f"{NOMINAL['diagnostic_figure_dir']}/population_and_assignment.png",
        f"{NOMINAL['diagnostic_dir']}/gmm_selection.json",
        f"{NOMINAL['diagnostic_dir']}/scene_support.json"


rule train_nominal_spectral_gmm:
    input:
        caches=NOM_CACHE,
        candidates=NOM_CANDIDATES,
        candidate_metadata=[path + ".json" for path in NOM_CANDIDATES],
        scores=NOM_SCORES,
        # Full mode promotes the reviewed K; the gate must pass first.
        summary=[NOM_CANDIDATE_SUMMARY, SCENE_TAXONOMY_GATE],
        code=NOM_CODE
    output:
        NOM_MODEL,
        NOM_MODEL + ".json",
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
    return [f"data/preprocessed_files/abi_{day}_res2km_step1.nc"
            for day in NOMINAL["validation_days"]]


rule assess_nominal_abi_assignment:
    input:
        model=NOM_MODEL,
        metadata=NOM_MODEL + ".json",
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["validation_days"]],
        abi=nominal_abi_sources,
        parts=NOM_ASSIGNMENT_PARTS,
        code=NOM_CODE
    output:
        f"{NOM_DIR}/abi_assignment.json"
    resources:
        mem_mb=2048
    shell:
        NOM_COMMAND + "assignment-combine"


rule diagnose_nominal_gmm:
    input:
        caches=NOM_CACHE,
        selection=NOM_CANDIDATE_SUMMARY,
        candidates=NOM_CANDIDATES,
        coefficients=config["narrowband_to_broadband_coeffs_file"],
        parts=NOM_DIAGNOSTIC_PARTS,
        code=NOM_CODE
    output:
        f"{NOMINAL['diagnostic_dir']}/gmm_component_diagnostics.csv",
        f"{NOMINAL['diagnostic_dir']}/gmm_component_diagnostics.csv.json"
    resources:
        mem_mb=2048
    shell:
        NOM_COMMAND + "diagnostics-combine"


rule assess_nominal_spatial:
    input:
        model=NOM_MODEL,
        metadata=NOM_MODEL + ".json",
        caches=NOM_CACHE,
        abi=nominal_abi_sources,
        coefficients=config["narrowband_to_broadband_coeffs_file"],
        parts=NOM_SPATIAL_PARTS,
        library=NOM_NOMINAL_ADM,
        code=NOM_CODE
    output:
        f"{NOM_DIR}/spatial_report.json",
        f"{NOM_DIR}/spatial_residuals.npz",
        f"{NOM_DIR}/abi_spatial_adm.joblib"
    resources:
        mem_mb=2048
    shell:
        NOM_COMMAND + "spatial-combine"


rule calculate_nominal_budget:
    input:
        f"{NOM_DIR}/sunny.npz", f"{NOM_DIR}/sunny.npz.json",
        NOM_MODEL, NOM_MODEL + ".json",
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
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_stability.png",
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_diagnostics_criteria.png",
        f"{NOMINAL['diagnostic_figure_dir']}/spatial_processing.png",
        f"{NOMINAL['diagnostic_figure_dir']}/population_and_assignment.png"
    resources:
        mem_mb=1024
    shell:
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python plotting/plot_nominal_budget.py "
        "--report {input.report:q}"
