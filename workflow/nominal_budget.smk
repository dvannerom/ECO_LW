import shlex

NOM_SETTINGS = config.get("nominal_budget_settings", "config/nominal_budget.yaml")
with open(NOM_SETTINGS) as handle:
    NOMINAL = yaml.safe_load(handle)
NOM_FULL = NOMINAL.get("abi_mode", "sampled") == "full"
NOM_DIR = NOMINAL["output_dir"]
NOM_INITIALIZATION_DIR = f"{NOM_DIR}/candidate_initializations"
NOM_CODE = [
    "scripts/run_nominal_budget.py", "src/nominal_budget.py",
    NOM_SETTINGS, "config.yaml", ECO_CHANNEL_SCENARIOS_FILE,
    "src/nominal_streaming.py", "src/streaming_gmm.py",
    "src/gmm_stability.py",
    "src/abi_sunny_comparison.py", "src/geometry_sensitivity.py",
    "src/spectral_response.py",
    "scripts/find_nComponents.py", "scripts/average_resolution.py",
    "scripts/preprocess_data_ABI.py", "src/broadband.py",
    "scripts/run_sensitivity_convergence.py", *SENS_CODE,
]
NOM_COMMAND = (
    "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    f"python scripts/run_nominal_budget.py --settings {shlex.quote(str(NOM_SETTINGS))} "
)
NOM_CACHE = [f"{NOMINAL['cache_root']}/{day}"
             for day in NOMINAL["training_days"] + NOMINAL["validation_days"]]
NOM_CANDIDATES = (
    [f"{NOM_DIR}/candidate_k{count}_s{seed}.joblib"
     for count in NOMINAL["components"] for seed in NOMINAL["seeds"]] if NOM_FULL else []
)
NOM_SCORES = (
    [f"{NOM_DIR}/candidate_scores/k{count}_s{seed}.json"
     for count in NOMINAL["components"] for seed in NOMINAL["seeds"]] if NOM_FULL else []
)
NOM_ASSIGNMENT_PARTS = (
    [f"{NOM_DIR}/assignment_parts/day{day}.json"
     for day in NOMINAL["validation_days"]] if NOM_FULL else []
)
NOM_SPATIAL_PARTS = (
    [f"{NOM_DIR}/spatial_parts/day{day}_b{block}.json"
     for day in NOMINAL["validation_days"]
     for block in NOMINAL["spatial"]["block_sizes"]] if NOM_FULL else []
)
NOM_DIAGNOSTIC_PARTS = (
    [f"{NOMINAL['diagnostic_dir']}/component_parts/k{count}.json"
     for count in NOMINAL["components"]] if NOM_FULL else []
)
NOM_NOMINAL_ADM = (
    [f"{NOM_DIR}/component_adms/adm_k{NOMINAL['nominal_components']}.joblib",
     f"{NOM_DIR}/component_adms/adm_k{NOMINAL['nominal_components']}.joblib.json"]
    if NOM_FULL else []
)


def nominal_initialization_paths(wildcards):
    return [
        f"{NOM_INITIALIZATION_DIR}/candidate_k{wildcards.components}_s{wildcards.seed}_i{index}.joblib"
        for index in range(NOMINAL["gmm_initializations"])
    ]


if NOM_FULL:
    rule prepare_nominal_full_abi:
        input:
            data=lambda wc: f"data/preprocessed_files/abi_{wc.day}_res2km_step1.nc",
            code=NOM_CODE
        output:
            directory(f"{NOMINAL['cache_root']}/{{day}}")
        resources:
            mem_mb=3072,
            nominal_io=1
        shell:
            NOM_COMMAND + "cache --day {wildcards.day}"

    rule prepare_nominal_full_training:
        input:
            caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["training_days"]],
            code=NOM_CODE
        output:
            directory(f"{NOM_DIR}/training_pca")
        resources:
            mem_mb=1024
        shell:
            NOM_COMMAND + "preprocessing"

    rule fit_nominal_full_initialization:
        input:
            caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["training_days"]],
            preprocessing=f"{NOM_DIR}/training_pca",
            code=NOM_CODE
        output:
            model=f"{NOM_INITIALIZATION_DIR}/candidate_k{{components}}_s{{seed}}_i{{initialization}}.joblib",
            metadata=f"{NOM_INITIALIZATION_DIR}/candidate_k{{components}}_s{{seed}}_i{{initialization}}.joblib.json"
        threads:
            NOMINAL.get("gmm_workers", 1)
        resources:
            mem_mb=lambda wildcards, threads: 1024 + 384 * threads
        shell:
            NOM_COMMAND + "candidate-init --components {wildcards.components} --seed {wildcards.seed} --initialization {wildcards.initialization} --workers {threads}"

    rule fit_nominal_full_candidate:
        input:
            initializations=nominal_initialization_paths,
            metadata=lambda wc: [path + ".json" for path in nominal_initialization_paths(wc)],
            code=NOM_CODE
        output:
            f"{NOM_DIR}/candidate_k{{components}}_s{{seed}}.joblib",
            f"{NOM_DIR}/candidate_k{{components}}_s{{seed}}.joblib.json"
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

    rule fit_nominal_full_adm:
        input:
            selection=f"{NOMINAL['diagnostic_dir']}/gmm_selection.json",
            model=f"{NOM_DIR}/spectral_gmm.joblib",
            candidates=lambda wc: [
                f"{NOM_DIR}/candidate_k{wc.components}_s{seed}.joblib"
                for seed in NOMINAL["seeds"]],
            metadata=lambda wc: [
                f"{NOM_DIR}/candidate_k{wc.components}_s{seed}.joblib.json"
                for seed in NOMINAL["seeds"]],
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

    rule assess_nominal_full_assignment_day:
        input:
            model=f"{NOM_DIR}/spectral_gmm.joblib",
            metadata=f"{NOM_DIR}/spectral_gmm.joblib.json",
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
            model=f"{NOM_DIR}/spectral_gmm.joblib",
            metadata=f"{NOM_DIR}/spectral_gmm.joblib.json",
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
            selection=f"{NOMINAL['diagnostic_dir']}/gmm_selection.json",
            stability_sample=f"{NOM_DIR}/stability_sample.npz",
            stability_metadata=f"{NOM_DIR}/stability_sample.npz.json",
            library=f"{NOM_DIR}/component_adms/adm_k{{components}}.joblib",
            library_metadata=f"{NOM_DIR}/component_adms/adm_k{{components}}.joblib.json",
            candidates=lambda wc: [
                f"{NOM_DIR}/candidate_k{wc.components}_s{seed}.joblib"
                for seed in NOMINAL["seeds"]],
            metadata=lambda wc: [
                f"{NOM_DIR}/candidate_k{wc.components}_s{seed}.joblib.json"
                for seed in NOMINAL["seeds"]],
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
    if NOM_FULL:
        return [f"data/preprocessed_files/abi_{day}_res2km_step1.nc"
                for day in NOMINAL["validation_days"]]
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
        parts=NOM_ASSIGNMENT_PARTS,
        code=NOM_CODE
    output:
        f"{NOM_DIR}/abi_assignment.json"
    resources:
        mem_mb=2048
    shell:
        NOM_COMMAND + ("assignment-combine" if NOM_FULL else "assignment")


rule diagnose_nominal_gmm:
    input:
        caches=NOM_CACHE,
        model=f"{NOM_DIR}/spectral_gmm.joblib",
        selection=f"{NOMINAL['diagnostic_dir']}/gmm_selection.json",
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
        NOM_COMMAND + ("diagnostics-combine" if NOM_FULL else "diagnostics")


rule assess_nominal_spatial:
    input:
        model=f"{NOM_DIR}/spectral_gmm.joblib",
        metadata=f"{NOM_DIR}/spectral_gmm.joblib.json",
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
        NOM_COMMAND + ("spatial-combine" if NOM_FULL else "spatial")


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
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_stability.png",
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_diagnostics_criteria.png",
        f"{NOMINAL['diagnostic_figure_dir']}/spatial_processing.png",
        f"{NOMINAL['diagnostic_figure_dir']}/population_and_assignment.png"
    resources:
        mem_mb=1024
    shell:
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python plotting/plot_nominal_budget.py "
        "--report {input.report:q}"
