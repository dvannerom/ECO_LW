SENS_RUNNER = "scripts/run_uncertainty_sensitivity.py"
SENS_CODE = [
    SENS_RUNNER, "src/sensitivity.py", "src/scene_features.py", "src/adm.py",
    "src/adm_fitting.py", "src/broadband.py", "src/radiometry.py",
    "src/eco_spectral_response.py", "src/uncertainty.py",
]
SENS_COMMON = [
    SENS_SETTINGS, "config.yaml", "config/error_budget.yaml",
    "config/uncertainty_experiments.yaml", ECO_CHANNEL_SCENARIOS_FILE, *SENS_CODE,
]
SENS_COMMAND = (
    "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    f"python {SENS_RUNNER} --settings {SENS_SETTINGS} "
)
SENS_TRAINING = [f"{SENS_DIR}/abi/cache/{day}.npz" for day in SENS["training_days"]]
SENS_EVALUATION = [f"{SENS_DIR}/abi/cache/{day}.npz" for day in SENS["evaluation_days"]]
SENS_BASE_COMPONENTS = SENS["abi"]["baseline_components"]
SENS_ABI_VARIANTS = ["baseline", "spatial", "assignment", "adm"] + [
    f"components_{count}" for count in SENS["abi"]["components"] if count != SENS_BASE_COMPONENTS
]
SENS_SUNNY_VARIANTS = ["baseline", "spectral", "adm", "robust", "noise", "joint"]


def sensitivity_realizations(variant):
    return range(SENS["realizations"]) if variant in ("noise", "assignment", "joint") else [0]


SENS_COMPARISONS = [
    f"{SENS_DIR}/comparisons/{branch}_{variant}_{realization}.json"
    for branch, variants in (("abi", SENS_ABI_VARIANTS), ("sunny", SENS_SUNNY_VARIANTS))
    for variant in variants if variant != "baseline"
    for realization in sensitivity_realizations(variant)
]


def sensitivity_components(variant):
    return int(variant.removeprefix("components_")) if variant.startswith("components_") else SENS_BASE_COMPONENTS


def sensitivity_form(variant):
    return "quadratic" if variant in ("adm", "joint") else "regularized"


def sensitivity_target_report(wildcards):
    if not USE_SENSITIVITY:
        raise ValueError(
            "Enable the first assessment with --config uncertainty_sensitivity=true "
            "so budget plotting depends on the new report."
        )
    return SENS_REPORT


rule uncertainty_sensitivity_assessment:
    input:
        report=sensitivity_target_report,
        figures=UNCERTAINTY_FIGURES,
        csv="docs/Uncertainty budget and propagation for ECO_LW - Sheet1.csv",
        diagnostics=f"{SENS['figure_dir']}/broadband_sensitivity.png"


rule prepare_sensitivity_abi:
    input:
        data=lambda wc: f"data/preprocessed_files/abi_{wc.day}_res2km_step1.nc",
        code=SENS_COMMON
    output:
        cache=f"{SENS_DIR}/abi/cache/{{day}}.npz",
        metadata=f"{SENS_DIR}/abi/cache/{{day}}.npz.json"
    resources:
        mem_mb=1024
    shell:
        SENS_COMMAND + "prepare-abi --day {wildcards.day} "
        "--inputs {input.data:q} --output {output.cache:q}"


rule train_sensitivity_gmm:
    input:
        caches=SENS_TRAINING,
        metadata=[path + ".json" for path in SENS_TRAINING],
        code=SENS_COMMON
    output:
        model=f"{SENS_DIR}/abi/models/gmm_{{components}}.joblib",
        metadata=f"{SENS_DIR}/abi/models/gmm_{{components}}.joblib.json"
    resources:
        mem_mb=2048
    shell:
        SENS_COMMAND + "train-gmm --inputs {input.caches:q} "
        "--components {wildcards.components} --output {output.model:q}"


rule fit_sensitivity_abi_adm:
    input:
        caches=SENS_TRAINING,
        metadata=[path + ".json" for path in SENS_TRAINING],
        model=f"{SENS_DIR}/abi/models/gmm_{{components}}.joblib",
        model_metadata=f"{SENS_DIR}/abi/models/gmm_{{components}}.joblib.json",
        code=SENS_COMMON
    output:
        library=f"{SENS_DIR}/abi/models/adm_{{components}}_{{form}}.npz",
        metadata=f"{SENS_DIR}/abi/models/adm_{{components}}_{{form}}.npz.json"
    resources:
        mem_mb=2048
    shell:
        SENS_COMMAND + "fit-abi-adm --inputs {input.caches:q} "
        "--model {input.model:q} --components {wildcards.components} "
        "--form {wildcards.form} --output {output.library:q}"


rule run_sensitivity_abi:
    input:
        caches=SENS_EVALUATION,
        metadata=[path + ".json" for path in SENS_EVALUATION],
        model=lambda wc: f"{SENS_DIR}/abi/models/gmm_{sensitivity_components(wc.variant)}.joblib",
        model_metadata=lambda wc: f"{SENS_DIR}/abi/models/gmm_{sensitivity_components(wc.variant)}.joblib.json",
        library=lambda wc: f"{SENS_DIR}/abi/models/adm_{sensitivity_components(wc.variant)}_{sensitivity_form(wc.variant)}.npz",
        library_metadata=lambda wc: f"{SENS_DIR}/abi/models/adm_{sensitivity_components(wc.variant)}_{sensitivity_form(wc.variant)}.npz.json",
        coefficients=NARROWBAND_TO_BROADBAND_COEFFS_FILE,
        code=SENS_COMMON
    output:
        flux=f"{SENS_DIR}/abi/outputs/{{variant}}_{{realization}}.npz",
        metadata=f"{SENS_DIR}/abi/outputs/{{variant}}_{{realization}}.npz.json"
    params:
        form=lambda wc: sensitivity_form(wc.variant)
    resources:
        mem_mb=2048
    shell:
        SENS_COMMAND + "run-abi --inputs {input.caches:q} --model {input.model:q} "
        "--library {input.library:q} --variant {wildcards.variant} "
        "--form {params.form} --realization {wildcards.realization} --output {output.flux:q}"


rule prepare_sensitivity_sunny:
    input:
        spectra=sorted(str(path) for path in Path(SUNNY_DIR).glob("radiance_lw_*/sunny_lw_*")),
        code=SENS_COMMON
    output:
        cache=f"{SENS_DIR}/sunny/cache.npz",
        metadata=f"{SENS_DIR}/sunny/cache.npz.json"
    resources:
        mem_mb=1024
    shell:
        SENS_COMMAND + "prepare-sunny --output {output.cache:q}"


rule fit_sensitivity_sunny:
    input:
        cache=f"{SENS_DIR}/sunny/cache.npz",
        metadata=f"{SENS_DIR}/sunny/cache.npz.json",
        code=SENS_COMMON
    output:
        model=f"{SENS_DIR}/sunny/models/{{variant}}.joblib",
        metadata=f"{SENS_DIR}/sunny/models/{{variant}}.joblib.json"
    resources:
        mem_mb=1024
    shell:
        SENS_COMMAND + "fit-sunny --inputs {input.cache:q} --variant {wildcards.variant} "
        "--output {output.model:q}"


rule run_sensitivity_sunny:
    input:
        cache=f"{SENS_DIR}/sunny/cache.npz",
        metadata=f"{SENS_DIR}/sunny/cache.npz.json",
        model=lambda wc: f"{SENS_DIR}/sunny/models/{'robust' if wc.variant in ('robust', 'joint') else 'baseline'}.joblib",
        model_metadata=lambda wc: f"{SENS_DIR}/sunny/models/{'robust' if wc.variant in ('robust', 'joint') else 'baseline'}.joblib.json",
        code=SENS_COMMON
    output:
        flux=f"{SENS_DIR}/sunny/outputs/{{variant}}_{{realization}}.npz",
        metadata=f"{SENS_DIR}/sunny/outputs/{{variant}}_{{realization}}.npz.json"
    params:
        form=lambda wc: sensitivity_form(wc.variant)
    resources:
        mem_mb=1024
    shell:
        SENS_COMMAND + "run-sunny --inputs {input.cache:q} --model {input.model:q} "
        "--variant {wildcards.variant} --form {params.form} "
        "--realization {wildcards.realization} --output {output.flux:q}"


rule compare_sensitivity:
    wildcard_constraints:
        branch="abi|sunny",
        realization=r"\d+"
    input:
        variant=f"{SENS_DIR}/{{branch}}/outputs/{{variant}}_{{realization}}.npz",
        metadata=f"{SENS_DIR}/{{branch}}/outputs/{{variant}}_{{realization}}.npz.json",
        baseline=f"{SENS_DIR}/{{branch}}/outputs/baseline_0.npz",
        baseline_metadata=f"{SENS_DIR}/{{branch}}/outputs/baseline_0.npz.json",
        code=SENS_COMMON
    output:
        f"{SENS_DIR}/comparisons/{{branch}}_{{variant}}_{{realization}}.json"
    shell:
        SENS_COMMAND + "compare --inputs {input.variant:q} {input.baseline:q} --output {output:q}"


rule assemble_sensitivity_budget:
    input:
        comparisons=SENS_COMPARISONS,
        baseline=f"{SENS_DIR}/sunny/outputs/baseline_0.npz",
        spectral=f"{SENS_DIR}/sunny/outputs/spectral_0.npz",
        paired_fluxes=[
            f"{SENS_DIR}/sunny/outputs/{variant}_{realization}.npz"
            for variant in ("baseline", "spectral", "adm", "robust", "noise", "joint")
            for realization in sensitivity_realizations(variant)
        ],
        paired_metadata=[
            f"{SENS_DIR}/sunny/outputs/{variant}_{realization}.npz.json"
            for variant in ("baseline", "spectral", "adm", "robust", "noise", "joint")
            for realization in sensitivity_realizations(variant)
        ],
        code=SENS_COMMON
    output:
        SENS_REPORT
    shell:
        SENS_COMMAND + "assemble --inputs {input.comparisons:q} "
        "--paired-fluxes {input.paired_fluxes:q} --output {output:q}"


rule plot_sensitivity_diagnostics:
    input:
        report=SENS_REPORT,
        plotter="plotting/plot_uncertainty_sensitivity.py",
        uncertainty_module="src/uncertainty.py",
        layout="src/figure_layout.py"
    output:
        f"{SENS['figure_dir']}/broadband_sensitivity.png"
    shell:
        "python {input.plotter:q} --report {input.report:q} --output {output:q}"
