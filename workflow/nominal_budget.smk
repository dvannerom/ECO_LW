rule assess_nominal_full_assignment_day:
    input:
        model=NOM_MODEL,
        metadata=NOM_MODEL + ".json",
        cache=f"{NOMINAL['cache_root']}/{{day}}",
        abi=lambda wc: preprocessed(wc.day),
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
        abi=lambda wc: preprocessed(wc.day),
        coefficients=config["narrowband_to_broadband_coeffs_file"],
        code=NOM_CODE
    output:
        f"{NOM_DIR}/spatial_parts/day{{day}}_b{{block}}.json"
    resources:
        mem_mb=3072,
        nominal_io=1
    shell:
        NOM_COMMAND + "spatial-part --day {wildcards.day} --block {wildcards.block}"

rule uncertainty_nominal_budget:
    input:
        f"{NOM_DIR}/budget_report.json",
        NOMINAL_PRODUCTS,
        f"{NOM_DIR}/numerical_budget.csv",
        f"{NOMINAL['figure_dir']}/eco_uncertainty_numerical_budget.png",
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_diagnostics.png",
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_stability.png",
        f"{NOMINAL['diagnostic_figure_dir']}/gmm_diagnostics_criteria.png",
        f"{NOMINAL['diagnostic_figure_dir']}/spatial_processing.png",
        f"{NOMINAL['diagnostic_figure_dir']}/population_and_assignment.png",
        f"{NOMINAL['diagnostic_dir']}/gmm_selection.json",
        f"{NOMINAL['diagnostic_dir']}/scene_support.json"

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
        *NOMINAL_PRODUCTS,
        f"{NOM_DIR}/residuals.npz",
        f"{NOM_MODEL_DIR}/cv_models.joblib",
        f"{NOM_MODEL_DIR}/nominal_models.joblib",
        f"{NOM_MODEL_DIR}/nominal_models.joblib.json",
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

rule export_nominal_budget:
    input:
        report=f"{NOM_DIR}/budget_report.json",
        code=["scripts/export_budget_tables.py", "src/budget_reporting.py", "src/uncertainty.py"]
    output:
        f"{NOM_DIR}/numerical_budget.csv"
    shell:
        "python scripts/export_budget_tables.py nominal --input {input.report:q} --output {output:q}"


rule nominal_flux_figures:
    input:
        expand("figures/nominal_production/products/{kind}/heldout_flux.png",
               kind=["narrowband_flux", "broadband_flux"])


rule plot_nominal_flux:
    input:
        product=lambda wc: next(path for path in NOMINAL_PRODUCTS
                                if f"/{wc.kind}/" in path and path.endswith(".npz")),
        metadata=lambda wc: next(path for path in NOMINAL_PRODUCTS
                                 if f"/{wc.kind}/" in path and path.endswith(".npz.json")),
        code=["plotting/plot_nominal_flux.py", "src/uncertainty.py"]
    output:
        "figures/nominal_production/products/{kind}/heldout_flux.png"
    wildcard_constraints:
        kind="narrowband_flux|broadband_flux"
    shell:
        "python plotting/plot_nominal_flux.py --input {input.product:q} --output {output:q}"
