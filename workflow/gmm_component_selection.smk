rule prepare_gmm_stability_sample:
    input:
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["training_days"]],
        code=NOM_CODE
    output:
        sample=f"{NOM_CANDIDATE_DIR}/stability_sample.npz",
        metadata=f"{NOM_CANDIDATE_DIR}/stability_sample.npz.json"
    resources:
        mem_mb=1024
    shell:
        NOM_COMMAND + "stability-sample"

rule diagnose_gmm_component:
    input:
        selection=NOM_CANDIDATE_SUMMARY,
        training_pca=f"{NOM_TRAINING_DIR}",
        stability_sample=f"{NOM_CANDIDATE_DIR}/stability_sample.npz",
        stability_metadata=f"{NOM_CANDIDATE_DIR}/stability_sample.npz.json",
        library=f"{NOM_ADM_DIR}/adm_k{{components}}.joblib",
        library_metadata=f"{NOM_ADM_DIR}/adm_k{{components}}.joblib.json",
        candidates=lambda wc: f"{NOM_CANDIDATE_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib",
        metadata=lambda wc: f"{NOM_CANDIDATE_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib.json",
        abi=[preprocessed(day)
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





# K-independent review package for the manual component-count choice.
def full_review_diagnostics(wildcards):
    review_path = SCENE_TAXONOMY["diagnostics"]["settings"]
    reviewed_components = load_settings(review_path)["components"]
    if sorted(NOMINAL["components"]) != sorted(reviewed_components):
        raise ValueError(
            "The GMM selection package requires the full review range. "
            f"Use --config nominal_budget_settings={review_path}"
        )
    return f"{NOMINAL['diagnostic_dir']}/gmm_component_diagnostics.csv"

rule gmm_selection_diagnostics:
    input:
        full_review_diagnostics,
        f"{NOMINAL['diagnostic_dir']}/gmm_component_diagnostics.csv.json",
        GMM_SELECTION_FIGURES

rule plot_gmm_selection_diagnostics:
    input:
        diagnostics=full_review_diagnostics,
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

rule assemble_gmm_diagnostics:
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
