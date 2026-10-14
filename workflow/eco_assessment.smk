rule fit_eco_n2bc_coefficients:
    input:
        sunny=SUNNY_DIR,
        scenarios=ECO_CHANNEL_SCENARIOS_FILE,
        error_budget="config/error_budget.yaml",
        evaluator="scripts/evaluate_eco_spectral_reconstruction.py",
        response_module="src/eco_spectral_response.py",
        nominal_module="src/nominal_budget.py",
        uncertainty_module="src/uncertainty.py",
        scene_model=MODEL,
        sunny_abi=f"{NOMINAL_SETTINGS_CONFIG['output_dir']}/sunny.npz"
    output:
        models=ECO_N2BC_MODELS,
        metrics=ECO_N2BC_METRICS
    params:
        folds=ECO_SPECTRAL_CV_FOLDS
    shell:
        "PYTHONPATH=src python {input.evaluator} --stage fit-n2bc "
        "--sunny-dir {input.sunny} --channel-scenarios {input.scenarios} "
        "--error-budget {input.error_budget} --folds {params.folds} "
        "--scene-model {input.scene_model} --sunny-abi {input.sunny_abi} "
        "--n2bc-models-output {output.models} --n2bc-metrics-output {output.metrics}"

rule retrieve_eco_adm_flux:
    input:
        sunny=SUNNY_DIR,
        scene_model=MODEL,
        sunny_abi=f"{NOMINAL_SETTINGS_CONFIG['output_dir']}/sunny.npz",
        scenarios=ECO_CHANNEL_SCENARIOS_FILE,
        evaluator="scripts/evaluate_eco_spectral_reconstruction.py",
        response_module="src/eco_spectral_response.py",
        adm_module="src/adm.py",
        uncertainty_module="src/uncertainty.py"
    output:
        fluxes=ECO_ADM_OUTPUTS,
        metrics=ECO_ADM_METRICS
    params:
        adm_dir=ECO_ADM_DIR
    shell:
        "PYTHONPATH=src python {input.evaluator} --stage retrieve-adm "
        "--sunny-dir {input.sunny} --channel-scenarios {input.scenarios} "
        "--scene-model {input.scene_model} --sunny-abi {input.sunny_abi} "
        "--adm-output-dir {params.adm_dir} --adm-metrics-output {output.metrics}"

rule evaluate_eco_spectral_reconstruction:
    input:
        n2bc_models=ECO_N2BC_MODELS,
        n2bc_metrics=ECO_N2BC_METRICS,
        adm_fluxes=ECO_ADM_OUTPUTS,
        adm_metrics=ECO_ADM_METRICS,
        abi_adm_evidence=ABI_ADM_PROXY_SUMMARY,
        scenarios=ECO_CHANNEL_SCENARIOS_FILE,
        evaluator="scripts/evaluate_eco_spectral_reconstruction.py",
        uncertainty_module="src/uncertainty.py"
    output:
        metrics=ECO_SPECTRAL_METRICS,
        residuals=ECO_SPECTRAL_RESIDUALS
    params:
        adm_dir=ECO_ADM_DIR
    shell:
        "PYTHONPATH=src python {input.evaluator} --stage combine "
        "--n2bc-models-input {input.n2bc_models} "
        "--n2bc-metrics-input {input.n2bc_metrics} "
        "--adm-output-dir {params.adm_dir} --adm-metrics-input {input.adm_metrics} "
        "--abi-adm-evidence {input.abi_adm_evidence} "
        "--channel-scenarios {input.scenarios} "
        "--metrics-output {output.metrics} --residuals-output {output.residuals}"

rule plot_eco_spectral_assessment:
    input:
        metrics=ECO_SPECTRAL_METRICS,
        residuals=ECO_SPECTRAL_RESIDUALS,
        error_budget="config/error_budget.yaml",
        plotter="plotting/plot_eco_spectral_assessment.py"
    output:
        figures=ECO_SPECTRAL_FIGURES
    params:
        output_dir=ECO_SPECTRAL_FIGURE_DIR
    shell:
        "python {input.plotter} --metrics {input.metrics} "
        "--residuals {input.residuals} --error-budget {input.error_budget} "
        "--output-dir {params.output_dir}"
