rule build_uncertainty_registry:
    input:
        metrics=ECO_SPECTRAL_METRICS,
        residuals=ECO_SPECTRAL_RESIDUALS,
        n2bc_metrics=ECO_N2BC_METRICS,
        adm_metrics=ECO_ADM_METRICS,
        abi_summary=ABI_ADM_PROXY_SUMMARY,
        abi_narrowbands=NARROWBAND_OUTPUTS,
        abi_broadbands=BROADBAND_OUTPUTS,
        abi_monthly=config["output"]["monthly"],
        abi_sensitivity=SENS_REPORT,
        nominal_budget=NOMINAL_BUDGET_REPORT,
        scene_taxonomy_gate=SCENE_TAXONOMY_GATE,
        scenarios=ECO_CHANNEL_SCENARIOS_FILE,
        error_budget="config/error_budget.yaml",
        registry_builder="scripts/build_uncertainty_registry.py",
        uncertainty_module="src/uncertainty.py"
    output:
        registry=UNCERTAINTY_REGISTRY,
        report=UNCERTAINTY_GAP_REPORT
    shell:
        "PYTHONPATH=src python {input.registry_builder} "
        "--metrics {input.metrics} --residuals {input.residuals} "
        "--n2bc-metrics {input.n2bc_metrics} --adm-metrics {input.adm_metrics} "
        "--abi-summary {input.abi_summary} --scenarios {input.scenarios} "
        "--abi-narrowband-products {input.abi_narrowbands:q} "
        "--abi-daily-products {input.abi_broadbands:q} "
        "--abi-monthly-product {input.abi_monthly:q} "
        "--abi-sensitivity-report {input.abi_sensitivity:q} "
        "--nominal-budget-report {input.nominal_budget:q} "
        "--scene-taxonomy-gate {input.scene_taxonomy_gate:q} "
        "--error-budget {input.error_budget} "
        "--registry-output {output.registry} --report-output {output.report}"

rule uncertainty_assessment:
    input:
        UNCERTAINTY_REGISTRY,
        UNCERTAINTY_GAP_REPORT,
        *UNCERTAINTY_FIGURES,
        BUDGET_CSV

rule export_uncertainty_budget:
    input:
        registry=UNCERTAINTY_REGISTRY,
        sensitivity=SENS_REPORT,
        experiments="config/uncertainty_experiments.yaml",
        code=["scripts/export_budget_tables.py", "src/budget_reporting.py", "src/uncertainty.py"]
    output:
        BUDGET_CSV
    params:
        scenario=config.get("uncertainty_panel", {}).get("scenario", "rfma_goal_6"),
        mode=config.get("uncertainty_panel", {}).get("training_mode", "abi_population_weighted"),
        degree=config.get("uncertainty_panel", {}).get("degree", 2)
    shell:
        "python scripts/export_budget_tables.py registry --input {input.registry:q} "
        "--output {output:q} --experiment-spec {input.experiments:q} "
        "--sensitivity-report {input.sensitivity:q} --scenario {params.scenario:q} "
        "--training-mode {params.mode:q} --degree {params.degree}"

rule plot_uncertainty_registry:
    input:
        registry=UNCERTAINTY_REGISTRY,
        code=["plotting/plot_uncertainty_registry.py", "src/budget_reporting.py",
              "src/figure_layout.py"]
    output:
        variance_decomposition=UNCERTAINTY_FIGURES[0]
    shell:
        "python plotting/plot_uncertainty_registry.py --registry {input.registry:q} "
        "--variance-decomposition {output.variance_decomposition:q}"

rule summarize_abi_adm_proxy:
    input:
        adms=DAILY_ADM_OUTPUTS,
        metadata=[path + ".json" for path in DAILY_ADM_OUTPUTS],
        summarizer="scripts/summarize_abi_adm_proxy.py",
        code=PRODUCTION_CODE,
        uncertainty_module="src/uncertainty.py",
        error_budget="config/error_budget.yaml"
    output:
        ABI_ADM_PROXY_SUMMARY
    params:
        days=" ".join(str(day) for day in DAYS),
        components=SCENE_COMPONENTS,
        resolution=RESOLUTION
    shell:
        "PYTHONPATH=src python {input.summarizer} --inputs {input.adms} "
        "--days {params.days} --n-components {params.components} "
        "--resolution {params.resolution} --error-budget {input.error_budget} "
        "--output {output}"
