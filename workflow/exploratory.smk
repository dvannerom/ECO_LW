rule abi_sunny_spectral_comparison:
    input:
        *ABI_SUNNY_COMPARISON_OUTPUTS,
        *ABI_SUNNY_POPULATION_OUTPUTS

rule sunny_scene_population:
    input:
        consistency=ABI_SUNNY_POPULATION_SUMMARY[0],
        summary=ABI_SUNNY_POPULATION_SUMMARY[1],
        plotter="plotting/plot_sunny_scene_population.py"
    output:
        ABI_SUNNY_POPULATION_OUTPUTS
    params:
        output_dir=ABI_SUNNY_POPULATION_DIR
    shell:
        "python {input.plotter:q} --input {input.consistency:q} "
        "--summary {input.summary:q} --output-dir {params.output_dir:q}"

rule summarize_sunny_population:
    input:
        classifications=f"{ABI_SUNNY_COMPARISON_DIR}/sunny_scenes.csv",
        summary=f"{ABI_SUNNY_COMPARISON_DIR}/summary.json",
        code=["scripts/summarize_sunny_population.py", "src/scene_population.py"]
    output:
        csv=ABI_SUNNY_POPULATION_SUMMARY[0],
        report=ABI_SUNNY_POPULATION_SUMMARY[1]
    shell:
        "python scripts/summarize_sunny_population.py --input {input.classifications:q} "
        "--summary {input.summary:q} --csv {output.csv:q} --report {output.report:q}"

rule compare_abi_sunny_scenes:
    threads: ABI_SUNNY_COMPARISON.get("gmm_jobs", 1)
    input:
        training=TRAINING_FILES,
        sunny=sorted(str(path) for path in Path(SUNNY_DIR).glob("radiance_lw_*/sunny_lw_*")),
        filters=[f"{GOES_FILTER_DIR}/goes-r_abi_{channel}" for channel in
                 ("ch08", "ch11", "ch12", "ch14", "ch15", "ch16")],
        code=[
            "scripts/compare_abi_sunny_scenes.py", "src/abi_sunny_comparison.py",
            "src/scene_features.py", "src/spectral_response.py",
            "src/radiometry.py", "src/parallel_gmm.py",
        ]
    output:
        ABI_SUNNY_COMPARISON_OUTPUTS
    params:
        output_dir=ABI_SUNNY_COMPARISON_DIR,
        sunny_dir=SUNNY_DIR,
        filter_dir=GOES_FILTER_DIR,
        components=ABI_SUNNY_COMPARISON.get("components", 7),
        points_per_file=ABI_SUNNY_COMPARISON.get("points_per_file", 50000),
        pixel_step=ABI_SUNNY_COMPARISON.get("pixel_step", 10),
        chunk_rows=ABI_SUNNY_COMPARISON.get("chunk_rows", 64),
        seed=ABI_SUNNY_COMPARISON.get("seed", 42),
        roundtrip_limit=ABI_SUNNY_COMPARISON.get("max_roundtrip_error_K", 0.1)
    shell:
        "PYTHONPATH=src python scripts/compare_abi_sunny_scenes.py "
        "--abi-files {input.training:q} --sunny-dir {params.sunny_dir:q} "
        "--filter-dir {params.filter_dir:q} --output-dir {params.output_dir:q} "
        "--components {params.components} --points-per-file {params.points_per_file} "
        "--pixel-step {params.pixel_step} --chunk-rows {params.chunk_rows} "
        "--seed {params.seed} --gmm-jobs {threads} "
        "--max-roundtrip-error-K {params.roundtrip_limit}"
