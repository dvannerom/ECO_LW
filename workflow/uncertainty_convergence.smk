with open("config/sensitivity_convergence.yaml") as handle:
    CONV = yaml.safe_load(handle)
CONV["figure_dir"] = str(figure_path(CONV["figure_dir"]))

CONV_DIR = CONV["output_dir"]
CONV_STUDY = "config/sensitivity_convergence.yaml"
CONV_COMMON = [
    CONV_STUDY, CONV["base_settings"], "config.yaml",
    "scripts/run_sensitivity_convergence.py", "src/sensitivity_diagnostics.py",
    *SENS_CODE,
]
CONV_COMMAND = (
    "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    "python scripts/run_sensitivity_convergence.py "
)
CONV_MAX = max(CONV["tile_counts"])
CONV_SEED = CONV["training_sampling_seed"]
CONV_INIT = CONV["initialization_seeds"][0]
CONV_POINTS = CONV["fixed_gmm_points"]
CONV_TRAIN = [
    f"{CONV_DIR}/cache/{CONV_SEED}/{day}" for day in CONV["training_days"]
]
CONV_CASES = sorted(set(
    [(tiles, CONV_POINTS, CONV_INIT) for tiles in CONV["tile_counts"]]
    + [(CONV_MAX, points, CONV_INIT) for points in CONV["gmm_point_counts"]]
    + [(CONV_MAX, CONV_POINTS, seed) for seed in CONV["initialization_seeds"]]
))
CONV_EVALUATIONS = sorted(set(
    [(CONV_MAX, CONV_POINTS, CONV_INIT, seed) for seed in CONV["sampling_seeds"]]
    + [(tiles, points, init, CONV["sampling_seeds"][0])
       for tiles, points, init in CONV_CASES]
))
CONV_RESULTS = [
    f"{CONV_DIR}/evaluation/t{tiles}_p{points}_i{init}_s{seed}.npz"
    for tiles, points, init, seed in CONV_EVALUATIONS
]
CONV_REPORT = f"{CONV_DIR}/convergence_report.json"
CONV_FIGURES = [
    f"{CONV['figure_dir']}/{name}.png"
    for name in ("convergence", "regime_diagnostics", "adm_profiles")
]


rule uncertainty_convergence_assessment:
    input:
        CONV_REPORT,
        CONV_FIGURES


rule prepare_convergence_abi:
    input:
        data=lambda wc: f"data/preprocessed_files/abi_{wc.day}_res2km_step1.nc",
        code=CONV_COMMON
    output:
        cache=directory(f"{CONV_DIR}/cache/{{seed}}/{{day}}")
    resources:
        mem_mb=4096
    shell:
        CONV_COMMAND + "prepare --day {wildcards.day} --sampling-seed {wildcards.seed} "
        "--inputs {input.data:q} --output {output.cache:q}"


rule train_convergence_abi:
    input:
        caches=CONV_TRAIN,
        code=CONV_COMMON
    output:
        model=f"{CONV_DIR}/models/t{{tiles}}_p{{points}}_i{{init}}_c{{components}}.joblib",
        metadata=f"{CONV_DIR}/models/t{{tiles}}_p{{points}}_i{{init}}_c{{components}}.joblib.json"
    resources:
        mem_mb=2048
    shell:
        CONV_COMMAND + "train --inputs {input.caches:q} --tiles {wildcards.tiles} "
        "--points {wildcards.points} --initialization-seed {wildcards.init} "
        "--components {wildcards.components} --output {output.model:q}"


rule evaluate_convergence_abi:
    input:
        caches=lambda wc: [f"{CONV_DIR}/cache/{wc.seed}/{day}"
                           for day in CONV["evaluation_days"]],
        models=lambda wc: [
            f"{CONV_DIR}/models/t{wc.tiles}_p{wc.points}_i{wc.init}_c{component}.joblib"
            for component in (7, 6)
        ],
        metadata=lambda wc: [
            f"{CONV_DIR}/models/t{wc.tiles}_p{wc.points}_i{wc.init}_c{component}.joblib.json"
            for component in (7, 6)
        ],
        coefficients=NARROWBAND_TO_BROADBAND_COEFFS_FILE,
        code=CONV_COMMON
    output:
        flux=f"{CONV_DIR}/evaluation/t{{tiles}}_p{{points}}_i{{init}}_s{{seed}}.npz",
        metadata=f"{CONV_DIR}/evaluation/t{{tiles}}_p{{points}}_i{{init}}_s{{seed}}.npz.json"
    resources:
        mem_mb=1024
    shell:
        CONV_COMMAND + "evaluate --inputs {input.caches:q} --models {input.models:q} "
        "--tiles {wildcards.tiles} --points {wildcards.points} "
        "--initialization-seed {wildcards.init} --sampling-seed {wildcards.seed} "
        "--output {output.flux:q}"


rule summarize_convergence:
    input:
        runs=CONV_RESULTS,
        metadata=[path + ".json" for path in CONV_RESULTS],
        code=CONV_COMMON
    output:
        CONV_REPORT
    resources:
        mem_mb=1024
    shell:
        CONV_COMMAND + "summarize --inputs {input.runs:q} --output {output:q}"


rule plot_convergence:
    input:
        report=CONV_REPORT,
        model=f"{CONV_DIR}/models/t{CONV_MAX}_p{CONV_POINTS}_i{CONV_INIT}_c7.joblib",
        metadata=f"{CONV_DIR}/models/t{CONV_MAX}_p{CONV_POINTS}_i{CONV_INIT}_c7.joblib.json",
        code=["plotting/plot_sensitivity_convergence.py", "src/sensitivity_diagnostics.py", "src/figure_layout.py"]
    output:
        CONV_FIGURES
    resources:
        mem_mb=1024
    shell:
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg "
        "python plotting/plot_sensitivity_convergence.py --report {input.report:q} "
        "--model {input.model:q}"
