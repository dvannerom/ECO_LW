with open("config/sensitivity_extensions.yaml") as handle:
    EXT = yaml.safe_load(handle)
EXT["figure_dir"] = str(figure_path(EXT["figure_dir"]))
EXT_DIR = EXT["output_dir"]
EXT_COMMON = [
    "config/sensitivity_extensions.yaml", "scripts/run_sensitivity_extensions.py",
    "src/geometry_sensitivity.py", "config.yaml",
    "scripts/run_sensitivity_convergence.py", "src/sensitivity_diagnostics.py",
    *SENS_CODE,
]
EXT_COMMAND = (
    "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    "python scripts/run_sensitivity_extensions.py "
)


rule uncertainty_extended_diagnostics:
    input:
        f"{EXT_DIR}/eco_geometry.json",
        f"{EXT_DIR}/assignment_curve.json",
        f"{EXT['figure_dir']}/eco_geometry.png",
        f"{EXT['figure_dir']}/assignment_curve.png"


rule eco_geometry_diagnostics:
    input:
        cache=f"{SENS_DIR}/sunny/cache.npz",
        cache_metadata=f"{SENS_DIR}/sunny/cache.npz.json",
        model=f"{SENS_DIR}/sunny/models/baseline.joblib",
        model_metadata=f"{SENS_DIR}/sunny/models/baseline.joblib.json",
        code=EXT_COMMON
    output:
        f"{EXT_DIR}/eco_geometry.json"
    resources:
        mem_mb=1024
    shell:
        EXT_COMMAND + "geometry --cache {input.cache:q} --model {input.model:q} --output {output:q}"


rule assignment_curve_diagnostics:
    input:
        caches=[f"{CONV_DIR}/cache/{CONV['sampling_seeds'][0]}/{day}"
                for day in CONV["evaluation_days"]],
        model=f"{CONV_DIR}/models/t{CONV_MAX}_p{CONV_POINTS}_i{CONV_INIT}_c7.joblib",
        model_metadata=f"{CONV_DIR}/models/t{CONV_MAX}_p{CONV_POINTS}_i{CONV_INIT}_c7.joblib.json",
        reference=f"{CONV_DIR}/evaluation/t{CONV_MAX}_p{CONV_POINTS}_i{CONV_INIT}_s{CONV['sampling_seeds'][0]}.npz",
        reference_metadata=f"{CONV_DIR}/evaluation/t{CONV_MAX}_p{CONV_POINTS}_i{CONV_INIT}_s{CONV['sampling_seeds'][0]}.npz.json",
        coefficients=NARROWBAND_TO_BROADBAND_COEFFS_FILE,
        settings="config/sensitivity_convergence.yaml",
        code=EXT_COMMON
    output:
        f"{EXT_DIR}/assignment_curve.json"
    resources:
        mem_mb=2048
    shell:
        EXT_COMMAND + "assignment --caches {input.caches:q} --model {input.model:q} "
        "--reference {input.reference:q} --coefficients {input.coefficients:q} --output {output:q}"


rule plot_extended_diagnostics:
    input:
        geometry=f"{EXT_DIR}/eco_geometry.json",
        assignment=f"{EXT_DIR}/assignment_curve.json",
        code=["plotting/plot_sensitivity_extensions.py", "src/figure_layout.py"]
    output:
        f"{EXT['figure_dir']}/eco_geometry.png",
        f"{EXT['figure_dir']}/assignment_curve.png"
    resources:
        mem_mb=1024
    shell:
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "python plotting/plot_sensitivity_extensions.py --geometry {input.geometry:q} "
        "--assignment {input.assignment:q}"
