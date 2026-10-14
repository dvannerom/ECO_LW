with open("config/sensitivity_extensions.yaml") as handle:
    EXT = yaml.safe_load(handle)
EXT["figure_dir"] = str(figure_path(EXT["figure_dir"]))
EXT_DIR = EXT["output_dir"]
EXT_COMMON = [
    "config/sensitivity_extensions.yaml", "scripts/run_sensitivity_extensions.py",
    "src/geometry_sensitivity.py", "src/sensitivity.py", "src/uncertainty.py", "config.yaml",
]
EXT_COMMAND = (
    "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    "python scripts/run_sensitivity_extensions.py "
)


rule uncertainty_extended_diagnostics:
    input:
        f"{EXT_DIR}/eco_geometry.json",
        f"{EXT['figure_dir']}/eco_geometry.png"


rule eco_geometry_diagnostics:
    input:
        cache=f"{NOM_DIR}/sunny.npz",
        cache_metadata=f"{NOM_DIR}/sunny.npz.json",
        model=f"{NOM_MODEL_DIR}/cv_models.joblib",
        budget=f"{NOM_DIR}/budget_report.json",
        residuals=f"{NOM_DIR}/residuals.npz",
        code=EXT_COMMON
    output:
        f"{EXT_DIR}/eco_geometry.json"
    resources:
        mem_mb=2048
    shell:
        EXT_COMMAND + "geometry --cache {input.cache:q} --model {input.model:q} "
        "--budget {input.budget:q} --residuals {input.residuals:q} --output {output:q}"


rule plot_extended_diagnostics:
    input:
        geometry=f"{EXT_DIR}/eco_geometry.json",
        code=["plotting/plot_sensitivity_extensions.py", "src/figure_layout.py"]
    output:
        f"{EXT['figure_dir']}/eco_geometry.png"
    resources:
        mem_mb=1024
    shell:
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "python plotting/plot_sensitivity_extensions.py --geometry {input.geometry:q}"
