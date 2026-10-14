# Production-aligned sensitivity study (config/sensitivity_run.yaml, schema 2).
# Every variant perturbs one choice of the configuration `all` produces; the
# recomputed baselines must close on the production daily products (ABI) and
# on the nominal residuals (Sunny/ECO). Defined after nominal_budget.smk names.
import shlex

SENS_RUNNER = "scripts/run_product_sensitivity.py"
SENS_CODE = [
    SENS_RUNNER, "src/product_sensitivity.py", *PRODUCTION_CODE,
    "src/nominal_budget.py", "src/nominal_streaming.py", "src/scene_features.py",
    "src/geometry_sensitivity.py", "src/uncertainty.py", "scripts/run_nominal_budget.py",
    SENS_SETTINGS, NOMINAL_SETTINGS_PATH, "config.yaml", ECO_CHANNEL_SCENARIOS_FILE,
]
SENS_COMMAND = (
    "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    f"python {SENS_RUNNER} --settings {shlex.quote(str(SENS_SETTINGS))} "
    f"--nominal-settings {shlex.quote(str(NOMINAL_SETTINGS_PATH))} "
    f"--taxonomy {shlex.quote(str(SCENE_TAXONOMY_FILE))} "
    f"--coefficients {shlex.quote(str(NARROWBAND_TO_BROADBAND_COEFFS_FILE))} "
    f"--max-vza {PRODUCTION_MAX_VZA_DEG} "
)
SENS_DAYS = sorted(NOMINAL["training_days"] + NOMINAL["validation_days"])
if SENS_DAYS != sorted(DAYS):
    raise ValueError("config.yaml days must equal the nominal training plus validation days")
SENS_COMPONENT_SHIFTS = {"components_minus_1": -1, "components_plus_1": 1}


def sensitivity_shifted_components():
    """K-1/K+1 counts needed by any configured variant and present in the candidate range."""
    names = set(SENS["abi_variants"]) | set(SENS["sunny_variants"])
    return sorted(SCENE_COMPONENTS + shift for name, shift in SENS_COMPONENT_SHIFTS.items()
                  if name in names and SCENE_COMPONENTS + shift in NOMINAL["components"])


def sensitivity_candidates(wildcards=None):
    """Fixed-seed K+-1 candidates used by the sensitivity variants."""
    return [f"{NOM_DIR}/candidate_k{count}_s{NOM_GMM_SEED}.joblib{suffix}"
            for count in sensitivity_shifted_components()
            for suffix in ("", ".json")]


def sensitivity_abi_libraries(day):
    libraries = [PRODUCTION_ADM]
    if "adm_quadratic" in SENS["abi_variants"]:
        libraries.append(f"{NOM_ADM_DIR}/adm_k{SCENE_COMPONENTS}_quadratic.joblib")
    if "adm_daily" in SENS["abi_variants"]:
        libraries.append(f"{NOM_ADM_DIR}/daily/adm_k{SCENE_COMPONENTS}_day{day}.joblib")
    if set(SENS["abi_variants"]) & set(SENS_COMPONENT_SHIFTS):
        libraries += [f"{NOM_ADM_DIR}/adm_k{count}.joblib" for count in sensitivity_shifted_components()]
    return [path + suffix for path in libraries for suffix in ("", ".json")]


SENS_DAY_PARTS = [f"{SENS_DIR}/abi_days/day{day}.json" for day in SENS_DAYS]
SENS_DAY_MAPS = [f"{SENS_DIR}/abi_days/day{day}_maps.npz" for day in SENS_DAYS]
SENS_SUNNY = f"{SENS_DIR}/sunny_variants.json"
SENS_MONTHLY_MAPS = f"{SENS_DIR}/monthly_difference_maps.npz"


rule uncertainty_sensitivity_assessment:
    input:
        report=SENS_REPORT,
        variance_decomposition=UNCERTAINTY_FIGURES[0],
        csv="docs/Uncertainty budget and propagation for ECO_LW - Sheet1.csv",
        diagnostics=f"{SENS['figure_dir']}/broadband_sensitivity.png",
        maps=f"{SENS['figure_dir']}/monthly_difference_maps.png"


rule run_sensitivity_abi_day:
    input:
        data=lambda wc: preprocessed(wc.day),
        scene=lambda wc: scene_id(wc.day),
        product=lambda wc: broadband(wc.day),
        model=MODEL,
        model_metadata=MODEL + ".json",
        libraries=lambda wc: sensitivity_abi_libraries(wc.day),
        candidates=sensitivity_candidates,
        summary=NOM_CANDIDATE_SUMMARY,
        coefficients=NARROWBAND_TO_BROADBAND_COEFFS_FILE,
        code=SENS_CODE
    output:
        part=f"{SENS_DIR}/abi_days/day{{day}}.json",
        maps=f"{SENS_DIR}/abi_days/day{{day}}_maps.npz"
    resources:
        mem_mb=6144
    shell:
        SENS_COMMAND + "abi-day --day {wildcards.day} --output {output.part:q} --maps {output.maps:q}"


rule run_sensitivity_sunny:
    input:
        data=[f"{NOM_DIR}/sunny.npz", f"{NOM_DIR}/sunny.npz.json"],
        model=[MODEL, MODEL + ".json"],
        nominal=[NOMINAL_BUDGET_REPORT, f"{NOM_DIR}/residuals.npz"],
        candidates=sensitivity_candidates,
        summary=NOM_CANDIDATE_SUMMARY,
        code=SENS_CODE
    output:
        SENS_SUNNY
    resources:
        mem_mb=4096
    shell:
        SENS_COMMAND + "sunny --output {output:q}"


rule assemble_sensitivity_budget:
    input:
        days=SENS_DAY_PARTS,
        maps=SENS_DAY_MAPS,
        sunny=SENS_SUNNY,
        code=SENS_CODE
    output:
        report=SENS_REPORT,
        maps=SENS_MONTHLY_MAPS
    resources:
        mem_mb=2048
    shell:
        SENS_COMMAND + "assemble --days {input.days:q} --maps {input.maps:q} "
        "--sunny {input.sunny:q} --monthly-maps {output.maps:q} --output {output.report:q}"


rule plot_sensitivity_diagnostics:
    input:
        report=SENS_REPORT,
        maps=SENS_MONTHLY_MAPS,
        plotter="plotting/plot_uncertainty_sensitivity.py",
        uncertainty_module="src/uncertainty.py",
        layout="src/figure_layout.py"
    output:
        summary=f"{SENS['figure_dir']}/broadband_sensitivity.png",
        maps=f"{SENS['figure_dir']}/monthly_difference_maps.png"
    resources:
        mem_mb=2048
    shell:
        "python {input.plotter:q} --report {input.report:q} --maps {input.maps:q} "
        "--output {output.summary:q} --maps-output {output.maps:q}"
