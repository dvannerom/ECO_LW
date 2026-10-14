rule prepare_abi_features:
    input:
        data=lambda wc: preprocessed(wc.day),
        code=NOM_FEATURE_CODE
    output:
        directory(f"{NOMINAL['cache_root']}/{{day}}")
    params:
        stage_signature=lambda wc: nominal_stage_signature(
            "features", NOMINAL, {"day": int(wc.day)}
        )
    resources:
        mem_mb=3072,
        nominal_io=1
    shell:
        NOM_COMMAND + "cache --day {wildcards.day}"

rule prepare_scene_training:
    input:
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["training_days"]],
        code=NOM_PCA_CODE
    output:
        directory(f"{NOM_TRAINING_DIR}")
    params:
        stage_signature=nominal_stage_signature("pca", NOMINAL)
    resources:
        mem_mb=1024
    shell:
        NOM_COMMAND + "preprocessing"

rule fit_scene_candidates:
    input:
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["training_days"]],
        preprocessing=f"{NOM_TRAINING_DIR}",
        code=NOM_GMM_CODE
    output:
        f"{NOM_CANDIDATE_DIR}/initialization_sweep.json"
    params:
        # Queue membership affects the manifest but not already fitted starts.
        stage_signature=nominal_stage_signature("sweep", NOMINAL),
        requested_jobs_signature=nominal_stage_signature(
            "sweep", NOMINAL,
            {
                "components": NOMINAL["components"],
                "seed": NOM_GMM_SEED,
                "initializations": list(range(NOMINAL["gmm_initializations"])),
            },
        )
    threads:
        workflow.cores
    resources:
        mem_mb=lambda wildcards, threads: 1024 + 384 * threads
    shell:
        NOM_COMMAND + "candidate-sweep --workers {threads}"

rule combine_scene_candidate:
    input:
        sweep=ancient(f"{NOM_CANDIDATE_DIR}/initialization_sweep.json"),
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["training_days"]],
        preprocessing=f"{NOM_TRAINING_DIR}",
        code=NOM_GMM_CODE
    output:
        f"{NOM_CANDIDATE_DIR}/candidate_k{{components}}_s{{seed}}.joblib",
        f"{NOM_CANDIDATE_DIR}/candidate_k{{components}}_s{{seed}}.joblib.json"
    params:
        stage_signature=lambda wc: nominal_stage_signature(
            "candidate", NOMINAL,
            {
                "components": int(wc.components),
                "seed": int(wc.seed),
                "initializations": list(range(NOMINAL["gmm_initializations"])),
            },
        )
    resources:
        mem_mb=512
    shell:
        NOM_COMMAND + "candidate-combine --components {wildcards.components} --seed {wildcards.seed}"

rule score_scene_candidate:
    input:
        model=f"{NOM_CANDIDATE_DIR}/candidate_k{{components}}_s{{seed}}.joblib",
        metadata=f"{NOM_CANDIDATE_DIR}/candidate_k{{components}}_s{{seed}}.joblib.json",
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["validation_days"]],
        code=NOM_CODE
    output:
        f"{NOM_CANDIDATE_DIR}/candidate_scores/k{{components}}_s{{seed}}.json"
    resources:
        mem_mb=1024
    shell:
        NOM_COMMAND + "score --components {wildcards.components} --seed {wildcards.seed}"

rule summarize_scene_candidates:
    input:
        candidates=NOM_CANDIDATES,
        candidate_metadata=[path + ".json" for path in NOM_CANDIDATES],
        scores=NOM_SCORES,
        code=NOM_CODE
    output:
        NOM_CANDIDATE_SUMMARY
    resources:
        mem_mb=512
    shell:
        NOM_COMMAND + "candidates"

rule fit_abi_scene_adm:
    input:
        selection=NOM_CANDIDATE_SUMMARY,
        candidates=lambda wc: f"{NOM_CANDIDATE_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib",
        metadata=lambda wc: f"{NOM_CANDIDATE_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib.json",
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["training_days"]],
        abi=[preprocessed(day)
             for day in NOMINAL["training_days"]],
        code=NOM_CODE
    output:
        f"{NOM_ADM_DIR}/adm_k{{components}}.joblib",
        f"{NOM_ADM_DIR}/adm_k{{components}}.joblib.json"
    resources:
        mem_mb=3072,
        nominal_io=1
    shell:
        NOM_COMMAND + "adm --components {wildcards.components}"

# Model-choice variant for the sensitivity study: quadratic angular form,
# same held-in days, scene taxonomy and paired-ratio objective.

rule fit_abi_quadratic_adm:
    input:
        selection=NOM_CANDIDATE_SUMMARY,
        candidates=lambda wc: f"{NOM_CANDIDATE_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib",
        metadata=lambda wc: f"{NOM_CANDIDATE_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib.json",
        caches=[f"{NOMINAL['cache_root']}/{day}" for day in NOMINAL["training_days"]],
        abi=[preprocessed(day)
             for day in NOMINAL["training_days"]],
        code=NOM_CODE
    output:
        f"{NOM_ADM_DIR}/adm_k{{components}}_quadratic.joblib",
        f"{NOM_ADM_DIR}/adm_k{{components}}_quadratic.joblib.json"
    resources:
        mem_mb=3072,
        nominal_io=1
    shell:
        NOM_COMMAND + "adm --components {wildcards.components} --form quadratic"

# Single-day in-sample refits: between-day ADM variability (proxy summary)
# and the per-day ADM sensitivity variant.

rule fit_abi_daily_adm:
    input:
        selection=NOM_CANDIDATE_SUMMARY,
        candidates=lambda wc: f"{NOM_CANDIDATE_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib",
        metadata=lambda wc: f"{NOM_CANDIDATE_DIR}/candidate_k{wc.components}_s{NOM_GMM_SEED}.joblib.json",
        cache=f"{NOMINAL['cache_root']}/{{day}}",
        abi=lambda wc: preprocessed(wc.day),
        code=NOM_CODE
    output:
        f"{NOM_ADM_DIR}/daily/adm_k{{components}}_day{{day}}.joblib",
        f"{NOM_ADM_DIR}/daily/adm_k{{components}}_day{{day}}.joblib.json"
    resources:
        mem_mb=2048,
        nominal_io=1
    shell:
        NOM_COMMAND + "adm --components {wildcards.components} --day {wildcards.day}"

rule promote_scene_model:
    input:
        caches=NOM_CACHE,
        candidates=NOM_CANDIDATES,
        candidate_metadata=[path + ".json" for path in NOM_CANDIDATES],
        scores=NOM_SCORES,
        # Full mode promotes the reviewed K; the gate must pass first.
        summary=[NOM_CANDIDATE_SUMMARY, SCENE_TAXONOMY_GATE],
        code=NOM_CODE
    output:
        NOM_MODEL,
        NOM_MODEL + ".json",
        f"{NOMINAL['diagnostic_dir']}/gmm_selection.json"
    resources:
        mem_mb=2048
    shell:
        NOM_COMMAND + "train"
