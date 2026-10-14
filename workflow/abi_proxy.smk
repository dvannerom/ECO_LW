rule abi_scene_id_development:
    input:
        model=MODEL,
        scene_ids=SCENE_ID_OUTPUTS

rule abi_proxy_adm_corrected_narrowband_inputs:
    input:
        narrowbands=NARROWBAND_OUTPUTS

rule all_goes_proxy:
    input:
        ABI_PRODUCTION_OUTPUTS

rule preprocess:
    input:
        code=["scripts/preprocess_data_ABI.py", "src/netcdf_io.py", "src/goes_navigation.py",
              "src/geospatial.py", "src/radiometry.py"]
    output:
        PREPROCESSED_ROOT + "/abi_{day}_res" + str(RESOLUTION) + "km_step" + str(STEP) + ".nc"
    params:
        step=STEP,
        resolution=RESOLUTION,
        lambda_center=float(config["lambda_center"]),
        raw_sources=json.dumps(config["raw_abi"], sort_keys=True)
    shell:
        "PYTHONPATH=src python scripts/preprocess_data_ABI.py --day {wildcards.day} "
        "--step {params.step} --res_km {params.resolution} "
        "--lambda_center {params.lambda_center} --output {output:q} "
        "--raw-sources-json {params.raw_sources:q}"

rule classify_scene:
    input:
        data=lambda wildcards: preprocessed(wildcards.day),
        model=MODEL,
        gate=SCENE_TAXONOMY_GATE,
        code=["scripts/scene_id_pca.py", "src/scene_features.py"]
    output:
        SCENE_ROOT + "/scene_id_{day}_res" + str(RESOLUTION) + "km_" + str(SCENE_COMPONENTS) + "comp.nc"
    params:
        lambda_center=float(config["lambda_center"])
    shell:
        "PYTHONPATH=src python scripts/scene_id_pca.py --input_file {input.data} "
        "--model {input.model:q} --gate {input.gate:q} --lambda_center {params.lambda_center} "
        "--output {output:q}"

rule plot_proxy_scene_id:
    input:
        scene=lambda wildcards: scene_id(SCENE_ID_DIAGNOSTIC_DAY),
        plotter="plotting/plot_scene_id.py"
    output:
        SCENE_ID_DIAGNOSTIC_FIGURE
    params:
        lambda_center=float(config["lambda_center"])
    shell:
        "python {input.plotter} --input {input.scene} "
        "--lambda-center {params.lambda_center}"

rule radiance_to_flux:
    input:
        data=lambda wildcards: preprocessed(wildcards.day),
        scene=lambda wildcards: scene_id(wildcards.day),
        library=PRODUCTION_ADM,
        library_metadata=PRODUCTION_ADM + ".json",
        gate=SCENE_TAXONOMY_GATE,
        code=PRODUCTION_CODE
    output:
        narrowbands=expand(
            NARROWBAND_ROOT + "/narrowband_flux_{{day}}_res" + str(RESOLUTION) + "km_C{channel}.nc",
            channel=range(6),
        )
    params:
        resolution=RESOLUTION,
        n_components=SCENE_COMPONENTS,
        max_vza=PRODUCTION_MAX_VZA_DEG,
        lambda_center=float(config["lambda_center"])
    resources:
        mem_mb=4096
    shell:
        "PYTHONPATH=src python scripts/radiance_to_flux.py --day {wildcards.day} "
        "--resolution {params.resolution} --n-components {params.n_components} "
        "--adm-library {input.library:q} --max-vza {params.max_vza} "
        "--lambda_center {params.lambda_center} --input {input.data:q} "
        "--scene-input {input.scene:q} --outputs {output.narrowbands:q}"

rule fit_channel_power_law:
    input:
        filters=GOES_FILTER_DIR,
        code=["scripts/fit_irradiance.py", "src/channel_emulation.py", "src/spectral_response.py"]
    output:
        CHANNEL_POWER_LAW_FILE
    params:
        filter_dir=GOES_FILTER_DIR
    shell:
        "PYTHONPATH=src python scripts/fit_irradiance.py "
        "--filter-dir {params.filter_dir} --output {output}"

rule fit_narrowband_to_broadband_coeffs:
    input:
        sunny=SUNNY_DIR,
        filters=GOES_FILTER_DIR,
        power_law=CHANNEL_POWER_LAW_FILE,
        code=["scripts/compute_temperature_SBDART.py", "src/channel_emulation.py",
              "src/spectral_response.py"]
    output:
        NARROWBAND_TO_BROADBAND_COEFFS_FILE
    params:
        sunny_dir=SUNNY_DIR,
        filter_dir=GOES_FILTER_DIR
    shell:
        "PYTHONPATH=src python scripts/compute_temperature_SBDART.py "
        "--sunny-dir {params.sunny_dir} --filter-dir {params.filter_dir} "
        "--power-law-file {input.power_law} --output {output}"

rule narrowband_to_broadband:
    input:
        narrowbands=expand(
            NARROWBAND_ROOT + "/narrowband_flux_{{day}}_res" + str(RESOLUTION) + "km_C{channel}.nc",
            channel=range(6),
        ),
        coefficients=NARROWBAND_TO_BROADBAND_COEFFS_FILE,
        code=PRODUCTION_CODE
    output:
        BROADBAND_ROOT + "/broadband_flux_{day}_res" + str(RESOLUTION) + "km.nc"
    params:
        resolution=RESOLUTION,
        lambda_center=float(config["lambda_center"])
    shell:
        "PYTHONPATH=src python scripts/narrowband_to_broadband.py --day {wildcards.day} "
        "--resolution {params.resolution} --lambda_center {params.lambda_center} "
        "--coefficients-file {input.coefficients:q} --form cubic "
        "--inputs {input.narrowbands:q} --output {output:q}"

rule monthly_flux:
    input:
        broadband=BROADBAND_OUTPUTS,
        reference=lambda wildcards: preprocessed(DAYS[0])
    output:
        config["output"]["monthly"]
    params:
        resolution=RESOLUTION,
        lambda_center=float(config["lambda_center"])
    shell:
        "PYTHONPATH=src python scripts/monthly_flux.py --inputs {input.broadband} "
        "--reference-data {input.reference} --output {output} "
        "--resolution {params.resolution} --lambda-center {params.lambda_center}"
