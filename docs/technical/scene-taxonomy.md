# Scene taxonomy

The GOES production chain and nominal ECO/Sunny budget share one reviewed,
spectral-only GMM taxonomy. It uses six channel brightness temperatures and
four brightness-temperature differences. The selected component count, K, is a
manual, versioned choice; diagnostics inform review but do not select K
automatically.

The active selection and diagnostics identity are recorded in
[`config/scene_taxonomy.yaml`](../../config/scene_taxonomy.yaml). The
K-independent full-population diagnostics use
[`config/gmm_component_selection.yaml`](../../config/gmm_component_selection.yaml);
the normal production profile is
[`config/nominal_production.yaml`](../../config/nominal_production.yaml).
K-dependent workflow rules check the taxonomy gate and stop if the selection
is missing or no longer matches the reviewed diagnostics.

## Review procedure

1. Generate the full-population diagnostics:

   ```bash
   snakemake -s workflow/Snakefile --configfile config.yaml --cores N \
       gmm_selection_diagnostics \
       --config nominal_budget_settings=config/gmm_component_selection.yaml
   ```

2. Review the table and figures under
   `figures/gmm_component_selection/diagnostics/`, then print the summary and
   diagnostics hash:

   ```bash
   PYTHONPATH=src python scripts/check_scene_taxonomy.py --summary
   ```

3. Record K, its rationale, reviewer, date, and matching diagnostics hash in
   `config/scene_taxonomy.yaml`.
4. Rerun affected K-dependent products through Snakemake so scene labels and
   downstream products use the reviewed model.

The component-selection figure tools are listed in the
[plotting guide](plotting.md). The authoritative mission and scientific
context remains in [mission_context.md](../mission_context.md).
