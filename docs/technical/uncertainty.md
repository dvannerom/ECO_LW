# Uncertainty assessment

The repository reports evidence-backed components and conditional subtotals;
it does not establish total ECO mission uncertainty or compliance. Mission
requirements and transfer assumptions belong in the
[mission context](../mission_context.md). The
[SRL4 readiness assessment](../srl4_lw_readiness_assessment.md) records scope,
evidence, and remaining gaps.

## Current workflow products

The integrated workflow currently requests these distinct evidence products:

- **Nominal ECO/Sunny budget** — a source-based estimate using grouped
  held-out Sunny spectra, ABI-population scene weighting, and the configured
  ECO channel scenario. It is conditional on the stated simulation, transfer,
  and independence assumptions. The active settings are selected through
  `nominal_budget_settings` in [`workflow/Snakefile`](../../workflow/Snakefile);
  the production and full-diagnostics profiles are
  [`config/nominal_budget_production.yaml`](../../config/nominal_budget_production.yaml)
  and [`config/nominal_budget_full.yaml`](../../config/nominal_budget_full.yaml).
- **Production-aligned sensitivity** — paired, one-at-a-time changes to
  production choices. These are sensitivity results, not independent
  uncertainties to add to the nominal budget. The workflow implementation and
  settings are in
  [`workflow/uncertainty_sensitivity.smk`](../../workflow/uncertainty_sensitivity.smk),
  [`config/sensitivity_run.yaml`](../../config/sensitivity_run.yaml), and
  [`config/sensitivity_extensions.yaml`](../../config/sensitivity_extensions.yaml).
- **ECO spectral/angular assessment** — separate Sunny simulation evidence for
  spectral reconstruction and ADM retrieval. Interpret its metrics using their
  stage labels and configured channel scenario; do not treat this assessment
  as interchangeable with the nominal budget.
- **Evidence registry and gap report** — provenance-aware inventory of
  estimates, evidence freshness, and unquantified terms. The registry does not
  make missing terms zero or propagate every term through every ABI product.

Run the integrated assessment with:

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores 8 \
    uncertainty_assessment
```

The full `all` target also requests these reports. To request only the nominal
budget, use:

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores 2 \
    uncertainty_nominal_budget
```

## Interpretation

- Keep ABI proxy measurements, Sunny simulation results, and ECO-transfer
  assumptions distinguishable.
- A sensitivity variant describes a change in a model or processing choice; it
  is not an uncertainty-of-record term and is not summed into the nominal
  total.
- Do not infer confidence intervals, independence, or compliance from
  descriptive spreads unless the corresponding method explicitly supports that
  interpretation.
- Treat missing calibration, flight-response, registration, temporal, and
  simulation-representativeness terms as unquantified where the reports say
  they are unquantified.
- Rebuild products through Snakemake after changing code, configuration, or
  source inputs. Consult the report's provenance rather than trusting a saved
  file solely because it exists.

For generated figures and the relevant plotters, see the
[plotting guide](plotting.md).
