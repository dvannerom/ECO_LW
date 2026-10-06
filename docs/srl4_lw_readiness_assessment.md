# ECO LW SRL4 Evidence Assessment

## Scope and conclusion

This note maps the ESA SRL Handbook's SRL4 criteria to evidence in this
repository for **ECO MO2's longwave (LW) radiance-to-broadband-flux data flow**.
It is not an independent Scientific Readiness Assessment (SRA), does not
reassess mission-wide SRL, and does not replace the ECO mission's Phase-0
self-assessment in the Report for Mission Assessment (RfMA), Appendix A,
pp.122-125. The handbook notes that missions with multiple objectives or
instruments may need separate SRL lines (§6, p.11), supporting this narrower
assessment boundary.

SRL4 means **Mission Concept Feasibility Shown**. The handbook is an evidence
and assessment framework, not an algorithm or software-design specification.
It does not, by itself, mandate a change to the current code. The present
workflow supplies useful model and sensitivity evidence, but this repository
alone does not close every SRL4 assessment question. In particular, model
independent review, a formal information-content analysis, and a maintained
scientific risk register are not evidenced here. Existing results are
simulation/proxy evidence, not ECO flight-performance or compliance claims.
The handbook also expects SRA evidence to be traceable, publicly available, and
reproducible (§7, p.24); this note does not establish public availability of
every local input and generated output.

## Sources

- ESA Earth and Mission Science Division, *Scientific Readiness Levels (SRL)
  Handbook*, ESA-EOPSM-SRL-MA-4267, Issue 2, Rev. 1 (Issue 2.1), 25 April 2025;
  local source: [SRL handbook PDF](SRL_Handbook_V2.1_20250425_Issued.pdf).
  SRL4 definition and questions: §6, p.15. SRA guidance: §7, pp.24-25.
- ESA, *Report for Mission Assessment: Earth Explorer 12 Candidate Mission
  ECO*, ESA-EOPSM-ECO-RP-5024, Issue 1.0, 12 June 2026. Mission-level SRL4
  self-assessment: Appendix A, pp.122-125.

## Current evidence update (2026-10-06)

The dated audit below describes the earlier production and noise-free
spectral/angular branches. Subsequent opt-in workflows now execute radiometric
noise, ABI spatial processing, assignment stress, training/evaluation
convergence, and sparse/noisy ECO geometry experiments. The scene-conditioned
nominal budget additionally uses a spectral-only ABI classifier, grouped
held-out Sunny scene ADMs/N2BC, and exact-boundary 4-100 um truth.

The current nominal result is a conditional combined RMSE of 1.9930 W/m2,
with signed bias +0.6423 W/m2, scene-dependent SD 1.8484 W/m2, and random
SD 0.3780 W/m2. Noise-mediated assignment is included under an assumed
ABI-to-ECO response transfer; the ABI spatial-processing increment is combined
under an independence hypothesis. This does not close calibration, flight-SRF,
registration, temporal evolution, independent validation, model-training/library
discrepancy, or mission sampling/stability terms. It is not a complete mission
budget or a compliance assessment.

The [README](../README.md#scene-conditioned-nominal-uncertainty-budget) documents
the current configuration, assumptions, results and explicit target. Small
nominal reports and diagnostics are versioned as evidence snapshots; bulk inputs,
caches, residual arrays and application models remain external/local. The
nominal budget is separate from the default registry and its historical panel.

## Historical workflow purpose audit (2026-10-04)

The project scientist clarified three required deliverables: a complete ABI
narrowband LW radiance-to-broadband-flux demonstration; scoped SRL4 feasibility
evidence for ECO's similar processing; and source-traceable uncertainties that
can be interpreted for ECO, with explicit transfer assumptions and gaps.
ECO's expected advantage from per-pixel multi-view sampling and overlapping
channels is a hypothesis to test, not a conclusion from ABI agreement.

### Current coverage

| Deliverable or term | Implementation/evidence | Remaining limitation or requirement link |
|---|---|---|
| Complete ABI chain | `all_goes_proxy` reaches preprocessing, GMM scene ID, scene ADM fitting, corrected BTs, broadband flux, and aggregation. A dry run found all requested products present. | Default `all` stops ABI at corrected narrowband BTs. Existence of outputs does not validate physical correctness; two rules lack Snakemake provenance. |
| Angular-to-flux conversion | ABI scene fits and Sunny per-scene/channel retrieval use the repository ADM form. | Normalization defect corrected in code with known-profile regression checks; existing products and ObsReq 15 scores require regeneration. |
| Spectral reconstruction | ECO true-band-flux N2BC has grouped held-out predictions, bias, scatter, and RMSE. | Conditional on the Sunny library and assumed SRFs; ObsReq 16 confidence convention is unresolved. ABI coefficient fitting reports training-set error only. |
| ECO multi-view advantage | Sunny supplies per-scene directional radiances; current retrieval uses 15 noise-free views including 55 degrees. | No representative N=1..20 experiment, N=1 prior/fallback, noise propagation, or matched ABI/ECO comparison. ObsReq 12/15 and MeasReq 8. |
| ECO channel advantage | Idealized SRFs include smooth-edge overlap at adjacent band boundaries. | This is not measured instrument overlap or an isolated overlap-benefit experiment; channel covariance is not propagated. MeasReq 8 and ObsReq 16/17. |
| ABI ADM transfer | Summary retains channel/scene/day sources and requirement caveats. | Between-day spread of scene-average fits is not within-scene pixel-level population spread and is not an ECO prior width. ObsReq 15/17. |
| Scene ID, CTH/parallax, co-registration | Proxy algorithms exist. | No quantified ECO flux-error term or validated scene/channel/geometry transfer. ObsReq 12/15/17. |
| Instrument noise/calibration and WFOV anchoring | Mission requirements are recorded. | No propagated ECO measurement covariance or anchoring residual in the chain. MeasReq 8/9, ObsReq 14/15. |
| Monthly/seasonal products and stability | ABI aggregation averages the configured daily samples and saves difference variability. | Not a full diurnal monthly mean or an ECO sampling/stability uncertainty. ObsReq 8/9/10/11. |
| Independent absolute validation | CERES is identified as an evidence source. | No CERES comparison in the inspected DAG. East-West agreement cannot detect shared error or establish absolute accuracy. ObsReq 10/15. |
| Combined uncertainty and gaps | The default workflow now emits `data/uncertainty/eco_uncertainty_registry.json` and `data/uncertainty/eco_uncertainty_gaps.md`. Matched Sunny residuals preserve signed spectral-only and angular-plus-interaction terms and their observed covariance. | Quantification remains conditional on Sunny/SRF/geometry assumptions; existing evidence without producer provenance is labelled unverified. Missing radiometric, scene, ADM-population, calibration, validation, and aggregation terms remain explicit and are not combined into a total. |

### Scientific blockers found

1. **ADM normalization defect: corrected in code; products not regenerated.**
   The former `src/adm.py:radiance_integrand` integrated the angular shape with
   `sin(theta)` but without `cos(theta)`. Both ABI fitting and ECO simulation
   used that integral. Physical upward flux is
   `2*pi*integral[L(theta)*cos(theta)*sin(theta) dtheta]` for an
   azimuth-independent radiance field. A focused check using exact repository
   profiles and the configured 0-70-degree views gives zero error for b=0,
   +2.154% for b=-0.3, and -2.112% for b=+0.3. Thus even a perfect shape fit
   did not recover known flux; this was an implementation error, not physical
   ECO uncertainty. The shared integrand now includes `2*cos(theta)` and the
   existing degree-to-radian conversion, preserving the convention that
   corrected radiance is flux/pi. Three regression tests check isotropic
   normalization and physical flux recovery through both active retrieval
   paths, including multi-scene/channel simulation arrays. Existing ABI ADMs,
   corrected BTs, broadband/aggregate products, and ECO ADM/end-to-end results
   must be regenerated. ABI ADM fitting now tracks the shared source modules
   as workflow inputs; the ECO ADM rule already tracks `src/adm.py`.
2. **Radiometric conventions need an explicit closure check.**
   `fit_irradiance.py:blackbody_spectral_radiance` returns pi times Planck
   radiance despite its radiance name/units. The ABI coefficient fit integrates
   hemispheric spectral flux, so those two pi conventions may compensate;
   neither should be changed alone. Verify the full training-to-operational
   SRF/Planck/BT mapping, including units, rather than infer a net factor-pi
   error from the names. Saved narrowband products contain corrected BTs, not
   physical band fluxes, and ABI outputs are labeled "ECO" in metadata.
3. **Requirement comparisons must use the same quantity.**
   Joined metadata associates the assessment with ObsReq 16, although the
   angular-plus-spectral score belongs under ObsReq 15, with ObsReq 16 reserved
   for spectral-only error. The simulated reference integrates 2.5-500 um,
   consistent with the RfMA study convention (§6.2.1), whereas ObsReq 8 defines
   4-100 um. Document or quantify that mapping; retain both conventions rather
   than silently changing the reference. Report bias separately from 2-sigma
   scatter; the latter excludes bias and is not automatically accuracy.
4. **Saved results are stale and lack producer provenance.**
   The inspected joined, N2BC, and ADM metric files contain nine scenarios
   while the current catalog defines four. They also do not carry producer
   code/input/configuration identities. The registry reports scenario mismatch
   separately from provenance status, hashes each source artifact, and marks
   out-of-catalog terms stale and remaining legacy terms unverified. Workflow
   reruns embed producer identities; do not interpret existing scores as
   regenerated merely because some scenario names match. This assessment did
   not rerun large products.

### Completion priorities

Physical flux normalization is now corrected in code. Verify radiometric
closure, then regenerate a reproducible ABI broadband demonstration and ECO
simulation before interpreting products affected by the ADM correction. Next,
test ECO view-count, noise, SRF and within-scene ADM-population cases on matched
physical scenes. Extend paired covariance to calibrated radiance, scene
classification, co-registration, anchoring and aggregation only when supporting
data exist. Do not invent values, transfer ABI spread universally, or combine
unvalidated proxy and simulation terms in quadrature.

### Uncertainty registry vertical slice (2026-10-04)

The `all` target depends on the registry and gap report. It captures hashes for
small code/configuration files, bounded-cost path/size/mtime inventories for
large input trees, exact hashes for generated metric artifacts, scenario-catalog
comparisons, and producer-provenance status. Input inventory fingerprints do not
read or hash the contents of large simulation trees; this limitation is recorded
in each producer provenance record.

The Sunny residual join pairs spectral-only and end-to-end out-of-fold errors by
scenario, regression, shared scene index and fold. For each full-scene and
clear/cloud subset it computes the angular-plus-interaction increment as the
paired end-to-end residual minus the spectral-only residual, then propagates
their measured 2-by-2 sample covariance. This does not assume independence and
does not separate pure ADM error from its interaction with N2BC. The spectral
term maps to ObsReq 16 and end-to-end to ObsReq 15. The current 15-view,
noise-free simulation and idealized SRFs remain conditional evidence; no
monthly/seasonal propagation or requirement-compliance conclusion is made.

ABI ADM evidence remains a separate proxy statistic: between-day spread of
scene-average fitted parameters. It is not within-scene pixel spread, an ECO
prior, or an ECO flux uncertainty. The report also retains missing noise and
calibration covariance, scene-ID/CTH/parallax/co-registration error, ADM
population/model form, temporal coincidence, WFOV anchoring, spectral-domain
and SRF uncertainty, independent validation, aggregation covariance/sampling,
and stability as unquantified terms. A missing term is never encoded as zero.

## Criterion mapping

| SRL4 question | Evidence in this repository | Status and limitation |
|---|---|---|
| Trace science goals through mission and system requirements. | [Mission context](mission_context.md) links MO2 to RfMA requirements; [error budget](../config/error_budget.yaml) records ObsReq 9-16 and explicitly avoids invented allocations. | Partial for this LW processing slice. The repo tracks relevant product and retrieval requirements, but does not reproduce the mission's complete objective-to-system traceability. |
| Provide a model that computes measurement data from geophysical inputs. | [Workflow](../workflow/Snakefile) and [evaluation implementation](../scripts/evaluate_eco_spectral_reconstruction.py) convolve GERB/Clerbaux spectra with ECO channel scenarios, retrieve narrowband flux with an ADM, and evaluate narrow-to-broadband reconstruction. | Model and runnable stages exist. Idealized passbands and a prescribed 15-view case are not a complete ECO instrument/mission simulator. |
| Establish technical/scientific adequacy and independent review. | The workflow records intermediate metrics and residuals; the [README](../README.md) documents stages and limitations. | Independent scientific review of this repository's implementation is not recorded. The RfMA's statements about review of mission-level models do not automatically review this code. |
| Demonstrate measurement sensitivity to the target parameter. | The simulation scores 4,620 GERB/Clerbaux clear/cloud spectra, grouped by shared index. Nominal and follow-up workflows now propagate NEdT and compare sparse/wide/clustered geometries as well as spectral/angular residuals. | Conditional numerical evidence; the nominal case uses 15 views and follow-up geometries are idealized, not the mission's N=1..20 orbit/noise distribution. The older k=2 comparison convention remains provisional. |
| Analyze information content and identify contributing geophysical parameters. | Channel scenarios and clear/cloud regime scores expose some spectral sensitivity; [scenario definitions](../config/eco_channel_scenarios.yaml) state the modeled bands and assumptions. | A formal information-content/observability analysis and parameter covariance treatment are not documented for this code path. |
| Perform a scientific risk analysis. | [Mission context](mission_context.md) records algorithm and transferability limitations; the [error budget](../config/error_budget.yaml) declares allocations unassigned and registers conditional spectral/angular and ABI-proxy components with explicit gaps. | Useful risk notes exist, but there is no maintained risk register with likelihood/impact, mitigation, evidence, and disposition. |
| Produce demonstration measurement data. | The workflow produces simulated ECO ADM fluxes, stage metrics, cross-validated residuals, and figures under `data/uncertainty/` and `figures/uncertainty/`. | Demonstration outputs exist locally and are reproducible from the configured inputs. They are not ECO observations, and local outputs/input availability should be preserved for an externally reviewable evidence package. |
| Discuss complementary and alternative missions. | The mission context documents GOES ABI as term-specific proxy evidence, CERES as potential independent validation, and configured ECO channel alternatives. | The rationale and alternatives are documented; a concise comparison tied specifically to this LW evidence package would improve the SRL4 technical report. |

## Evidence actions

These are actions to complete or strengthen the scoped evidence package, not
code changes required by the handbook:

1. Prepare an SRL4 technical report that links each claim and result to the
   requirement, source data, configuration, code version, metric convention,
   and reproducible output.
2. Obtain and record independent scientific review of the model, assumptions,
   and results. Local tests and reproducible execution are valuable but are not
   independent review.
3. Maintain a scientific risk register for the LW chain and link each risk to
   a mitigation, uncertainty term, requirement, and evidence status.
4. Document an information-content/sensitivity analysis for the LW channels and
   relevant scene parameters, including assumptions and covariance where
   available.
5. For claims about expected mission performance beyond concept feasibility,
   extend the simulation to representative ECO view-count/geometry and
   instrument-noise cases, then validate the metric convention against the
   requirement. This is a scientific evidence enhancement, not an SRL4-mandated
   code change.
6. Keep a clear roadmap for unresolved items and preserve the demonstration
   inputs and generated outputs needed for independent reproduction.