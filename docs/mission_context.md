# ECO / GOES / CERES mission context

Authoritative record of the mission-level facts this repository's algorithms and
uncertainty analysis depend on.

Each entry is tagged:

- `[RfMA §x p.N]` — from the Report for Mission Assessment; authoritative, traceable.
- `[SRL Handbook §x p.N]` — from ESA's Scientific Readiness Levels Handbook; assessment framework and evidence guidance.
- `[user]` — stated directly by the project scientist; may be more recent than the RfMA.
- `[derived]` — computed from the above; assumptions stated inline.
- `[assumed]` — not yet confirmed; check before relying on it.

Keep this file updated whenever new mission information arrives.

## ABI nominal assignment noise update

- [user 2026-10-07] Replace additive Gaussian noise with mean-preserving
  lognormal radiance noise in the nominal ABI assignment assessment, and
  document the change.
- [derived 2026-10-07] Sampled and full ABI assignment now use
  `L'=exp(log(L)+sqrt(v)*Z-v/2)`, with
  `v=log(1+(sigma_L/L)^2)` and independent standard-normal `Z`.
  This preserves clean-radiance mean and the existing radiance-space variance;
  `sigma_L` remains NEdT times the satellite/channel Planck derivative at 255 K.
  Positive clean radiances remain mathematically positive without clipping or
  rejecting draws. Zero SD is an exact identity; numerical range failures and
  invalid BTs remain explicit errors.
- [assumed 2026-10-07] Lognormal skewness, particularly its positive tail for
  dim channels, is a positivity-preserving modelling choice, not an
  instrument-validated distribution or an RfMA requirement. This update
  neither removes existing ABI noise nor establishes the ABI-to-ECO transfer.
  The separate ECO/Sunny radiometric-noise model remains Gaussian. Earlier
  Gaussian ABI assignment and propagated budget scores must be regenerated
  before being interpreted as lognormal-scenario evidence.

## 0. Source document

- ESA (2026), *Report for Mission Assessment: Earth Explorer 12 Candidate
  Mission ECO*, ESA-EOPSM-ECO-RP-5024, Issue 1.0, 12/06/2026, 135 pp.
  DOI 10.5281/zenodo.20543209. Public since the EE12 User Consultation Meeting.
- ESA Earth and Mission Science Division (2025), *Scientific Readiness Levels
  (SRL) Handbook*, ESA-EOPSM-SRL-MA-4267, Issue 2, Rev. 1 (Issue 2.1),
  25/04/2025. Local copy: `docs/SRL_Handbook_V2.1_20250425_Issued.pdf`.
- Local copy `docs/ECO_Report_for_Mission_Assessment.pdf` (git-ignored) plus an
  extracted-text sidecar `.pdf.txt` with `===== PAGE n =====` markers
  (regenerate with `pymupdf`, installed in the `tf-gpu` env).
- ECO = **Earth Climate Observatory**, an Earth Explorer **12** candidate (one
  of four: CryoRad, ECO, Hydroterra+, Keystone). Phase 0 complete.
- Note: the repo owner (David Vannerom, ROB) is credited in the RfMA
  acknowledgements, and §6.2.4 describes *this repository's* GOES ABI work.

## 1. Mission objectives

- [RfMA §3.2] **MO1**: observe global mean absolute Earth Energy Imbalance (EEI)
  on annual to seasonal timescales.
- [RfMA §3.2] **MO2**: global high-resolution mapping of outgoing short- and
  longwave fluxes (RSR / OLR). **<- this repository serves MO2, LW side.**
- [RfMA Exec. Summary p.5] Concept slogans: "the constellation is the
  instrument" (sampling) and "the viewing geometry is the instrument"
  (multi-angle radiance-to-flux).
- Mission duration: 5 yr nominal + 5 yr extended, after 12-month commissioning;
  single Vega-C launch, stacked configuration. [RfMA §7.2.2 p.75]

## 1a. ECO_LW evaluation framework

- [user 2026-10-04] ECO_LW must implement the full ABI narrowband LW
  radiance-to-broadband-flux workflow as a processing demonstration contributing
  to the scoped SRL4 feasibility case for ECO's similar LW processing design.
  The complete ABI chain is a core deliverable, not merely an optional
  comparator to the ECO simulation branch.
- [user 2026-10-04] ECO's per-pixel multi-view observations and overlapping
  channels are expected to improve on ABI. This is a hypothesis to assess with
  explicit ECO geometry and channel-response scenarios, not demonstrated by
  ABI consistency alone. The stated overlap must be reconciled with the
  current idealized channel scenarios before claiming a design advantage.
- [user 2026-10-04] Wherever possible, derive uncertainties applicable to ECO
  directly or through justified extrapolation, rescaling, or other transfer.
  Each estimate must retain its source, transfer assumptions, and RfMA
  requirement traceability. Explicitly identify missing terms and stages where
  the workflow does not yield an ECO-interpretable uncertainty.
- [user 2026-10-01] The purpose of ECO_LW is to evaluate the uncertainty of
  **ECO's radiance-to-broadband-flux processing**, with uncertainty decomposition
  and traceability to mission requirements. ABI data are a proxy where useful;
  GERB/Clerbaux SBDART spectra are simulation evidence where useful; CERES is
  independent validation evidence where useful.
- [user 2026-10-01] **No ECO flight observations are available.** This is why
  the study uses GOES ABI proxy data and SBDART/GERB simulation spectra; together
  they are the intended evidence base for estimating expected ECO performance,
  not temporary substitutes for an available ECO dataset.
- [user 2026-10-01] The workflow must not choose one sensor definition or dataset
  instead of another (e.g. GOES channels *or* ECO channels). Keep the target
  fixed as ECO and use each source for the uncertainty terms it can inform:
  - ECO requirements, channel responses, geometry, and instrument performance
    define the target configuration and its scenarios.
  - ABI observations constrain empirical scene/space/time behavior and test
    transferable algorithm behavior, with proxy limitations recorded.
  - GERB/Clerbaux spectra, convolved with ECO response functions, constrain
    spectral reconstruction and its regime dependence.
  - CERES observations/ADMs provide independent validation for applicable
    broadband and angular-retrieval terms.
- [user 2026-10-01] The main Snakefile workflow should orchestrate multiple
  complementary stages: use ABI proxy data where they help develop or test
  algorithms such as scene identification, and use ECO instrument/channel
  parameters with simulation inputs where those are the relevant basis, such as
  narrowband-to-broadband reconstruction. It should not be only an ECO spectral
  assessment, nor should ABI's production chain stand in for ECO performance.
- [derived 2026-10-01] The default DAG follows two evidence branches:
  (1) ABI proxy preprocessing -> GMM scene ID -> scene ADMs -> ABI narrowband
  brightness-temperature inputs for GOES-specific N2BC, and (2) GERB/Clerbaux
  spectra that branch into two independent ECO analyses: (2a) integrate true
  ECO-band fluxes and fit grouped N2BC coefficients; (2b) convolve directional
  radiances and retrieve narrowband flux through per-scene/channel ADM fits.
  The join applies fold-specific N2BC models, trained on true band flux, to the
  held-out ADM-retrieved flux, separating spectral-only from end-to-end error.
  The ADM simulation currently uses 15 noise-free views from 0-70 deg (Sunny
  provides 5-degree angles); this is an idealized high-information case, not the
  mission's full N=1..20 geometry/noise distribution.
  These branches remain separate at the N2BC boundary: ABI channel fluxes are
  not ECO channel fluxes, so ECO coefficients must not be applied to ABI data.
  The complete ABI-channel N2BC/monthly chain remains available as the explicit
  `all_goes_proxy` comparator. The default DAG does not claim a combined
  per-pixel ECO broadband product or total propagated uncertainty yet.
- [derived] This is a **target-centric, evidence-fused assessment**, not a
  choice between GOES and ECO processing pipelines. Estimates from different
  sources must not be blindly averaged: retain each source, uncertainty term,
  transfer assumption, validation domain, and requirement link; combine only
  evidence that estimates the same physical quantity with defensible covariance.
- [derived] Channel alternatives are ECO design scenarios, not a GOES-versus-ECO
  switch. Evaluate the RfMA baseline and any explicitly configured ECO
  alternatives separately, reporting uncertainty conditional on each scenario.

## 1b. Scientific Readiness Level scope

- [SRL Handbook §6 p.11] A mission with multiple science objectives or
  instruments may need more than one SRL line; detailed questions should
  address mission-specific scientific risks, and any question treated as
  irrelevant should be justified.
- [SRL Handbook §6 p.15] SRL 4 is **Mission Concept Feasibility Shown**. Its
  evidence concerns traceable objectives and requirements, an adequate model,
  demonstrated measurement sensitivity, information content, scientific risks,
  demonstration data, and comparison with complementary or alternative
  missions. The handbook calls for an SRL-4 technical report and a clear
  follow-on roadmap; independent review and reproducibility are addressed in
  its assessment guidance (§7 pp.24-25).
- [RfMA Appendix A pp.122-125] ECO's Phase-0 self-assessment says the mission
  is at least SRL 4 for MO1 and MO2, and answers the SRL-4 questions at mission
  level. ECO_LW contributes evidence only for the **MO2 longwave
  radiance-to-broadband-flux data flow**; it is not a separate assessment of
  mission-wide SRL and does not supersede the RfMA assessment.
- [derived] The repository's simulation, proxy-data processing, requirement
  references, and demonstration outputs are relevant evidence for that scoped
  contribution. The current idealized noise-free 15-view simulation is not a
  full instrument/mission performance simulation, and local code results do
  not by themselves establish independent review, complete information-content
  analysis, or a formally maintained scientific risk register. See
  [`srl4_lw_readiness_assessment.md`](srl4_lw_readiness_assessment.md) for the
  criterion-by-criterion evidence and limitations.

## 2. Requirements (the anchor for any error budget)

Requirements are given as Goal (G) / Threshold (T); k=1 or k=2 noted where the
RfMA states it.

### EEI (MO1)

| Req | Content |
|---|---|
| ObsReq 1 | EEI accuracy **0.3 (G) – 0.7 (T) W/m², k=2**, annual (T) to seasonal (G) |
| ObsReq 2 | EEI stability **0.1 (G) – 0.2 (T) W/m²/decade** |
| ObsReq 6 | EEI error allocation (W/m², G/T): Sun ISR sampling 0.05/0.10; Earth TOR sampling 0.10/0.20; Earth-Sun relative accuracy 0.20/0.40; twilight beam sampling 0.20/0.40 |
| ObsReq 7 | Regional TOR sampling error (5°x5°, continental, hemispheric asymmetry) within **4 W/m²** annually |

### RSR / OLR flux maps (MO2) — the requirements this repo must meet

| Req | Content |
|---|---|
| ObsReq 8 | OLR = broadband **4–100 µm**; RSR = 0.3–4 µm; averaged monthly (G) to seasonal (T) |
| ObsReq 9 | **Average** flux spatial resolution **10 km (G) – 50 km (T)** |
| ObsReq 10 | **Average OLR accuracy 0.3 % (G) – 1.25 % (T)**; at 240 W/m² ref => ~0.5 (G) / 2 (T) W/m². (RSR 0.5/2.12 %) |
| ObsReq 11 | Average RSR/OLR flux **stability better than 0.3 %** |
| ObsReq 15 | **Instantaneous** OLR at 10 km (G) – 50 km (T): **1.42 % (G) – 5.83 % (T)** => **2.4 (G) / 9.9 (T) W/m²** at 240 W/m². (RSR 2.4–9.9 % => 4.8/19.8 W/m² at 200) |
| ObsReq 16 | **Narrow-to-broadband reconstruction accuracy: 0.84 % for OLR** (2.7 % RSR). At 240 W/m² => **~2.0 W/m²**. <- the budget for `narrowband_to_broadband.py` |
| ObsReq 12 | Multi-angular views "sufficiently close in time that they can be statistically considered simultaneous with respect to the scene dynamics" |
| ObsReq 13/14 | Cameras share the radiometer's WFOV; ERB products cross-calibrated to (anchored on) the TOR/EEI observations |
| ObsReq 17 | Spectral+spatial information sufficient for scene identification AND for co-registration of multi-angular views |

### Instrument / measurement

| Req | Content |
|---|---|
| MeasReq 1 | Constellation: **minimum 2 satellites**, 82° inclination, planes equally separated in RAAN, near-circular, same inclination/arg-perigee/altitude |
| MeasReq 2 | FoV embraces entire visible Earth incl. atmosphere to **100 km** (the "control surface"); control-surface altitude stable to **±0.5 km (G) / ±1 km (T)** between observations at the same latitude |
| MeasReq 3 | TOR cadence **1 min** (~100 points per orbit); **multispectral cadence >= 30 s**; TSI >= 2/day |
| MeasReq 4/5 | Radiometers 0.2–100 µm, dynamic range 0–1500 W/m² |
| MeasReq 6 | TOR/TSI absolute uncertainty **< 1 W/m² (2σ)** |
| MeasReq 7 | TOR/TSI stability **0.01 (G) / 0.1 (T) W/m²** over lifetime |
| MeasReq 8 | Channel definitions + sensitivity, Table 4.1 (see §4 below) |
| MeasReq 9 | Camera absolute radiometric accuracy **~10 % (2σ)** — deliberately loose, relies on radiometer anchoring |
| MeasReq 10 | Spatial resolution **1 km (G) – 5 km (T) VIS**; **2 km (G) – 10 km (T) SWIR and TIR** |
| MeasReq 11 | Imaging FoV cut-off **55°** (at 800 km => ~2800 km diameter central area); outer ring used for budget closure / intercalibration only |

## 3. Orbit and constellation

- [RfMA Table 7.1 p.75] Two competing concepts, both 2 satellites, 90° RAAN
  separation, frozen eccentricity, argument of perigee 90°:

  | | Concept A | Concept B |
  |---|---|---|
  | Orbit altitude | **600 km** | **800 km** |
  | Inclination | **82.2°** | **81.42°** |
  | RAAN separation | 90° | 90° |
  | Diurnal cycles in 1 year | **8** | **8** |

- [user] 600 km / 82° / 90° RAAN — i.e. **Concept A**.
- [RfMA §4.3.1 p.40] Inclination is set by the need for an **integer number of
  cycles per year** (year-on-year ground-track repeatability, so interannual EEI
  comparisons are not corrupted by differing diurnal/seasonal sampling).
  ~82° gives **4 full diurnal cycles per year per satellite**; the 2-satellite
  constellation gives **8 per year** => full diurnal coverage in **~46 days**.
  Residual polar gap at 82° is **1 % of Earth's surface**.
- [derived, confirmed by RfMA] Nodal precession
  dOmega/dt = -(3/2) J2 (Re/a)^2 n cos(i) ~= -1.01 deg/day at 82.2°/600 km;
  local-time drift ~= -2.0 deg/day = -0.13 h/day => ~91 days per satellite for
  full diurnal coverage. Consistent with the RfMA's "4 diurnal cycles per year".
- [RfMA MeasReq 11] Each location gets **at least 4 diurnal measurements per
  day** with the 2-satellite configuration.
- [derived] Diurnal aliasing remains a real systematic for *monthly* products
  (a month covers only ~2/3 of the diurnal cycle) and the sampled local times
  drift month to month, so the residual bias oscillates. This matters much less
  at the annual scale the mission is designed around. Second-order trap: the
  spacecraft thermal environment cycles with the precession, so calibration
  drift is correlated with local time and is partly degenerate with the diurnal
  signal.
- Orbit maintenance: in-plane manoeuvres every ~8 days (Concept A, 100 m control
  band) or ~20 days (Concept B, 150 m). [RfMA §7.4.3]

## 4. Instruments

### Radiometers (MO1)

- Earth-viewing **WFOV** radiometer, limb-to-limb, 0.2–100 µm. A single
  observation is a circular footprint **~6000 km diameter** — effectively the
  whole visible disk. Measures a **true flux at satellite altitude**, with **no
  ADM needed**. [RfMA §4.2.1 p.33]
- Separate **NFOV Sun-looking** radiometer (changed during Phase 0 from the
  original interchangeable flip-calibration concept). Intercalibration now by
  design + pre-flight characterisation, with occasional in-flight Sun/deep-space
  pointing of the Earth radiometer for verification. [RfMA §4.4, §6.6]
- Radiometer operates in **quasi-continuous acquisition** (no shutter cycling).
- [RfMA Tables 7.7/7.8 p.90] Predicted Level-1 performance:

  | Metric | Requirement | Earth WFOV | Sun NFOV |
  |---|---|---|---|
  | Absolute accuracy [W/m²] | < 1 | **0.37** | 0.15 |
  | Long-term stability [W/m²] | < 0.1 (T) / 0.05 (G) | **0.04** | 0.07 |
  | Sun-Earth relative [W/m²] | < 0.3 (T) / 0.1 (G) | **0.18** | |

### Cameras

- Suite = **3 VIS (RGB) + 1 SWIR + 6 TIR** channels, co-aligned with the
  radiometers and sharing the same WFOV.
- [RfMA Table 4.1 p.43] **TIR channel definition — the authoritative ECO LW
  channel set. [user 2026-09-30] ADOPTED AS THE BASELINE for this repository**
  (the commented/uncommented state of `lw_spectral_responses_tmp.py` as supplied
  was arbitrary and is not a statement of the current design):

  | # | Name | λmin (µm) | λmax (µm) | Tmin | Tref | Tmax | NEdT@Tref |
  |---|---|---|---|---|---|---|---|
  | 1 | Water Vapour | 5 (G); 6 (T) | 8 | 180 | 255 | 280 | 0.4 (G); 0.7 (T) |
  | 2 | Cloud Phase | 8 | 9.1 | 180 | 255 | 350 | 0.4 |
  | 3 | Ozone | 9.1 | 10.3 | 180 | 255 | 310 | 0.4 |
  | 4 | Cloud Temperature | 10.3 | 11.4 | 180 | 255 | 350 | 0.4 |
  | 5 | Cloud Optical Depth | 11.4 | 12.5 | 180 | 255 | 350 | 0.4 |
  | 6 | Carbon Dioxide | 12.5 | 16 (G); 14 (T) | 180 | 255 | 290 | 0.4 (G); 0.6 (T) |

  => **NEdT = 0.4 K at 255 K** answers the previously-open noise question.
  Goal spectral coverage is **5–16 µm**, threshold 6–14 µm.
- [RfMA §7.4.2] Achieved performance: TIR spatial resolution compliant for
  Concept A across the FoV; Concept B reaches 11.1–11.8 km at the 55° edge
  (vs 10 km requirement). NEdT: Concept B non-compliant for bands 4 and 5 at the
  edge (0.41 / 0.45 K) — judged non-critical because **radiance-to-flux error
  propagation is smallest for 55°-viewing pixels**.
- TIR absolute radiometric accuracy achieved: **10.3–12 % (2σ)** Concept A
  (slightly over the ~10 % requirement, mitigated by radiometer anchoring);
  inter-pixel / inter-band relative accuracy ~**2 %**.
- **Flux retrieval is limited to VZA <= 70°**; the majority of ECO camera views
  fall in **20°–60°**. [RfMA §6.2.4, §6.2.6]

## 5. Processing chain and product levels

[RfMA §6.2, Fig. 6.7 p.61] — this repo implements the LW branch of L1b -> L2b -> L2c:

- **L1b**: instantaneous narrow-band camera radiances (multi-angle).
- **L2a**: broadband *radiances*, via narrow-to-broadband conversion of L1b.
- **L2b**: narrow-band *fluxes*, via scene-dependent **narrow-band ADMs**.
- **L2c**: broadband *fluxes*, via narrow-to-broadband conversion of L2b.
- **L3**: mapped monthly/seasonal RSR and OLR fluxes.
- Broadband ADMs serve the calibration transfer from the WFOV TOR.
- The RfMA states the ECO studies focus on **L2b and L2c** as the novel parts,
  and that these "place the strongest constraints on instantaneous pixel scale
  errors" — i.e. exactly what this repository computes.

### Cross-calibration strategy (constrains what the radiometer can fix)

- [RfMA §6.6 p.71] "The L1b TOR flux measured by the Earth viewing radiometers
  should equal the integral of the L2a OLR plus RSR radiances observed by the
  multispectral camera suite." **"The strategy will not calibrate the imagery at
  the pixel level, but utilises the high accuracy and stability of the
  radiometer to monitor the integral accuracy of the cameras."**
- => the radiometer anchors only the **disk-integrated** scale. Scene- and
  regime-dependent biases that cancel over the disk are invisible to it. A
  synthetic-WFOV forward model built from a high-resolution flux field is
  therefore a faithful proxy for this strategy, and yields per error source the
  fraction that survives disk integration and can be corrected.
- LW-only cross-calibration is possible in **night scenes**.
- Also planned: vicarious calibration on deserts, Antarctic ice, the Moon, and
  **deep convective cloud cores**; GSICS framework. Validated in the **MAGIC
  Avalon campaign** [RfMA §6.4].

## 6. ADM retrieval regime (differs fundamentally from the GOES proxy)

- [RfMA §6.2.3] LW ADMs assume no dependence on solar zenith or relative
  azimuth: `F = pi L(theta_v) / R(theta_v)` — a pure limb-darkening function.
  Confirms the approach used in `src/adm.py`.
- [RfMA §6.2.4] The RfMA writes the limb-darkening function as
  **`L(theta_v) = b (1 + a ln cos theta)`**, citing Elmer et al. (2016), with
  `a` the *shape* parameter and `b` the *normalisation*.
  **The repo uses a genuinely DIFFERENT form** — `src/adm.py:elmer()` is
  `ln(cos theta + 2) / ln(cos 55 deg + 2)` and `radiance_linear` is
  `1 + b (elmer(theta) - 1)`, normalised to unity at 55 deg — and it swaps the
  parameter names (repo `b` is the shape parameter, RfMA `a` is).
- [derived 2026-09-30] **The two forms are NOT mathematically equivalent** —
  verified numerically. Fitting the RfMA form to the repo form over 0-70 deg
  leaves residuals of:

  | repo b | best-fit RfMA a | max dev | RMS |
  |---|---|---|---|
  | -0.60 | -0.1817 | 2.6e-2 | 6.3e-3 |
  | -0.30 | -0.0865 | 1.3e-2 | 3.1e-3 |
  | -0.10 | -0.0279 | 4.3e-3 | 1.0e-3 |

  i.e. ~1.3 % shape difference at typical b => **~3 W/m² at 240 W/m²**, which is
  NOT negligible against ObsReq 15 (2.4 W/m² goal).
  The repo's `ln(cos theta + 2)` is a **regularised variant** that stays finite
  at the limb (1.077 at 89 deg) where the true Elmer `ln cos theta` diverges.
  [user 2026-09-30] **Keep the repo convention**, but it must be documented as a
  deliberate variant, not as Elmer et al. (2016), and the difference between the
  two must be carried as one of the quantified ADM model-form error terms. ECO
  restricts flux retrieval to VZA <= 70 deg, where the regularisation is a
  modest effect.
- [user] ECO fits ADMs **per pixel**, on the same pixel seen from **N views**,
  N = 1 to ~20, **quasi-simultaneous within one overpass**.
- [RfMA §4.2.2 p.38] "ten or more multi-angle radiance to flux conversions of
  the same scene will bring a significant reduction of the angular conversion
  error".
- [RfMA §6.2.3] Baseline is a **static ADM set** built from a limited training
  period; **dynamic, frequently updated ADMs** are a Phase A option
  (Gristey et al. 2023: WFOV sampling populates the angular distribution in
  days-to-weeks rather than the months-to-years of scanning instruments).
- [RfMA §6.2.6 p.66] **Angular weighting is the key lever**, driven by
  *sensitivity to scene misclassification*, not by parameter leverage: using
  CERES TRMM ADMs, for a 56–64.9 K cloud-surface contrast case, the LW flux
  sensitivity to cloud emissivity is **6.6 W/m² averaged over VZA 0–40°** but
  only **0.8 W/m² over 50–60°**. With 30 s camera sampling every scene gets at
  least one view in this low-sensitivity zone. EarthCARE BBR uses the same idea
  (fore/aft 55° views).
- [derived] Two distinct angular figures of merit must not be confused:
  (i) *estimation leverage* for fitting the ADM shape parameter, which grows with
  the spread of the angular basis function; (ii) *retrieval insensitivity* to
  scene-ID error, which peaks near 50–60°. The RfMA optimises (ii).
- [derived] ABI's `curve_fit` `pcov` is NOT the transferable quantity (artefact
  of a ~1e7-sample fit). What ABI must deliver is the **within-scene population
  spread** of the shape parameter, which is ECO's prior width and its N=1 error.
- [derived] Regularized per-pixel estimator:
  `1/sigma_b^2(N) = 1/sigma_b_pop^2 + sum_i (dL_i/db)^2 / sigma_Li^2`.
  At N=1 the pixel falls back entirely on the scene-based ADM library, so the
  ABI-style scene ADM product is ECO's prior/fallback, not something superseded.

## 7. Scene identification

- [user 2026-10-05] Assess angular scene consistency within each Sunny file:
  identify its most frequent assigned scene across the supplied VZAs and
  report the percentage of views assigned to that scene. For count ties,
  select the tied scene with highest mean posterior across all views and
  flag the tie. Plot file population by this main scene, counting each file
  once. The user proposes label changes across VZA as a scene-ID bias diagnostic.
- [derived 2026-10-05] This statistic measures angular label consistency, not
  established physical classification bias: a fixed simulated atmospheric
  column has angle-dependent spectra, and unsupervised spectral GMM components
  need not coincide with invariant physical scene labels.
- [user 2026-10-05] The ABI/Sunny comparison uses a separate spectral-only
  classifier: train the GMM on ABI data only, using GOES-16/18 averaged BT
  inputs, then apply the same fitted preprocessing and GMM to Sunny spectra.
  Each provided Sunny directional view is used directly as the equivalent
  averaged-view input. Do not emulate GOES-16/18 viewing-angle pairs or
  synthesize spatial-texture features for Sunny.
- [derived 2026-10-05] The direct-view convention supports a common spectral
  scene taxonomy, not physical equivalence between a Sunny view and the
  arithmetic mean of two ABI BT views. Report Sunny occupancy by angle and
  regime; do not interpret classification probabilities as accuracy or library
  frequencies as climatology. The comparison classifier has its own scene
  IDs and must not select production texture-GMM ADMs without a separate
  justified mapping/retraining.
- [RfMA §6.2.4 p.63] LW ADM scene ID is expected to use **ECO LW channels 2, 4
  and 5** (Cloud Phase 8–9.1, Cloud Temperature 10.3–11.4, Cloud Optical Depth
  11.4–12.5) — BTs and BTDs.
- The Phase 0 demonstration used the analogous GOES ABI channels (8.4, 11.2,
  12.3 µm) with features **BT(11.2), BTD(12.3-11.2), BTD(8.4-11.2)** and a
  **GMM separating the observations into 10 scene classes**.
- [user 2026-09-30] **The 10 classes are superseded.** That figure predates the
  improved GMM diagnostics (`scripts/find_nComponents.py`). Current status: do
  NOT use the lowest n_components candidate; **use 6 or 7**. The current
  `config.yaml` default is now **7**, matching the locally selected
  `selected_n_components.json`; this count drives the default ABI proxy
  scene-ID branch.
- [RfMA §4.3.3 p.42] Scene-ID error propagation is **non-linear** (a wrong class
  selects a wrong ADM) and is considered potentially more consequential than the
  narrow-to-broadband regression error. Window channels used for scene ID
  therefore need *higher* sensitivity than the absorption channels.

## 8. Cloud-top height and parallax

- [user] CTH is taken from the GOES ACHA product in the proxy chain, but ECO
  should derive **its own CTH from the LW camera**.
- [RfMA §6.2.3 p.63] In the Phase 0 GOES demonstration the two datasets were
  "co-registered at the surface and cloud parallax error corrected using the
  **measured cloud brightness temperature**".
- [derived] Consequence 1 — error correlation: an ECO-internal CTH comes from the
  SAME radiances as the fluxes, so the parallax-correction error is not
  independent of radiance noise/calibration error; propagation needs a joint
  covariance. The ACHA-based proxy IS independent, so the ABI study will
  UNDERSTATE this coupling.
- [derived] Consequence 2 — the channel trade gains a CTH axis, closing a loop:
  channel set -> CTH accuracy -> parallax residual -> ADM shape bias -> flux bias.
  ECO's tiles are wide (1.1–1.5 µm) vs ABI's narrow (~0.5–1 µm), diluting CO2
  contrast for CO2-slicing; but the **goal** channel 6 extends to 16 µm and so
  covers the 15 µm CO2 band far better than the 14 µm threshold variant.
- [derived] Consequence 3 — **stereo CTH** is available to ECO and not to ABI:
  N quasi-simultaneous multi-angle views make the parallax displacement itself
  observable (MISR-like), so parallax is potentially self-correcting by
  inter-view registration, with no radiometric CTH and no spectral assumptions.
  Downgrades the parallax risk wherever N >= 2; at N = 1 the radiometric CTH and
  its error are unavoidable. Note ObsReq 17 explicitly requires the spatial
  information needed "to allow the co-registration of multi-angular views".
  The GOES proxy (2 fixed views) cannot demonstrate stereo registration.
- [derived] Scale: H*tan(theta) = 21 km at H = 12 km, theta = 60°, i.e. ~2 ECO
  product footprints. Because the displacement is monotonic in the same theta
  being regressed, uncorrected parallax **biases** the ADM shape parameter
  rather than adding noise.

## 9. Candidate LW channel definitions (repo file)

- [user] `lw_spectral_responses_tmp.py` (repo root) holds 7 commented/uncommented
  candidate configurations: contiguous, near-square, non-overlapping tiles with
  sigmoid edges (`*10.` slope => ~0.4 µm transition width).
- [user 2026-09-30] **Which block was uncommented in the supplied file was
  random and carries no meaning.** The baseline is the RfMA Table 4.1 6-channel
  set (§4). The other blocks remain useful as *sweep candidates* for the
  channel-definition trade study, since the definitions are not yet final.
- Variants present: 6-ch square 6–14; **5-ch fusing Cloud Phase + Ozone into
  8–10.3**; 6-ch `fmb`-sloped; 6-ch "with tails"; **extended 4.5–18 µm**;
  an older 8–14 µm set; 3 triangular channels. The RfMA's own G/T split
  (5–16 µm goal vs 6–14 µm threshold) is itself a sweep axis.
- [derived] Contiguous tiling => sum_c L_c ~= integral over the tiled range, so
  the narrowband->broadband problem is **out-of-band extrapolation**, not
  interpolation (structurally unlike ABI's narrow, gapped sampling bands).
- [derived] Blackbody fractional function, share of emission:

  | band     | 250 K | 300 K |
  | -------- | ----- | ----- |
  | < 6 µm   |  1.3% |  3.9% |
  | 6-14 µm  | 37%   | 48%   |
  | > 14 µm  | 62%   | 48%   |

  => with the 6–14 µm **threshold** set, half to two-thirds of OLR is
  unmeasured, dominated by the far-IR H2O rotational band and 15 µm CO2. The
  **goal** 5–16 µm set captures substantially more. This out-of-band term is
  water-vapour-regime dependent and hence largely invisible to the disk-mean
  radiometer anchor.
- Housekeeping: the file has an unguarded module-level write loop into `out/`;
  must go under `if __name__ == "__main__":` before being imported.

## 10. GOES ABI proxy

- [user] GOES-16 + GOES-18 ABI are PROXY data for ECO tool development.
- [RfMA §6.2.4] The RfMA itself reports this approach: GOES ABI East/West Full
  Disk L1 radiances, co-registered and parallax-corrected, GMM scene
  classification into 10 classes, Elmer limb-darkening ADMs whose shape
  parameter is **constrained by the East/West radiance ratio**, assuming
  consistent calibration between the two satellites.
- [RfMA §6.2.4 p.64] **Headline proxy result: GOES East-West broadband OLR RMS
  difference = 5.5 W/m² at 20 km resolution, September 2023**, stratified by VZA
  (Fig. 6.10). Described as comparable to CERES broadband flux retrieval
  consistency (Loeb et al. 2007). This is the number this repository reproduces
  and must improve upon.
- [RfMA §6.2.4] Acknowledged proxy limitations: GOES view-angle combinations are
  "much more limited" than ECO's; angular sampling of each scene class is only
  obtained by **aggregating the same class from different geographical
  locations**, which aliases real geographical variability into the ADM training
  data — noise that ECO's WFOV multi-angle sampling largely removes.
- [RfMA §6.2.2] **ABI channels are a pessimistic proxy for the NTB step**: for
  SW, ABI's narrower channels give a 2-sigma narrow-to-broadband flux error
  **more than 3x** that of the nominal ECO channels.
- [derived] With exactly 2 views per pixel the ADM shape parameter is **exactly
  determined per pixel**, in closed form, no `curve_fit` needed:
  `b = (r - 1) / [ (e(th16) - 1) - r*(e(th18) - 1) ]`, `r = L16/L18`.
  Vectorizable in one pass. Yields sigma_b_pop (after deconvolving the
  analytically known noise term) and the conditioning curve
  `sigma_b ~ sigma_r / |e(th16) - e(th18)|`. Key unexploited diagnostic.
- 2 km native grid; `scripts/average_resolution.py --block-size 5` gives the
  10 km MO2 product grid exactly.
- GEO gives the continuous diurnal cycle -> the only way to quantify ECO's
  diurnal aliasing. NOTE: the pipeline currently ingests ONE slot per day; a
  multi-slot/diurnal ingest is a prerequisite for that study.
- GOES disk is comparable in size to ECO's WFOV radiometer footprint -> a
  synthetic WFOV forward model from the 2 km flux field is a faithful proxy for
  the §5 cross-calibration strategy.
- Proxy limits that do not transfer: ABI-specific NEdR, absolute calibration and
  INR; GEO-specific fixed geometry; and ECO channel emulation from ABI is only
  approximate where bands overlap (GMM spatial-texture features cannot be
  substituted by RT).

## 11. RT databases and CERES

- [RfMA §6.2.1] ECO Phase 0 NTB regressions use **LibRadtran and SBDART**, over
  0.25–5 µm (RSR) and 2.5–100 µm (OLR), the latter extended to 500 µm via Planck
  with the 100 µm brightness temperature to avoid a broadband bias.
- [RfMA §6.2.2] Phase 0 used the **GERB spectral databases** (Clerbaux et al.
  2008a,b): **>2300 LW scenes** and >700 SW scenes. Result: **k=2 fitting noise
  of the narrow-band to broadband FLUX regression = 1.5 % LW, 0.6 % SW, without
  any scene stratification.** Compare to ObsReq 16 (0.84 % OLR) — see D7.
- [user 2026-10-01] The same GERB/Clerbaux LW spectral database is already in
  `data/Sunny`; there is no need to obtain or replace it with another database.
  Local inventory confirms 2,310 clear-sky and 2,310 cloudy spectra, with
  matching IDs 0000-2309 (4,620 files total, two records per index). The file
  headers confirm GERB/Clerbaux provenance, but do not establish whether the
  clear/cloud records sharing an index use the same underlying atmospheric
  profile. `scripts/compute_temperature_SBDART.py` currently loads all 4,620
  files as separate regression records and writes that count as `n_scenes`.
- [derived 2026-10-01] The channel sweep can use the existing full GERB LW
  library directly. Until source metadata establishes the relationship between
  clear/cloud records with a shared index, use the index as a conservative
  grouping key for train/validation splits rather than randomly splitting files;
  report both 2,310 indices and 4,620 spectra so the statistical sample unit is
  explicit. Stratify scores by scene regime where labels/metadata permit, and
  compare the resulting k=2 fitting error to ObsReq 16 (0.84 %) and the RfMA's
  unstratified 1.5 % result.
- [user] CERES products can be obtained. **Relevant.**
- [RfMA §6.2.6] CERES TRMM ADMs describe **1035 LW and 536 SW scene classes**
  and are used in the RfMA to quantify angular sensitivity — a directly reusable
  external reference.
- CERES roles here: (a) SSF footprints (~20 km, the resolution at which the RfMA
  quotes its 5.5 W/m² result) as the first EXTERNAL truth — everything currently
  in this repo is internal G16-vs-G18 self-consistency, which by construction
  cannot detect common-mode error; (b) published CERES LW ADMs as an independent
  check on the Elmer model-form error; (c) check on footprint-heterogeneity bias.

## 12. Discrepancies — resolutions

Resolved by the user on 2026-09-30:

- **D1 — channel count. RESOLVED: use the RfMA baseline (6 TIR channels,
  Table 4.1).** The comment state of the supplied `lw_spectral_responses_tmp.py`
  was random. The other candidate blocks are retained as sweep candidates only.
- **D2 — "10 km". RESOLVED: follow the RfMA.** Two distinct quantities, both
  kept: TIR *measurement* resolution 2 km (G) / 10 km (T) [MeasReq 10]; *flux
  product* resolution 10 km (G) / 50 km (T) [ObsReq 9]. The proxy chain at 2 km
  therefore already sits at the camera goal resolution, and `--block-size 5`
  produces the product goal grid.
- **D3 — ADM functional form. RESOLVED: keep the repo convention**, but the two
  forms are NOT mathematically equivalent (verified; ~1.3 % shape difference
  over 0-70 deg at typical b => ~3 W/m²). See §6 for the numbers. Actions:
  (a) document the repo form as a regularised variant, not as Elmer et al.
  (2016); (b) carry the repo-vs-RfMA form difference as a quantified ADM
  model-form error term; (c) never quote repo `b` against RfMA `a` without
  conversion (roles are also swapped).
- **D4 — scene count. RESOLVED: 7** for the current ABI proxy workflow, matching
  the local `selected_n_components.json` result. Six remains an acceptable
  alternative for future sensitivity studies; neither the RfMA's legacy 10 nor
  the lowest candidate should be chosen automatically. The model path follows
  the selected component count and training-file list.
- **D5 — constellation size. RESOLVED: 2 satellites is the baseline and
  preferred, but a 1-satellite configuration is still live.** So the
  1-vs-2 sampling trade remains a worthwhile deliverable: quantify the monthly
  diurnal-aliasing penalty of a single satellite (4 diurnal cycles/yr, ~2 local
  times at a time) against the 2-satellite baseline (8 cycles/yr, 4 local times,
  full coverage in ~46 days).
- **D6 — altitude.** 600 km is **Concept A**; Concept B is 800 km. Not a single
  baseline; inclination differs accordingly (82.2° vs 81.42°). Still open at
  mission level; use Concept A for this repo's studies.
- **D7 — NTB gap.** RfMA reports **1.5 % (k=2) LW** NTB flux regression noise
  "without any scene stratification", against **ObsReq 16 = 0.84 % for OLR**.
  Scene stratification of the NTB regression is the obvious lever and an
  immediately actionable target for this repository.

## 13. Standing conclusions (uncertainty framework)

Current-status note (2026-10-06): the development entries below preserve earlier
experiments as history, including superseded automatic component selection and
Sunny-derived assignment probabilities. The current nominal configuration uses
manually specified six components, ABI-pixel-derived assignment transitions
included under assumed ECO response transfer, and the independent spatial proxy.
Its combined RMSE is 1.9930 W/m2; the paired Sunny subtotal is 1.9560 W/m2.
See the [current README summary](../README.md#current-uncertainty-status-2026-10-06).
Earlier excluded-assignment and 1.931 W/m2 results are not the current budget.

- [user 2026-10-06] The full nominal-budget run must use all 25 available ABI
  days, split between held-in and held-out observations. ABI chunking is only
  for computational efficiency, not sampling. Keep a fastest partial-data
  test profile alongside the full-data profile.
- [derived 2026-10-06] Full-data execution configuration retains the existing
  20-day production training split and its five-day complement
  246/252/259/265/271 for evaluation. Native spectral/radiometric eligibility
  and two-satellite VZA <=70 degrees replace inherited sampled-cache
  texture/tile screening. All eligible records enter scaler/PCA/GMM EM,
  ABI scene-ADM fitting, held-out population counts and noise transitions;
  every complete eligible grid-aligned spatial footprint is assessed without
  chunk-boundary loss. Full-mode reliability diagnostics are explicitly
  initialization ARI rather than the sampled mode's bootstrap ARI. This is
  a new execution scope, not a new validated numerical mission budget; ABI
  transfer, Sunny representativeness, spatial independence and ECO optical
  response limitations remain.
- [user 2026-10-08] Implement full nominal-workflow performance improvements
  without intentionally changing the physics or statistical experiment.
- [derived 2026-10-08] Optimized full execution shares float64 scaler/PCA
  preprocessing and nominal scene ADMs, uses bounded resource-aware parallel
  EM and independent stage jobs, and enlarges compressed-NetCDF chunk caches.
  Full-population eligibility, random streams, footprint alignment, covariance
  model, convergence tolerance and manual component choice are retained.
  Reduction-order roundoff may affect convergence or near-tied assignments;
  numerical equivalence is tested, not bitwise-identical complete mission
  budgets. This is an execution change, not new uncertainty evidence.

- [user 2026-10-06] Expand nominal four-panel GMM diagnostics to every
  component count 2 through 15 inclusive. Spatial diagnostics must contain
  every 2 km point from 2 through 20 km; nominal choices remain six
  components and 10 km.
- [assumed 2026-10-06] Expanded spatial averaging blocks are anchored to
  the full ABI grid origin, not restarted at tile boundaries. Incomplete
  tile-edge blocks are excluded at each resolution. Footprint populations
  can therefore differ by resolution within the same sampled tile locations.
- [derived 2026-10-06] Expanded run completed all GMM counts 2-15 and
  spatial resolutions 2-20 km in 2 km steps. Spatial RMSEs in that order:
  0.0000, 0.2715, 0.2970, 0.3202, 0.3477, 0.3706, 0.3882,
  0.4097, 0.4314, 0.4452 W/m2. At 10 km the footprint count remains
  306,644 and the combined nominal budget remains 1.9930 W/m2 RMSE.
  Evaluation is conditional on 192 eligible non-overlapping cached tiles
  per day, not the full overlapping region.
- [derived 2026-10-06] Implemented ABI spatial processing proxy on 306,644
  complete matched 10 km footprints across held-out days 246/252/259/271.
  Reused average_resolution block averaging on cached 2 km tile locations;
  fitted spectral-GMM-conditioned ABI ADMs only on training days and used
  existing ABI cubic broadband conversion in both processing routes.
  Coarse-minus-mean-fine bias +0.0200 W/m2, scene SD 0.3471 W/m2,
  deterministic random SD 0, RMSE 0.3477 W/m2. No extra spatial-noise
  perturbation or ECO optical PSF is simulated. Mean satellite flux is
  compared; ABI two-view geometry and broadband spectral reference differ
  from the Sunny/ECO branch. Identity block-size-one comparison closes.
  Under user-assumed spatial independence, combined bias +0.6423,
  scene SD 1.8484, random SD 0.3780 and RMSE 1.9930 W/m2.
- [derived 2026-10-06] Restored original four-panel component diagnostic
  tools on the spectral-only cached population. All tested counts receive
  exact production ADM/cubic-flux scoring; bootstrap reliability uses the
  existing three-refit procedure, not only initialization variability.
  Six-component bootstrap minimum ARI is 0.9703. No tested count passes
  every default diagnostic criterion; the manual nominal six remains
  unchanged. Seed variability is provided as a supplementary figure.
- [user 2026-10-06] Reuse the existing four-panel GMM component diagnostics
  (likelihood/ICL, reliability, ADM quality, angular coverage/production
  impact) for the spectral-only budget classifier. Seed variability may
  supplement these diagnostics, not replace them.
- [user 2026-10-06] Include the ABI coarse-radiance-first minus
  mean-fine-retrieval broadband processing difference in the numerical
  budget, and combine it with other sources assuming independence.
  Independence and ABI-to-ECO transfer are work hypotheses, not measured
  cross-source covariance or absolute spatial-error truth.
- [user 2026-10-06] Keep six components as the default for the nominal
  spectral-only GMM; the final component count will be selected manually
  by inspection of GMM diagnostic figures, not by automatic likelihood
  selection. This decision concerns the nominal budget classifier, not
  automatically the separate historical texture-feature ABI classifier.
- [user 2026-10-06] Treat observed ABI radiance as truth for the incremental
  noise experiment, despite its existing instrument noise. Add extra NEdT
  noise to ABI radiances and compare baseline/noisy scene assignments on
  ABI pixels; Sunny must not determine the assignment-change probability.
  Include the propagated scene-assignment contribution in the budget under
  the explicit assumption that ECO responds similarly. The user selected
  fixed-model noisy inference, not noisy GMM retraining.
- [derived 2026-10-06] Revised execution samples 50,000 held-out native ABI
  overlap pixels and 30 added-noise realizations. Baseline inputs are
  uncorrected radiances matching the cached BT features, with separate
  G16/G18 Planck coefficients. Independent channel/satellite radiance noise
  corresponds to 0.4 K at 255 K per satellite, before mean-BT features.
  ABI-population-weighted assignment-change probability is 9.03%.
  Scene-conditional transitions are transferred to Sunny file labels,
  independently of simulated ECO-channel noise; this assumed transfer is
  not a measured physical cross-channel/sensor noise covariance.
  Included assignment increment: bias -0.0216, scene SD 0.1034,
  random SD 0.2957, RMSE 0.3140 W/m2. Revised conditional subtotal:
  bias +0.6223, scene SD 1.8155, random SD 0.3780, RMSE 1.9560 W/m2.
  These results supersede the earlier Sunny-noise-based 25.9% assignment
  experiment and fixed-label 1.931 W/m2 subtotal for the current budget.
- [user 2026-10-06] Accept the currently limited Sunny scene
  representativeness provisionally; retain the coverage caveat but do not
  block the budget work on expanding the library.
- [user 2026-10-06] Implement the agreed scene-conditioned ABI/Sunny nominal
  uncertainty budget, with explicit bias, scene-dependent SD and random SD.
- [derived 2026-10-06] First nominal implementation uses separately selected
  spectral-only GMMs on bounded cached ABI overlap tiles, 50,000 training
  and 50,000 held-out-day rows. Component selection uses the smallest
  adequately occupied candidate within one held-out-day likelihood SE of
  the best tested fit; this is a conservative four-day heuristic, not a
  demonstrated universal optimum. ABI held-out complete-pixel populations
  weight Sunny scenes, conditional on coverage. Sunny file labels use a
  majority of 15 view labels with mean-posterior tie-breaking. ABI SRFs
  define taxonomy; ECO goal SRFs define physical band retrieval.
- [derived 2026-10-06] Nominal scene ADMs share original modified-log shape
  parameters per scene/channel with separate file amplitudes and equal
  relative-residual weights per file. Five shared-index Sunny folds keep
  evaluation files out of ADM/N2BC fitting. Weighted N2BC is trained on true
  band flux against exact-boundary integrated 4-100 um truth; full-library
  application models are saved separately from assessment fold models.
  Thirty noise realizations separate persistent bias, across-file mean-error
  spread and within-file random variance, correcting scene variance for
  finite Monte Carlo means. Covariance is retained on paired residuals.
- [assumed 2026-10-06] Noise-induced assignment changes use 0.4 K NEdT at
  255 K translated through ABI SRFs for the classifier, independently of
  ECO goal-channel noise. This is an ABI-equivalent taxonomy proxy, not an
  established ECO classifier noise transfer; its mediated flux increment
  is explicitly excluded from the conditional ECO subtotal.
- [derived 2026-10-06] Executed nominal grouped budget: selected six
  spectral-only components, seed 73, with initialization adjusted-Rand
  agreement >0.99. Sunny covers all six scene masses but weighted effective
  sample size is about 343; one 33.3%-mass scene has only 42 spectra.
  Weighted low-ABI-density view fraction is 13.5%, indicating transfer
  limitations despite nominal scene coverage. Conditional spectral/angular/
  fixed-label-noise RMSEs are 1.008/1.347/0.237 W/m2. Paired joint subtotal:
  bias +0.644, scene SD 1.805, random SD 0.237, ensemble RMSE 1.931 W/m2.
  Within-scene pooled-versus-per-file broadband spread is 1.156 W/m2 SD.
  ABI-equivalent noise changes file labels with weighted probability 25.9%;
  its mediated-flux RMSE is 0.537 W/m2, excluded from the ECO subtotal.
  Calibration, spatial mixing, model-training/library uncertainty and
  monthly sampling remain unquantified; no complete mission total claimed.
- [user 2026-10-06] Budget table must explicitly show the RfMA error
  classes; interpret as bias, scene-dependent and random (§7.4.1 p.90),
  distinct from physical contributor rows. Within-scene spread is to inform
  the SD entry after propagation into broadband-flux residuals, not merely
  coefficient spread. The radiometer-specific one-day averaging assumption
  must not automatically be transferred to LW camera/retrieval errors.
- [user 2026-10-06] Proposed nominal retrieval: train/validate a
  spectral-only ABI GMM with held-out days and component-count selection;
  apply the common feature representation to Sunny and assign each file
  a scene. Fit original-form ADMs per Sunny file using 15 views, and pooled
  scene ADMs from Sunny files in each assigned scene; use per-file versus
  pooled differences to assess within-scene angular spread. Use ABI scene
  populations to weight Sunny N2BC fitting. Estimate noise-induced
  scene-assignment changes, not an arbitrary switching fraction. This is
  approved for methodological review only; pooling, spectral transfer,
  weighting and validation details require agreement before implementation.
- [user 2026-10-06] Budget reference: instantaneous broadband OLR on a
  10 km product footprint; monthly estimates only if justified; RfMA goal
  channels; 15 equally spaced Sunny views; original repository modified-log
  ADM; RfMA broadband spectral domain (4-100 um). Retrieval estimator remains
  to be confirmed. Seek a justified nominal assignment-error scenario rather
  than treating an imposed switching fraction as measured misclassification.
  Separate diagnostic outputs from uncertainty calculations and distinguish
  nominal configurations, error sources and affected processing models.
- [RfMA MeasReq 10/11 pp.43-44; ObsReq 9 p.36; §7.4.2 p.91,
  checked 2026-10-06] Native multispectral radiance resolution: 2 km goal,
  10 km threshold for SWIR/TIR, applying from nadir through the central
  imaging field. Flux-product resolution: 10 km goal, 50 km threshold.
  Camera design assessment uses the 10 km threshold across the central
  field and reports degradation toward its edge. The text does not specify
  a native 10 km nadir goal; actual nadir performance requires reading the
  design curves in Fig.7.15. User interpretation of 10 km native nadir
  resolution is not adopted as a confirmed mission fact.
- [user 2026-10-06] Refocus on a well-motivated numerical broadband-flux
  uncertainty budget, distinguishing the four implemented diagnostic studies
  from uncertainty estimates. Do not include CERES at this stage.
- [user 2026-10-06] Add ECO geometry and assignment-response diagnostics
  before extending ABI evaluation to full views. Geometry option 2:
  direct identifiable multi-view amplitude/shape fits with equal weights,
  sparse/wide versus clustered angles and requirement-level Gaussian noise;
  no single-view scene prior and no angular-weighting experiment. Assignment
  stress fractions: 0/1/5/10/20/30/40/50 percent.
- [assumed 2026-10-06] Geometry diagnostic angle sets are idealized Sunny
  grid subsets with VZA <=70 degrees, not orbit-derived sampling. The
  55-degree angle remains mathematical normalization only; an observed
  55-degree radiance is not required. Regularized fits need >=2 distinct
  views, quadratic fits >=3; the minima provide no residual redundancy.
- [derived 2026-10-06] Executed option-2 geometry diagnostic on 4,620 Sunny
  spectra with fixed grouped held-out ECO N2BC models and five goal-noise
  realizations. Quadratic three-view 45/50/60-degree geometry: 0.987 W/m2
  noise-free versus 7.538 W/m2 mean noisy broadband truth RMSE; 0/35/70-degree
  geometry: 0.962 versus 1.242 W/m2. This identifies noise amplification
  in clustered minimal-view fits, not an orbit-derived ECO allocation.
  New direct amplitude/shape fitting does not use the historical observed
  55-degree normalization; its dense result is a different estimator.
- [derived 2026-10-06] Nested ABI second-choice stress on the fixed
  306,644-footprint population: mean RMSE over three seeds at
  0/1/5/10/20/30/40/50 percent is
  0.000/0.062/0.167/0.280/0.497/0.711/0.924/1.137 W/m2.
  Zero stress closes exactly; fractions are imposed stress levels, not
  estimated misclassification probabilities.
- [user 2026-10-05] Implement and execute the follow-up ABI regime diagnostics
  and convergence study. Approved ladder: 12/48/192 eligible tiles per day;
  training days 245/247/254/261; evaluation days 246/252/259/271; seeds
  42/73/109; two concurrent jobs and no full-disk copies. Keep the first-draft
  outputs intact, fix evaluation cases for training variations and separate
  sampling from GMM initialization.
- [derived 2026-10-05] The follow-up separately varies evaluation tile pools,
  training tile coverage, GMM training-point count and initialization. Scene
  IDs stay local to each fitted model. Day/tile hierarchical bootstrap
  intervals preserve paired satellites but describe only four selected
  held-out days and eligibility-screened tiles; they are exploratory, not
  climatological or absolute ECO-accuracy confidence intervals.
- [derived 2026-10-05] Completed follow-up largest fixed-baseline ABI pool:
  306,644 matched 10 km footprints on four held-out days, two paired
  satellites. Spatial-order/ADM-form/6-vs-7-component/assignment-stress RMSE:
  3.028/1.248/0.857/0.280 W/m2. Spatial training-coverage sensitivity remains
  unsettled; stable source scores do not establish baseline retrieval
  stability. No fixed-baseline scene/channel ADM touched the positivity
  boundary. Low C14 footprint heterogeneity dominates spatial squared
  residual in this population; separate coarse texture/classification
  changes from nonlinear averaging before treating this as physical ECO
  spatial-resolution uncertainty. Preserve the original first-draft table.
- [user 2026-10-05] Implement and execute the first-draft sensitivity DAG;
  populate the main numerical-budget figure with run-derived results.
  Approved execution scope: bounded complete ABI footprints on two training
  and two held-out days, full Sunny library, and five noise/assignment
  realizations rather than the full ABI inventory.
- [derived 2026-10-05] First-draft assessment uses training days 245/247 and
  evaluation days 246/252, 12 eligible non-overlapping 100x100 native tiles
  per day, production texture features and fixed training scene ADMs.
  These are temporal-holdout proxy sensitivities, not global climatology.
  Sunny noise excludes ECO scene-selection effects; library representativity
  and physical realism remain unquantified. The joint Sunny scenario combines
  noise, the quadratic ADM and robust N2BC against simulated truth, not by
  adding ABI source sensitivities.
- [derived 2026-10-05] Execution exposed unphysical negative limb
  extrapolation in unconstrained Sunny and sampled ABI ADM fits. The
  assessment enforces positive 0-90 degree profiles on a 0.1-degree grid
  and records constrained/boundary scene-channel counts. This deliberately
  changes the assessment baseline relative to earlier unconstrained scores;
  production fitting is unchanged. Keep the distinction in figure captions
  and do not attribute implementation validity failures to instrument error.
- [user 2026-10-05] Design a Snakemake sensitivity workflow to fill the
  unified uncertainty table by rerunning radiance-to-broadband processing for
  each contributor. For the first version, quantify scene-ID sensitivity
  through the existing ABI chain and label it as proxy evidence; retain the
  ECO scene-transfer gap explicitly rather than developing an ECO-compatible
  classifier and scene-conditioned Sunny retrieval at this stage.
- [user 2026-10-05] Unify the proposed CSV and main numerical-budget figure
  into one ECO assessment with source/experiment and final broadband-flux
  impact columns, not separate all/clear/cloud columns. Specify experiments
  for RfMA NEdT-driven Gaussian radiance noise; ABI 2 km versus 10 km
  processing; GMM component-count sensitivity and second-choice assignment
  on an assumed 10% of pixels; ADM functional-form sensitivity; and Sunny
  representativity, realism and outlier sensitivity. First unify the figure
  and experiment specification; do not run these new experiments yet.
- [derived 2026-10-05] The unified specification is maintained in
  `config/uncertainty_experiments.yaml`; the CSV is exported from the same
  rows as the figure. Use fixed models and pooled held-out regimes for
  supporting Sunny scores, not row-wise minimum clear/cloud errors.
  The 2 km ABI reference is finer-resolution evidence, not absolute truth;
  spatial comparison must put both outputs on matched 10 km footprints.
  New sensitivity impacts remain pending and are not identified with the
  existing angular/spectral residuals or included in a full ECO total.
- [user 2026-10-05] Proposed uncertainty experiments are recorded in
  `Uncertainty budget and propagation for ECO_LW - Sheet1.csv`: use ABI
  held-out scene classification and second-choice assignments as a scene-ID
  sensitivity experiment; GOES-16/18 narrowband-flux differences as an
  empirical spread diagnostic; and matched Sunny ABI/ECO angular retrieval
  errors to investigate transfer to ECO. The objective is source-separated
  ECO broadband-flux uncertainty using both datasets. This is a proposal for
  review, not an adopted transfer law or a validated uncertainty budget.
- Split every term into random / systematic / bias; only the random part averages
  down. The RfMA uses the same three-way split for the radiometers (bias,
  scene-dependent, random), with scene-dependent errors assumed to average over
  ~1 day (~14 orbits) but not within one orbit — adopt the same taxonomy for
  consistency with the mission error budgets.
- Store each uncertainty component as a compact LUT over
  (scene, theta, angular leverage, footprint, channel set) — NOT as
  full-resolution fields (3.8 GB per variable).
- Footprint heterogeneity biases arise from three nonlinear operators applied to
  a heterogeneous 10 km footprint: F = sigma*T^4 (~2 W/m² at subgrid sigma_T =
  10 K, ~20 W/m² in broken cloud), the cubic NTB regression, and the ADM
  correction. Measured as: operate at 2 km then average to 10 km, vs. average to
  10 km then operate.
- Work order:
    1. **Build the ECO-targeted radiance-to-flux simulation chain** using the
      already-present GERB/Clerbaux angular spectra: convolve directional
      radiances with each ECO channel scenario; fit the repository ADM form per
      simulated scene/channel; retrieve narrowband flux from the configured
      idealized 15-view geometry; then fit grouped held-out N2BC regressions and
      compare OLR errors against ObsReq 16. Report ADM-fit and narrowband-flux
      errors separately from the final OLR reconstruction error. Keep ABI
      processing as a parallel empirical proxy contribution for scene-ID and
      angular-algorithm development, not as an alternative ECO configuration.
      Record source and transferability for every result.
  2. **Per-pixel closed-form ADM inversion** on G16/G18 -> sigma_b_pop -> Fisher
     calculator -> confront with the now-known NEdT = 0.4 K.
  3. **Angle-weighted retrieval** following RfMA §6.2.6: quantify the
     scene-sensitivity minimum near 50–60° with the repo's own ADMs.
  4. **Resolution ladder** at block size 5 (2 km -> 10 km).
  5. **Parallax / CTH sensitivity**.
  6. **Synthetic WFOV** forward model -> which error sources the radiometer
     anchor can and cannot detect.
  7. **Diurnal sampling simulation** on multi-slot GOES -> monthly aliasing bias
     for the 1-satellite vs 2-satellite configurations (D5: 1 sat is still
     live). Requires extending the ingest beyond one slot per day.
  8. **CERES collocation** for external truth.
  9. Uncertainty plumbing last.
- Phase A roadmap items in the RfMA this repo can serve directly
  [RfMA Table 6.3 p.71]: "Refine Earth radiation budget sampling error for MO2"
  (started); "Fully exploit the ECO observational capabilities for improved scene
  identification and angular dependence model generation, potentially including
  alternative radiance to flux retrievals" (mature baseline available); "Develop
  performance simulation for MO2, including cross-calibration with the ECO
  radiometers, and the implementation of multi angular viewing radiance to flux
  conversion" (planned).
