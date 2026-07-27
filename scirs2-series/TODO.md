# scirs2-series TODO

## Status: v0.6.3 — Documentation Audit (July 15, 2026)

Untouched by this release's fix work (no series-specific changes shipped in 0.6.3); this documentation audit was performed for 0.6.2 and remains accurate for 0.6.3 since the crate source is unchanged.

A README/TODO accuracy pass was done directly against `src/` (grep + `Read`, not just skimming
doc comments). Headline findings:

- `todo!()`/`unimplemented!()` count in `src/`: **0**.
- The large majority of the feature surface is real and reachable: classical + neural
  forecasting (ARIMA/SARIMA/ETS/BATS/TBATS/Theta, TFT/N-BEATS/N-HiTS/DeepAR), the GARCH family,
  ARFIMA/Hurst, causality (Granger/transfer-entropy/CCM/PC/PC-stable/PCMCI/FCI), VAR with
  impulse response and variance decomposition, streaming statistics (P², ADWIN, reservoir
  sampling), hierarchical reconciliation (MinT/WLS/OLS), and the financial/environmental/
  biomedical/IoT domain modules all check out against `src/`.
- Several previously-`[x]` items turned out to be either unreachable (real code exists in
  `src/` but the containing directory is never `mod`-declared from `lib.rs`, so it is not part
  of the compiled/public crate) or simply not present anywhere in the tree. These are unmarked
  below with a **discrepancy** note at the point they were originally claimed, plus a summary
  in "Known Issues".
- Several README Quick Start code examples referenced types/methods/module paths that do not
  exist (`AutoArima::fit`, `neural_forecast::TFT`, `causality::granger_causality`,
  `streaming::adwin::ADWIN`, `reconciliation::{MinTReconciler, HierarchyMatrix}`,
  `conformal::SplitConformalForecaster`) and would not have compiled; these were rewritten
  against verified real signatures in README.md.
- Fresh test run this pass (2026-07-15): `cargo nextest run -p scirs2-series` (default features)
  → 1752 tests run, 1752 passed, 1 skipped, 0 failed; `--all-features` → 1805 tests run, 1805
  passed, 1 skipped, 0 failed. See README.md's "Testing" section for the runnable commands.

## Status: v0.4.3 Released (May 3, 2026)

34,275+ workspace tests pass (100% pass rate). All v0.4.3 features are complete and production-ready. The Wave 3 stub-check sweep implemented `PatternDetector`, `STLDecomposer`, `AnomalyDetector`, and `PELTDetector` as public, fully working types, and Wave 4 restored `SarimaModel.forecast(steps)` alongside the `SARIMAModel` type alias plus the `training_data` field used by integration tests.

## Status: v0.3.4 Released (March 18, 2026)

19,644 workspace tests pass (100% pass rate). All v0.3.4 features are complete and production-ready.

---

## v0.3.3 Completed

### Neural Architecture Forecasters
- [x] Temporal Fusion Transformer (TFT): variable selection networks, gating (GLU), static covariate encoding, multi-horizon attention decoder
- [x] N-BEATS: neural basis expansion with trend and seasonality stacks; generic and interpretable variants
- [x] N-HiTS: hierarchical interpolation with multi-rate signal sampling and multi-resolution blocks
- [x] DeepAR: autoregressive LSTM with probabilistic output distributions (Gaussian, negative binomial, student-t)
- [x] Simple neural forecast API: common interface across all neural models with configurable training loops

### State-Space Models & Kalman Filtering
- [x] Kalman filter and Rauch-Tung-Striebel smoother
- [x] Extended Kalman filter (EKF) with analytical or numerical Jacobians
- [x] Unscented Kalman filter (UKF) with sigma-point propagation (Merwe parametrization)
- [x] Structural time series: local level, local linear trend, seasonal, cycle components
- [x] Dynamic linear models (time-varying system matrices)
- [x] Innovations state space representation (ETS models)

### Volatility Models
- [x] GARCH(p,q) — QMLE estimation, forecasting, simulation
- [x] EGARCH — exponential GARCH with asymmetric leverage
- [x] FIGARCH — fractionally integrated GARCH for long-memory volatility
- [x] GJR-GARCH — asymmetric response to positive/negative shocks
- [x] ARCH-LM test, Ljung-Box test on squared residuals

### Long-Memory Processes
- [x] ARFIMA estimation (Whittle, CSS, and exact ML)
- [x] Hurst exponent: R/S analysis, DFA, Whittle spectral estimator, variogram method
- [x] Fractional differencing operator (exact and fast approximate via convolution)
- [x] FARIMA simulation with specified memory parameter

### Granger Causality & Cointegration
- [x] Granger causality: Wald test (F-statistic), bootstrap p-values, multivariate block-exogeneity — `causality::granger` (`granger_causality_test`, `conditional_granger_test`, `spectral_granger_causality`, etc.)
- [x] Transfer entropy with bootstrap significance testing and bias correction — `causality::transfer_entropy`
- [x] Convergent cross mapping (CCM) for nonlinear causal detection — `causality::CCMResult`/`CCMConfig`
- [ ] Engle-Granger two-step cointegration test — **discrepancy (2026-07-15)**: only exists as a
      *private* helper (`regression.rs::test_cointegration`, no `pub`) inside the ECM-fitting
      pipeline, with a hand-rolled step-function p-value approximation (its own comment calls
      it "a very simplified approximation"). There is no standalone, publicly callable
      Engle-Granger test. Unmarked.
- [ ] Johansen trace and maximum-eigenvalue cointegration tests with critical values —
      **partial discrepancy (2026-07-15)**: a real trace test with tabulated 5% critical values
      exists in `streaming::cointegration::StreamingCointegrationTester` (rolling-window /
      online form only — no static/batch Johansen test). The maximum-eigenvalue statistic does
      not exist anywhere. Unmarked pending a batch variant + max-eigenvalue statistic.
- [ ] VECM estimation, impulse response functions, and forecast error variance decomposition —
      **discrepancy (2026-07-15)**: `VECMModel::fit` in `var_models.rs` is a placeholder — it
      sets `is_fitted = true` and returns without estimating anything (comment: "Placeholder
      implementation / Would implement full Johansen procedure"); `VECMModel::to_var` is
      likewise a placeholder. Impulse response and variance decomposition themselves *are* real,
      but on `VARModel` (`var_models::VARModel::{impulse_response, variance_decomposition}`) and
      via `causality::structural_var::{compute_irf, compute_fevd}` — not through a fitted VECM.
      Unmarked.

### Conformal Prediction
**Discrepancy found (2026-07-15)**: none of this section is reachable from the public API.
`src/conformal/mod.rs` (`SplitConformal`, `AdaptiveConformal`, `WeightedConformal`, `EnbPI`,
`empirical_coverage`) and `src/forecast_uncertainty/mod.rs` (`ConformalPrediction`, `EnbPI`,
`BootstrapPI`, `BayesianPI`, `CoverageTest`) both contain real, substantial code, but neither
directory is declared as a `mod`/`pub mod` anywhere reachable from `lib.rs`. `cargo doc` and
`use scirs2_series::conformal::...` both confirm the path does not exist — this is dead code
sitting in the working tree, not a compiled part of the crate. All four items are unmarked
pending the modules being wired in (or removed if intentionally abandoned).
- [ ] Split conformal prediction: exchangeable and time-series-adapted (EnbPI) variants
- [ ] Adaptive conformal inference (ACI) for online coverage guarantees
- [ ] Mondrian conformal for conditional coverage by covariate stratum — additionally, no
      "Mondrian" implementation exists anywhere in `src/`, wired or not
- [ ] Calibration diagnostics: empirical coverage plots, Winkler score, interval sharpness —
      Winkler score / coverage / sharpness *are* independently real and reachable via
      `evaluation::{coverage_probability, winkler_score}` and
      `energy_forecast::{coverage, winkler_score, sharpness}`, just not as part of a working
      conformal-prediction pipeline

### Intermittent Demand Forecasting
- [x] Croston's method (separated demand size and interval models)
- [x] Syntetos-Boylan Approximation (SBA) with bias correction
- [x] Teunter-Syntetos-Babai (TSB) model with demand probability update
- [x] Intermittency classification (smooth, erratic, lumpy, intermittent)

### Hierarchical Forecasting & Reconciliation
- [x] Aggregation strategies: bottom-up, top-down (AHP, PHA, TDA), middle-out
- [x] MinT (trace minimisation) with sample, shrinkage, and structural covariance estimates
- [x] WLS (weighted least squares) reconciliation
- [x] OLS reconciliation (equal weight)
- [x] Cross-temporal reconciliation

### Streaming / Online Algorithms
- [x] ADWIN (Adaptive Windowing) concept drift detector with statistical guarantees — `online_algorithms::ADWINDetector`
- [x] Online ARIMA with recursive least squares coefficient tracking
- [ ] Streaming statistics: mean, variance (Welford), quantiles (P² and KLL sketch) — P² is real
      (`online_algorithms::OnlineQuantile`); **discrepancy (2026-07-15)**: no KLL sketch exists
      anywhere in `src/`. Unmarked.
- [x] Online anomaly: CUSUM, EWMA control charts, streaming isolation forest
- [x] Reservoir sampling and sliding window aggregation

### Functional Data Analysis (FDA)
- [x] Functional PCA (FPCA) — reachable via `dimensionality_reduction::functional_pca`
      (covariance-operator approach in a basis space); could not confirm this is specifically
      the named PACE algorithm
- [ ] B-spline and Fourier basis expansions, smoothing spline roughness penalties —
      **discrepancy (2026-07-15)**: this exists (`src/functional/basis.rs`,
      `src/functional/smoothing.rs`) but `src/functional/` is not declared as a module in
      `lib.rs`, so it is unreachable. Unmarked.
- [ ] Scalar-on-function regression (functional linear model) — **discrepancy (2026-07-15)**:
      exists in the same unreachable `src/functional/regression.rs`. Unmarked.
- [ ] Functional clustering (k-centres functional, hierarchical functional) —
      **discrepancy (2026-07-15)**: not found anywhere, reachable or not. Unmarked.
- [x] Dynamic time warping barycenter averaging (DBA) — `dimensionality_reduction::dtw`,
      including a medoid-based initialization option

### Regime-Switching Models
- [x] Markov-switching autoregression (MS-AR) with Hamilton filter and EM estimation — confirmed real: `regime::{fit_msar, hamilton_filter, kim_smoother}`
- [ ] Threshold autoregressive (TAR) and self-exciting TAR (SETAR) models — **discrepancy (2026-07-15)**: no `TAR`/`SETAR` type, struct, or function exists anywhere in `src/` (full-tree grep for `SETAR`, `STAR`, `ThresholdAutoregressive`, and `threshold`+`autoregress` co-occurrence all came back empty except for unrelated hits). Unmarked.
- [ ] Smooth transition autoregressive (STAR) models (logistic and exponential) — **discrepancy (2026-07-15)**: same search as above; no smooth-transition implementation exists anywhere. Unmarked.
- [ ] Bai-Perron multiple structural break test — **discrepancy (2026-07-15)**: does not exist anywhere in `src/`. The only structural-break-adjacent code is a much simpler CUSUM-of-innovations detector, `state_space::detect_structural_break_ssm`, which is not the Bai-Perron sequential-F-test procedure. Unmarked.

### Probabilistic Forecasting & Evaluation
- [x] CRPS (Continuous Ranked Probability Score) — reachable via `energy_forecast::crps`
      (re-exported from a private `energy_forecast::evaluation` submodule), not from the
      general `evaluation` module as the module name might suggest
- [ ] Log score — **discrepancy (2026-07-15)**: no log-score implementation found anywhere in
      `src/`. Unmarked.
- [x] Reliability diagrams — `energy_forecast::reliability_diagram`
- [ ] PIT histograms — **discrepancy (2026-07-15)**: no PIT (probability integral transform)
      histogram implementation found anywhere in `src/`. Unmarked.
- [x] Diebold-Mariano test for forecast comparison — `evaluation::diebold_mariano`
- [x] MASE, SMAPE, WAPE, hierarchical MASE — `evaluation::{mase, mase_seasonal, smape, wape}`,
      `reconciliation::mase_hierarchical`

### Classical Models (Enhanced)
- [ ] Auto-ARIMA: stepwise AIC/BIC search and grid search with parallel evaluation —
      **partial discrepancy (2026-07-15)**: `arima_models::auto_arima` really does both a
      stepwise search (`stepwise_search`) and an inline (p, d, q) grid search selecting by
      information criterion. Two things do not hold: (1) `ArimaSelectionOptions.parallel` is
      set but never read anywhere in `arima_models.rs` — there is no parallel evaluation, and
      the crate has no `rayon` dependency; (2) in grid-search mode, the *seasonal* order search
      loop only overwrites `best_params.seasonal_pdq` on every iteration without ever fitting or
      comparing a seasonal model (comment: `// Fit seasonal model (placeholder)`), so seasonal
      grid search always just returns `(max_seasonal_p, seasonal_d, max_seasonal_q)`. Unmarked
      pending real seasonal grid search + removal/implementation of the `parallel` option.
- [x] TBATS with automatic period selection
- [x] Theta method and Theta-F (optimized theta)
- [x] Prophet-style seasonality decomposition with Fourier-series seasonal components and holiday effects

### Change Detection & Anomaly Detection
- [x] PELT with multiple cost functions (L1, L2, RBF, AR)
- [x] Binary segmentation (greedy and exact)
- [x] Bayesian online change point detection (BOCPD) with hazard function
- [x] Kernel change detection via MMD statistics
- [x] SPC charts: Shewhart, CUSUM, EWMA with control limits
- [ ] Matrix profile and motif/discord discovery — **discrepancy (2026-07-15)**: `TimeSeriesMotif`,
      `Discord`, and matrix-profile code exist in `src/pattern/mod.rs`, but that directory is not
      declared as a module in `lib.rs` (same "unwired" pattern as `conformal`,
      `forecast_uncertainty`, `functional`, `change_detection`). Unreachable. Unmarked.

### Feature Engineering (60+ features)
- [x] Statistical: 20+ moment and distributional features
- [ ] Frequency domain: spectral entropy, centroid, dominant frequency ratio are real
      (`features::frequency::{calculate_spectral_entropy, calculate_spectral_centroid,
      find_dominant_frequency}`); **discrepancy (2026-07-15)**: "bandwidth" is not — the
      `noise_bandwidth`/`equivalent_noise_bandwidth` fields on `SpectralAnalysisFeatures` are
      only ever zero-initialized (`Default` impl), with no assignment anywhere in
      `features/frequency.rs`; the function that would compute them,
      `calculate_spectral_analysis_features`, is explicitly marked
      `// Placeholder Functions (to be fully implemented)`. Unmarked pending real bandwidth
      computation.
- [x] Complexity: ApEn, SampEn, permutation entropy, Lempel-Ziv, fractal dimension
- [x] Lag-based: ACF/PACF at multiple lags, partial correlations
- [x] Automated selection: filter (MI, F-test, variance), wrapper (forward/backward/RFE), embedded (LASSO, RF importance)

### Domain-Specific Extensions
- [x] Financial: GARCH volatility, 15+ technical indicators (RSI, MACD, Bollinger Bands, CCI, MFI, OBV, ATR, Parabolic SAR)
- [x] Environmental: heat wave detection, SPI drought index, growing degree days, SOI/NAO climate indices, atmospheric storm detection
- [x] Biomedical: ECG R-peak detection (Pan-Tompkins), HRV analysis, EEG frequency bands, EMG onset detection
- [x] IoT sensors: environmental sensor fusion, GPS activity recognition, predictive maintenance scoring, data quality assessment

---

## v0.4.0 Roadmap

### Foundation Model Interface
- [x] Fine-tuning interface for pre-trained time series foundation models (TimeGPT-style) — Implemented in v0.4.0 (`foundation/fine_tuning.rs`)
- [x] Zero-shot forecasting adapter layer — Implemented in v0.4.0 (`foundation/zero_shot.rs`)
- [x] Prompt-based time series conditioning API — Implemented in v0.4.2 (`prompt_conditioning/mod.rs`: enum-based prompts, `PromptConditioner`, trend/seasonal/anomaly/level/noise/custom signals)

### Neural ODE for Time Series
- [x] Latent ODE / ODE-RNN for irregular time series — Implemented in v0.4.0 (`neural_ode/latent_ode.rs`, `neural_ode/ode_rnn.rs`)
- [x] Continuous normalizing flow models for density estimation — Implemented in v0.4.0 (`neural_ode/cnf.rs`, `neural_ode/ffjord.rs`)
- [x] Physics-informed neural time series models — Implemented in v0.4.2 (`physics_ts/` module: `PhysicsInformedTs`, ODE/conservation/monotonicity/bounded-variation constraints)

### Ultra-Long Context Handling
- [x] FlashAttention integration for TFT with very long lookback windows (10k+) — Implemented in v0.4.0 (`tft/flash_attention_tft.rs`)
- [x] State-space sequence models (Mamba / S4) for linear-time long-range dependencies — Implemented in v0.4.0 (`ssm/s4.rs`, `ssm/mamba.rs`, `state_space/s4.rs`, `state_space/mamba.rs`)
- [x] Hierarchical attention with sparse patterns for ultra-long sequences — Implemented in v0.4.2 (`hierarchical_attention/mod.rs`: local-window O(N·W), pooled O((N/s)²), global-token O(G·N) levels)

### Advanced Causality — Note: PCMCI implemented in v0.4.0 Wave 1
- [x] PC algorithm for causal structure learning from time series — Implemented in v0.4.2 (`causality/pc.rs`: skeleton, v-structures, Meek rules; also `causality/pc_stable.rs` for time-series variant)
- [x] PCMCI algorithm (Peter and Clark Momentary Conditional Independence) — Implemented in v0.4.0
- [x] Causal discovery with latent confounders (FCI for time series) — Implemented in v0.4.0 (`causality/fci.rs`)

### Bayesian Nonparametric Time Series
- [x] GP-state-space models (GP-SSM) with particle MCMC fitting — Implemented in v0.4.0 (`gp_ssm/` module, `state_space/gp_ssm.rs`)
- [x] Infinite hidden Markov model (iHMM) via stick-breaking construction — Implemented in v0.4.0 (`state_space/ihmm.rs`)
- [x] Nonparametric GARCH via GP volatility functions — Implemented in v0.4.0 (`volatility/gp_garch.rs`)

### Streaming Enhancements
- [x] RIVER integration bridge for additional online learners — Implemented in v0.4.2 (`online_learner/mod.rs`: `OnlineLearner` trait, `OnlineLinearRegression`, `OnlineStandardScaler`, `Pipeline`, `OnlineHoeffdingTree`, `OnlineMetrics`)
- [x] Incremental cointegration testing in streaming VAR — Implemented in v0.4.0 (`streaming/cointegration.rs`)
- [x] Online hierarchical reconciliation with incremental MinT — Implemented in v0.4.2 (`hierarchical/online.rs`: incremental MinT/WLS/OLS with streaming covariance updates)

---

## Known Issues

- DeepAR with negative-binomial output can exhibit numerical instability when series contain long runs of zeros; use TSB for highly intermittent demand instead
- FIGARCH estimation is slow for series longer than 10,000 points. **Correction (2026-07-15)**:
  the previous text here referenced a `parallel` feature — that feature does not exist in
  `Cargo.toml` (features are `serde`, `wasm`, `python`, `r`, `simd`) and the crate has no
  `rayon` dependency at all, so there is no feature flag that speeds this up today.
- FPCA with very sparse observations (fewer than 5 observations per subject) may produce poorly estimated eigenfunctions

### Discrepancies found in the 2026-07-15 documentation audit

**Unwired/unreachable modules** — real source code exists under `src/` but is never
`mod`-declared from `lib.rs`, so none of it is part of the compiled public crate. This turned up
five separate directories, suggesting a one-time reorganization left a batch of modules behind
rather than five independent oversights — worth a dedicated pass to either wire each one in or
delete it:
- `src/conformal/` (`SplitConformal`, `AdaptiveConformal`, `WeightedConformal`, `EnbPI`)
- `src/forecast_uncertainty/` (`ConformalPrediction`, `EnbPI`, `BootstrapPI`, `BayesianPI`, `CoverageTest`)
- `src/functional/` (B-spline/Fourier/wavelet basis, functional linear model, functional ANOVA, kernel/spline smoothing — `FPCA`/DTW-barycenter equivalents *are* separately reachable via `dimensionality_reduction`)
- `src/change_detection/` (a second, richer PELT/binary-segmentation/BOCPD/RBOCPD implementation) — superseded in practice by the reachable `change_point` + `detection::pelt` + `streaming::change_detection`, so this one is lower priority to wire in, but it is still dead code worth resolving (delete or wire in) per the crate's own file-size/dead-code hygiene
- `src/pattern/` (`TimeSeriesMotif`, `Discord`, matrix-profile code, plus a second `SAX` implementation) — note this is a *different* directory from the reachable `detection::pattern` submodule (which only has `PatternDetector`); the real, reachable SAX/APCA/PLA implementation lives in `dimensionality_reduction::symbolic`, so only the motif/discord/matrix-profile part of this directory is an actual gap

A systematic sweep (`comm` between all `src/*/` directory names and every `mod`/`pub mod` name
declared anywhere in the tree) turned up three more orphaned top-level directories beyond the
five above. None of these currently change a README claim (the capability they'd duplicate is
already real and correctly described elsewhere, or — for `panel` — the crate doesn't advertise
the feature at all), but they are additional dead code worth resolving in the same pass:
- `src/deepar/mod.rs` and `src/nbeats/mod.rs` — single-file, top-level, unreachable; likely
  superseded by the correctly-wired `neural_forecast::deepar`/`neural_forecast::nbeats`
  (`DeepARModel`, `NBEATSModel`), not independently verified beyond that
- `src/arch_tests/mod.rs` — unreachable; ARCH-LM itself is independently real and reachable via
  `volatility::arch::engle_test`
- `src/panel/` (`hausman.rs`, `did.rs`, `fixed_effects.rs`, `random_effects.rs`) — a substantial,
  real-looking panel-data econometrics module (fixed effects, random effects, Hausman test,
  difference-in-differences) that is not mentioned anywhere in README.md and is not reachable.
  Not added to README.md as a feature since it cannot currently be used, but worth flagging as
  a "free" feature if it gets wired in.

**Public API methods that silently return fabricated/placeholder results instead of erroring**
(none of these panic or return `Err`, so they are invisible to a `todo!()`/`unimplemented!()`
grep or to `cargo check`):
- `var_models::VECMModel::fit` — sets `is_fitted = true` and returns `Ok(())` without running any estimation ("Placeholder implementation / Would implement full Johansen procedure")
- `var_models::VECMModel::to_var` — likewise a placeholder
- `var_models::VARModel::granger_causality` — hardcodes `f_stat = 2.5, p_value = 0.05` regardless of input data ("Placeholder implementation / Would implement proper F-test for coefficient restrictions"); use `causality::granger_causality_test` instead, which is a real implementation
- A broader `grep -rn "Placeholder implementation" src/` turns up further hits in
  `ensemble_automl.rs` (meta-learner, LSTM/Prophet-like sub-models), `features/frequency.rs`
  (an entire "Placeholder Functions (to be fully implemented)" section feeding the unused
  `SpectralAnalysisFeatures::{noise_bandwidth, equivalent_noise_bandwidth}` fields — see Feature
  Engineering above), and in modules not described in README.md at all (`advanced_fusion_intelligence`,
  `quantum_forecasting`, `gpu_acceleration`) — these were not individually audited this pass
  since they are not currently advertised as features, but the same "looks-implemented,
  isn't" risk likely applies there too.

**Advertised features not found anywhere in `src/`, wired or not:**
- Threshold/self-exciting threshold (TAR/SETAR) and smooth-transition (STAR) autoregressive models
- Bai-Perron multiple structural break test (only a much simpler CUSUM-of-innovations break detector exists, in `state_space`)
- Mondrian conformal prediction
- Log score and PIT (probability-integral-transform) histograms for probabilistic forecast evaluation
- KLL sketch for streaming quantiles (the P² algorithm *is* implemented, in `online_algorithms::OnlineQuantile`)
- Bootstrap confidence bands around VAR impulse response functions
- A standalone (non-streaming) Engle-Granger cointegration test and a Johansen maximum-eigenvalue statistic (only a private, simplified Engle-Granger helper inside `regression.rs`'s ECM path, and a streaming/rolling-window Johansen *trace* test, exist)

README.md has been corrected to reflect all of the above, including fixing several Quick Start
code examples that referenced nonexistent types/methods (`AutoArima::fit`,
`neural_forecast::TFT`, `causality::granger_causality`, `streaming::adwin::ADWIN`,
`reconciliation::{MinTReconciler, HierarchyMatrix}`, `conformal::SplitConformalForecaster`) and
would not have compiled as written.
