# scirs2-series

[![crates.io](https://img.shields.io/crates/v/scirs2-series.svg)](https://crates.io/crates/scirs2-series)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-series)](https://docs.rs/scirs2-series)
[![Version](https://img.shields.io/badge/version-0.6.3-green)]()
[![Status](https://img.shields.io/badge/status-partial-yellow)]()

**Comprehensive time series analysis for Rust** — part of the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.

`scirs2-series` is a large time series library covering classical econometric models through state-of-the-art deep learning forecasters: the neural architecture forecaster suite (TFT, N-BEATS, N-HiTS, DeepAR), streaming/online algorithms, long-memory processes, intermittent demand forecasting, and hierarchical reconciliation are all real and reachable through the public API. A handful of advertised areas are weaker than the rest of the crate — see [TODO.md](./TODO.md) "Known Issues" for specifics (most notably: split/adaptive conformal prediction code exists in the source tree but is not yet wired into `lib.rs`, so it is not reachable from outside the crate; only Markov-switching autoregression is implemented among the regime-switching models; and `VECMModel::fit` is currently a placeholder that does not run the Johansen procedure). Everything else described below has been spot-checked against `src/`.

---

## Overview

Time series problems span a wide spectrum: univariate forecasting with uncertainty quantification, multivariate causal modelling, streaming anomaly detection, hierarchical forecasting across organizational hierarchies, regime detection, and functional data analysis. `scirs2-series` covers all of these in a unified, type-safe API.

Key design goals:

- **Breadth**: classical (ARIMA, ETS) through neural (TFT, N-BEATS, DeepAR) through streaming (online ARIMA, ADWIN)
- **Uncertainty quantification**: confidence/prediction intervals on ARIMA and other forecasters, probabilistic neural outputs (DeepAR), and evaluation-side coverage/Winkler-score diagnostics
- **Ecosystem coherence**: built on `scirs2-core` abstractions; no C/Fortran dependencies
- **Performance**: SIMD-accelerated operations via `scirs2-core` (`simd` feature); the crate has no `rayon` dependency — the `out_of_core` module instead uses `std::thread` directly for its own worker pool

---

## Feature List (v0.6.3)

### Decomposition
- STL (Seasonal-Trend decomposition using Loess) with robustness iterations
- TBATS (Trigonometric seasonality, Box-Cox, ARMA errors, Trend, Seasonal)
- SSA (Singular Spectrum Analysis) with grouping and reconstruction
- STR (Seasonal-Trend decomposition with Regression)
- Multi-seasonal decomposition for complex seasonal patterns
- Classical additive and multiplicative decomposition
- Robust variants with outlier handling

### Forecasting: Classical & Statistical
- ARIMA / SARIMA with Auto-ARIMA (stepwise and grid search)
- Exponential smoothing: Simple ES, Holt's linear trend, Holt-Winters, ETS framework
- BATS / TBATS for complex multi-seasonal data
- Theta method and Theta-F variants
- Naive, seasonal naive, drift, moving average, and ensemble of simple methods
- Intermittent demand: Croston's method, Syntetos-Boylan Approximation (SBA), TSB (Teunter-Syntetos-Babai)

### Forecasting: Neural Architectures
- **Temporal Fusion Transformer (TFT)**: multi-horizon attention-based model with variable selection, gating, and static covariate encoding
- **N-BEATS**: neural basis expansion for interpretable time series forecasting (trend and seasonality stacks)
- **N-HiTS**: hierarchical interpolation with multi-rate signal sampling
- **DeepAR**: autoregressive RNN with probabilistic output (Gaussian, negative binomial) for Amazon-style probabilistic forecasting
- **Simple neural forecast API**: common interface across all neural models

### State-Space Models & Kalman Filtering
- Kalman filter and Rauch-Tung-Striebel smoother
- Extended Kalman filter (EKF) for nonlinear systems
- Unscented Kalman filter (UKF) with sigma-point propagation
- Structural time series (local level, local linear trend, seasonal, cycle)
- Unobserved components models
- Dynamic linear models with time-varying parameters

### Volatility & GARCH Models
- GARCH(p,q) and EGARCH (exponential GARCH)
- FIGARCH (fractionally integrated GARCH) for long-memory volatility
- GJR-GARCH (asymmetric leverage effects)
- ARCH-LM test, Ljung-Box test for model diagnostics

### Long-Memory Processes
- ARFIMA (Autoregressive Fractionally Integrated Moving Average) estimation and simulation
- Hurst exponent estimation: R/S analysis, detrended fluctuation analysis (DFA), Whittle estimator
- Fractional differencing (fractional-d operator) with memory-preserving transforms

### Causality & Cointegration
- Granger causality testing with F-statistics and block-exogeneity (`causality::granger_causality_test`), spectral (frequency-domain) Granger causality, conditional/multivariate variants
- Transfer entropy (Shannon, Renyi, conditional, effective) with bootstrap significance testing
- Convergent cross mapping (CCM) for nonlinear causality
- Causal discovery: PC, PC-stable, PCMCI, FCI (latent confounders)
- Johansen trace cointegration test over a rolling window (`streaming::cointegration::StreamingCointegrationTester`); a simplified Engle-Granger residual-based check is used internally by the ECM regression fitting path
- `VECMModel` type exists in `var_models`, but `VECMModel::fit` is currently a placeholder that does not run the Johansen estimation procedure — see Known Issues in [TODO.md](./TODO.md)

### Vector Autoregressive (VAR) Models
- VAR(p) fitting with OLS and information criterion lag selection (AIC, BIC, HQIC)
- Impulse response functions (IRF, `VARModel::impulse_response`) and forecast error variance decomposition (FEVD, `VARModel::variance_decomposition`); bootstrap confidence bands around the IRF are not implemented
- Granger causality block-exogeneity Wald test — real and reachable via `causality::granger_causality_test` and friends. **Caveat**: the separate convenience method `VARModel::granger_causality(&self, ...)` is currently hardcoded (`f_stat = 2.5`, `p_value = 0.05` regardless of input — comment says "Would implement proper F-test for coefficient restrictions"); use the `causality` module function instead
- VECM for cointegrated systems

### Functional Data Analysis (FDA)
- Functional PCA (`dimensionality_reduction::functional_pca`), including bivariate and multilevel variants
- Dynamic time warping barycenter averaging (DBA) with medoid/mean initialization (`dimensionality_reduction::dtw`)
- A richer FDA suite (B-spline/Fourier/wavelet basis expansions, functional linear model, functional ANOVA, k-centres functional clustering) exists as source files under `src/functional/` but is not currently declared as a module in `lib.rs`, so it is not part of the public API — see Known Issues in [TODO.md](./TODO.md)

### Hierarchical Forecasting & Reconciliation
- Bottom-up, top-down (average historical proportions, PHA, TDA), and middle-out aggregation
- Optimal reconciliation: MinT (trace minimisation), WLS (weighted least squares), OLS
- Cross-temporal reconciliation for multi-frequency hierarchies
- Evaluation with hierarchical MASE and weighted MAPE

### Conformal Prediction for Time Series (not yet exposed publicly)
- Split conformal, EnbPI, adaptive/weighted conformal, and interval calibration diagnostics have source code under `src/conformal/` and `src/forecast_uncertainty/`, but neither directory is declared as a module in `lib.rs`, so none of this is reachable as `scirs2_series::...` today
- Mondrian conformal prediction does not exist anywhere in the source tree
- See Known Issues in [TODO.md](./TODO.md) for the wiring gap

### Online / Streaming Algorithms
- ADWIN (Adaptive Windowing) concept drift detector (`online_algorithms::ADWINDetector`)
- Online ARIMA with recursive-least-squares-style parameter tracking
- Streaming quantile estimation via the P² algorithm (`online_algorithms::OnlineQuantile`); a KLL sketch is not implemented
- Online anomaly detection: CUSUM, EWMA control charts, streaming isolation forest
- Reservoir sampling (uniform and weighted) and sliding window statistics

### Change Detection
- PELT (Pruned Exact Linear Time) for multiple change point detection
- Binary segmentation (greedy and exact variants)
- CUSUM (cumulative sum) control charts
- Bayesian online change point detection (BOCPD)
- Kernel-based change detection (MMD statistics)

### Anomaly Detection
- Statistical process control (SPC): Shewhart, CUSUM, EWMA charts
- Z-score and modified Z-score methods
- IQR-based detection
- Isolation forest adapted for time series
- Prediction-error-based and reconstruction-based anomaly scores
- Distance-based approaches (matrix profile, LOF)

### Pattern Analysis
- Autocorrelation (ACF) and partial autocorrelation (PACF) with confidence bands
- Cross-correlation with bootstrap confidence intervals
- Dynamic time warping (DTW) with Sakoe-Chiba and Itakura constraints
- Symbolic Aggregate approXimation (SAX), APCA, PLA, and Persist — `dimensionality_reduction::symbolic::{apply_symbolic_approximation, SymbolicMethod}`
- Time-frequency analysis: STFT, CWT (Morlet), coherence analysis (`correlation`, `features::wavelet`)
- Motif discovery and discord detection via matrix profile has source code (`TimeSeriesMotif`, `Discord`) in `src/pattern/mod.rs`, but — like `src/conformal/`, `src/forecast_uncertainty/`, `src/functional/`, and `src/change_detection/` — that directory is not declared as a module in `lib.rs`, so it is not reachable from the public API; see Known Issues in [TODO.md](./TODO.md)

### Feature Engineering (60+ features)
- Statistical: mean, variance, skewness, kurtosis, entropy, crossing rate, linearity
- Frequency domain: spectral entropy, spectral centroid, dominant frequency (`features::frequency`); a further "noise bandwidth" field exists on `SpectralAnalysisFeatures` but is only ever zero-initialized, never actually computed, in the current code
- Complexity: approximate entropy, sample entropy, permutation entropy, Lyapunov exponent estimate
- Trend: linear trend slope, Hurst exponent, CUSUM range, range/IQR ratio
- Lag-based: ACF at specified lags, PACF, partial correlation coefficients
- Automated selection: filter, wrapper (forward/backward), embedded (LASSO, tree importance)

### Regression Models for Time Series
- Distributed lag (DL) models with flexible lag structures
- Autoregressive distributed lag (ARDL) with automatic lag selection
- Error correction models (ECM) for cointegrated series
- Regression with ARIMA errors (ARIMAX / REGARIMA)

### Clustering & Classification
- Time series clustering: k-means, hierarchical, DBSCAN, spectral, and Gaussian mixture models (`clustering::ClusteringAlgorithm`)
- k-NN classification with DTW, Euclidean, and correlation-based distances; feature-based and ensemble classification
- Shapelet discovery and shapelet transform classification
- k-medoids (PAM) and HDBSCAN are not implemented as distinct algorithms; functional-data clustering (k-centres functional) exists only in the unwired `src/functional/` tree (see FDA note above)

### Ensemble & Probabilistic Forecasting
- Ensemble forecasting (`ensemble_forecast`): simple average, weighted average, stacking, median ensemble, trimmed mean, bagging, dynamic ensemble selection
- Probabilistic forecast evaluation (`energy_forecast`): pinball loss, CRPS, coverage, Winkler score, reliability diagrams, sharpness, skill score
- General-purpose point-forecast evaluation (`evaluation`): MAE, MAPE, SMAPE, MASE (incl. seasonal), RMSE, MSE, WAPE, coverage probability, Winkler score, Diebold-Mariano test
- Bootstrap/conformal prediction intervals and quantile regression forests, and log score / PIT histograms specifically, are not currently implemented anywhere in the crate

### Domain-Specific Extensions
- **Financial**: GARCH volatility, 15+ technical indicators (CCI, MFI, OBV, Parabolic SAR, RSI, MACD, Bollinger Bands, ATR)
- **Environmental**: heat wave detection, SPI drought index, growing degree days, SOI/NAO climate indices
- **Biomedical**: ECG R-peak detection, HRV analysis, EEG frequency band decomposition, EMG onset detection
- **IoT sensors**: environmental sensor fusion, GPS activity recognition, predictive maintenance scoring

### Transformations
- Box-Cox transformation with automatic lambda estimation
- Differencing (regular and seasonal), fractional differencing
- Normalization: Z-score, Min-Max, robust (median/IQR)
- Stationarity transformation pipeline with ADF/KPSS guidance

### Regime-Switching Models
- Markov-switching autoregression (MS-AR) with Hamilton filter, Kim smoother, and EM estimation (`regime::fit_msar`)
- Simple CUSUM-of-innovations structural break detection on a fitted state-space model (`state_space::detect_structural_break_ssm`)
- Threshold/self-exciting (TAR/SETAR), smooth-transition (STAR), and Bai-Perron multiple structural break tests are **not implemented** anywhere in the crate despite being listed in earlier TODO notes — see Known Issues in [TODO.md](./TODO.md)

---

## Quick Start

```toml
[dependencies]
scirs2-series = "0.6.3"
```

### ARIMA Forecasting

```rust
use scirs2_series::arima_models::arima;
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: Array1<f64> = Array1::from(vec![
        110.0, 115.0, 118.0, 122.0, 120.0, 125.0, 130.0, 128.0,
        132.0, 135.0, 140.0, 138.0, 145.0, 148.0, 152.0, 155.0,
    ]);

    let model = arima(&data, 1, 1, 1)?;
    let forecast = model.forecast_with_confidence(5, &data, 0.95)?;
    println!("5-step forecast: {:?}", forecast.point_forecast);
    println!("95% interval:    {:?} .. {:?}", forecast.lower_ci, forecast.upper_ci);
    Ok(())
}
```

### Temporal Fusion Transformer

```rust
use scirs2_series::tft::{TFT, TFTConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = TFTConfig {
        hidden_size: 8,
        n_heads: 2,
        dropout: 0.0,
        horizon: 4,
        lookback: 12,
    };
    let model = TFT::new(config);

    let x_past: Vec<Vec<f32>> = vec![vec![0.5f32]; 12];
    let x_future: Vec<Vec<f32>> = vec![vec![0.5f32]; 4];
    let forecast = model.forward(&x_past, &x_future)?;
    println!("forecast: {:?}", forecast);
    Ok(())
}
```

(There is a separate, simpler `TFTModel` in `neural_forecast::tft` used by the "simple neural forecast API"; the example above uses the standalone `scirs2_series::tft` module, which also carries the FlashAttention-for-long-lookback variant.)

### Granger Causality Test

```rust
use scirs2_series::causality::granger_causality_test;
use scirs2_core::ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![1.0f64, 1.5, 2.0, 2.5, 3.0, 3.2, 2.8, 3.5, 4.0, 3.7];
    let y = array![0.5f64, 0.8, 1.2, 1.8, 2.2, 2.6, 2.4, 2.9, 3.3, 3.1];

    let result = granger_causality_test(&x, &y, 2)?;
    println!("F-statistic: {:.4}, p-value: {:.4}", result.f_statistic, result.p_value);
    println!("Does x Granger-cause y? {}", result.p_value < 0.05);
    Ok(())
}
```

### ADWIN Concept Drift Detection

```rust
use scirs2_series::online_algorithms::ADWINDetector;

let mut detector = ADWINDetector::new(0.002); // delta parameter

for &obs in &stream_of_values {
    if detector.update(obs) {
        println!("Concept drift detected at this point!");
    }
}
```

### Hierarchical Reconciliation (MinT)

```rust
use scirs2_series::reconciliation::{SummationMatrix, MinTraceReconciliation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Node 0 is the aggregate total; nodes 1 and 2 are its two bottom-level children.
    let parents: Vec<Option<usize>> = vec![None, Some(0), Some(0)];
    let s = SummationMatrix::from_parents(&parents)?;

    let reconciled =
        MinTraceReconciliation::reconcile_mint_shrink(&base_forecasts, &s, &residuals)?;
    println!("{:?}", reconciled);
    Ok(())
}
```

Note: split/adaptive conformal prediction intervals are not shown here because that code
(`src/conformal/`, `src/forecast_uncertainty/`) is not currently wired into `lib.rs` — see
Known Issues in [TODO.md](./TODO.md).

---

## API Overview

| Module | Description |
|---|---|
| `arima_models` | ARIMA, SARIMA, Auto-ARIMA, ARIMAX |
| `ets` | ETS (Error-Trend-Seasonal) exponential smoothing framework |
| `bats` / `tbats` | BATS and TBATS multi-seasonal models |
| `theta` | Theta method and Theta-F |
| `intermittent` | Croston, SBA, TSB for intermittent demand |
| `neural_forecast` | TFT, N-BEATS, N-HiTS, DeepAR, simple API |
| `state_space` | Kalman filter, EKF, UKF, structural time series |
| `forecasting` | Naive, drift, MA, ensemble of simple methods |
| `var_models` | VAR, impulse response, variance decomposition; `VECMModel` type is present but `fit()` is a placeholder |
| `causality` | Granger causality (incl. spectral/conditional), transfer entropy, CCM, PC/PC-stable/PCMCI/FCI causal discovery |
| `streaming::cointegration` | Rolling-window Johansen trace cointegration test |
| `volatility` | GARCH, EGARCH, FIGARCH, GJR-GARCH |
| `long_memory` | ARFIMA, Hurst estimation, fractional differencing |
| `decomposition` | STL, SSA, STR, TBATS, classical |
| `features` | 60+ time series features with automated selection |
| `feature_selection` | Filter, wrapper, embedded feature selection |
| `change_point` | Unified PELT / binary segmentation / CUSUM / Bayesian-online / kernel change-point dispatch; PELT also available standalone via `detection::pelt` |
| `anomaly` | SPC charts, isolation forest, prediction-error methods |
| `streaming` | Streaming statistics, change detection, cointegration (ADWIN itself lives in `online_algorithms`) |
| `online_algorithms` | ADWIN drift detector, P² streaming quantiles, online regression |
| ~~`conformal`~~ | Not wired into `lib.rs` — unreachable (see Known Issues) |
| `hierarchical` | Hierarchical aggregation strategies; `hierarchical::reconciliation` also has its own `HierarchyMatrix`/MinT/bottom-up/ERM functions |
| `reconciliation` | MinT (shrinkage), WLS, OLS optimal reconciliation, summation-matrix builder |
| `ensemble_forecast` | Forecast combination, stacking, bagging, dynamic ensemble selection |
| `regime` | Markov-switching AR (Hamilton filter) only — TAR/SETAR/STAR are not implemented |
| `structural` | Structural *time series* models (local level, local linear trend) with Kalman decompose/forecast — not a Bai-Perron break test |
| `dimensionality_reduction` | Functional PCA, DTW + barycenter averaging, other reduction utilities |
| `clustering` | k-means, hierarchical, DBSCAN, spectral, GMM clustering; DTW/kNN/shapelet classification |
| `correlation` | ACF, PACF, CCF, DTW, coherence |
| `regression` | DL, ARDL, ECM, regression with ARIMA errors |
| `transformations` | Box-Cox, differencing, normalization, stationarity |
| `tests` | Unit root and stationarity tests (ADF, KPSS, PP) |
| `evaluation` | MAE, MAPE, SMAPE, MASE, RMSE, WAPE, coverage, Winkler score, Diebold-Mariano |
| `energy_forecast` | Pinball loss, CRPS, reliability diagrams, sharpness, skill score (in addition to its energy-domain forecasting) |
| `financial` | Technical indicators, GARCH, financial metrics |
| `environmental` | Climate indices, drought, weather analysis |
| `biomedical` | ECG, EEG, EMG signal analysis |
| `iot_sensors` | Sensor fusion, predictive maintenance |

---

## Feature Flags

| Flag | Description |
|---|---|
| `simd` | SIMD-accelerated operations via `scirs2-core` |
| `serde` | Serialization support |
| `wasm` | WebAssembly bindings (pulls in `serde`) |
| `python` | Python interop layer (via `scirs2-core/python`) |
| `r` | R integration (`r_integration` module) |

`default = []` — no feature is enabled by default. There is no `parallel` feature; the crate
does not depend on `rayon`.

---

## Testing

`grep -rn "todo!()\|unimplemented!()" src/` returns 0 matches — but see [TODO.md](./TODO.md)
"Known Issues" for several public API methods that silently return placeholder/fabricated
results instead of panicking or erroring, which a `todo!()` grep cannot catch.

Freshly measured 2026-07-15:

```bash
cargo nextest run -p scirs2-series               # default features: 1752 tests run, 1752 passed, 1 skipped
cargo nextest run -p scirs2-series --all-features # all-features:     1805 tests run, 1805 passed, 1 skipped
```

Both runs: 0 failed. (The 1 skipped test is `#[ignore]`d in both modes.)

---

## Links

- [SciRS2 project](https://github.com/cool-japan/scirs)
- [docs.rs](https://docs.rs/scirs2-series)
- [crates.io](https://crates.io/crates/scirs2-series)
- [TODO.md](./TODO.md)

## License

Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Authors

COOLJAPAN OU (Team KitaSan)
