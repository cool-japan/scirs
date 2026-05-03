# scirs2-series

[![crates.io](https://img.shields.io/crates/v/scirs2-series.svg)](https://crates.io/crates/scirs2-series)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-series)](https://docs.rs/scirs2-series)
[![Version](https://img.shields.io/badge/version-0.4.3-green)]()

**Production-ready time series analysis for Rust** — part of the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.

`scirs2-series` is a comprehensive time series library covering classical econometric models through state-of-the-art deep learning forecasters. Version 0.4.3 carries forward the full neural architecture forecaster suite (TFT, N-BEATS, N-HiTS, DeepAR), streaming/online algorithms, conformal prediction intervals, long-memory processes, intermittent demand forecasting, and hierarchical reconciliation. The Wave 3 stub-check sweep also restored the public `PatternDetector`, `STLDecomposer`, `AnomalyDetector`, and `PELTDetector` types, and `SarimaModel.forecast(steps)` was reinstated alongside the `SARIMAModel` type alias.

---

## Overview

Time series problems span a wide spectrum: univariate forecasting with uncertainty quantification, multivariate causal modelling, streaming anomaly detection, hierarchical forecasting across organizational hierarchies, regime detection, and functional data analysis. `scirs2-series` covers all of these in a unified, type-safe API.

Key design goals:

- **Breadth**: classical (ARIMA, ETS) through neural (TFT, N-BEATS, DeepAR) through streaming (online ARIMA, ADWIN)
- **Uncertainty quantification**: prediction intervals via conformal prediction and probabilistic models
- **Ecosystem coherence**: built on `scirs2-core` abstractions; no C/Fortran dependencies
- **Performance**: parallel processing with Rayon, SIMD acceleration via `scirs2-core`

---

## Feature List (v0.4.3)

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
- Granger causality testing with F-statistics and block-exogeneity
- Transfer entropy with bootstrap significance testing
- Convergent cross mapping (CCM) for nonlinear causality
- Cointegration: Engle-Granger two-step, Johansen trace and max-eigenvalue tests
- Vector Error Correction Models (VECM) with cointegration rank selection

### Vector Autoregressive (VAR) Models
- VAR(p) fitting with OLS and information criterion lag selection (AIC, BIC, HQIC)
- Impulse response functions (IRF) with bootstrap confidence bands
- Forecast error variance decomposition (FEVD)
- Granger causality block-exogeneity Wald test
- VECM for cointegrated systems

### Functional Data Analysis (FDA)
- Functional PCA (FPCA) with PACE algorithm for sparse/irregular data
- B-spline and Fourier basis expansions for functional data representation
- Functional linear model (scalar-on-function regression)
- Functional clustering (k-centres, hierarchical functional)
- Dynamic time warping barycenter averaging (DBA)

### Hierarchical Forecasting & Reconciliation
- Bottom-up, top-down (average historical proportions, PHA, TDA), and middle-out aggregation
- Optimal reconciliation: MinT (trace minimisation), WLS (weighted least squares), OLS
- Cross-temporal reconciliation for multi-frequency hierarchies
- Evaluation with hierarchical MASE and weighted MAPE

### Conformal Prediction for Time Series
- Split conformal prediction intervals (exchangeable and time-series-adapted variants)
- Adaptive conformal inference (ACI) for online coverage guarantees
- Mondrian conformal prediction for conditional coverage
- Calibration diagnostics and coverage plots

### Online / Streaming Algorithms
- ADWIN (Adaptive Windowing) concept drift detector
- Online ARIMA with recursive least squares parameter tracking
- Streaming mean, variance, quantile estimation (P² algorithm, KLL sketch)
- Online anomaly detection: CUSUM, EWMA control charts, streaming isolation forest
- Reservoir sampling and sliding window statistics

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
- Motif discovery and discord detection via matrix profile
- Symbolic Aggregate approXimation (SAX), APCA, PLA
- Time-frequency analysis: STFT, CWT (Morlet), coherence analysis

### Feature Engineering (60+ features)
- Statistical: mean, variance, skewness, kurtosis, entropy, crossing rate, linearity
- Frequency domain: spectral entropy, spectral centroid, dominant frequency, bandwidth
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
- Time series k-means, k-medoids (PAM), hierarchical clustering
- DBSCAN and HDBSCAN with DTW distance
- k-NN classification with DTW, Euclidean, correlation-based distances
- Shapelet discovery and shapelet transform classification
- Functional data clustering (k-centres functional)

### Ensemble & Probabilistic Forecasting
- Ensemble forecasting: simple average, weighted average, stacking
- Prediction interval methods: bootstrap, conformal, quantile regression forests
- Probabilistic forecast evaluation: CRPS, log score, reliability diagrams, PIT histograms

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
- Markov-switching autoregression (MS-AR) with Hamilton filter
- Threshold autoregressive (TAR) and SETAR models
- Smooth transition autoregressive (STAR) models
- Structural break detection (Bai-Perron multiple break test)

---

## Quick Start

```toml
[dependencies]
scirs2-series = "0.4.3"
```

### ARIMA Forecasting

```rust
use scirs2_series::arima_models::AutoArima;
use scirs2_core::ndarray::Array1;

let data: Array1<f64> = Array1::from(vec![
    110.0, 115.0, 118.0, 122.0, 120.0, 125.0, 130.0, 128.0,
    132.0, 135.0, 140.0, 138.0, 145.0, 148.0, 152.0, 155.0,
]);

let model = AutoArima::fit(&data, None).unwrap();
let forecast = model.predict(5).unwrap();
println!("5-step forecast: {:?}", forecast.values);
println!("95% intervals:   {:?}", forecast.intervals_95);
```

### Temporal Fusion Transformer

```rust
use scirs2_series::neural_forecast::{TFT, TFTConfig};

let config = TFTConfig {
    hidden_size: 64,
    num_heads: 4,
    num_encoder_steps: 24,
    forecast_horizon: 12,
    ..Default::default()
};

let mut model = TFT::new(config);
model.fit(&train_data, &covariates, 100).unwrap();
let forecasts = model.predict(&test_context, &future_covariates).unwrap();
```

### Granger Causality Test

```rust
use scirs2_series::causality::granger_causality;
use scirs2_core::ndarray::array;

let x = array![1.0f64, 1.5, 2.0, 2.5, 3.0, 3.2, 2.8, 3.5, 4.0, 3.7];
let y = array![0.5f64, 0.8, 1.2, 1.8, 2.2, 2.6, 2.4, 2.9, 3.3, 3.1];

let result = granger_causality(&x.view(), &y.view(), 2).unwrap();
println!("F-statistic: {:.4}, p-value: {:.4}", result.f_stat, result.p_value);
println!("Does x Granger-cause y? {}", result.p_value < 0.05);
```

### ADWIN Concept Drift Detection

```rust
use scirs2_series::streaming::adwin::ADWIN;

let mut detector = ADWIN::new(0.002);  // delta parameter

for &obs in &stream_of_values {
    if detector.update(obs) {
        println!("Concept drift detected at this point!");
    }
}
```

### Hierarchical Reconciliation

```rust
use scirs2_series::reconciliation::{MinTReconciler, HierarchyMatrix};

let hierarchy = HierarchyMatrix::from_summing_matrix(&s_matrix);
let reconciler = MinTReconciler::sample_covariance(hierarchy);

let reconciled = reconciler.reconcile(&base_forecasts, &residual_matrix).unwrap();
```

### Conformal Prediction Intervals

```rust
use scirs2_series::conformal::SplitConformalForecaster;

let mut cp = SplitConformalForecaster::new(0.90);  // 90% coverage target
cp.calibrate(&calibration_residuals);

let interval = cp.predict_interval(&point_forecast);
println!("[{:.2}, {:.2}]", interval.lower, interval.upper);
```

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
| `var_models` | VAR, VECM, impulse response, variance decomposition |
| `causality` | Granger causality, transfer entropy, CCM |
| `cointegration` | Engle-Granger, Johansen tests |
| `volatility` | GARCH, EGARCH, FIGARCH, GJR-GARCH |
| `long_memory` | ARFIMA, Hurst estimation, fractional differencing |
| `decomposition` | STL, SSA, STR, TBATS, classical |
| `features` | 60+ time series features with automated selection |
| `feature_selection` | Filter, wrapper, embedded feature selection |
| `change_detection` | PELT, binary segmentation, BOCPD, CUSUM |
| `anomaly` | SPC charts, isolation forest, prediction-error methods |
| `streaming` | ADWIN, online ARIMA, streaming statistics |
| `conformal` | Split conformal, ACI, Mondrian conformal |
| `hierarchical` | Hierarchical aggregation strategies |
| `reconciliation` | MinT, WLS, OLS optimal reconciliation |
| `ensemble_forecast` | Forecast combination and stacking |
| `regime` | Markov-switching AR, TAR, SETAR, STAR |
| `structural` | Structural break detection (Bai-Perron) |
| `functional` | FPCA, functional linear model, FDA utilities |
| `clustering` | k-means/medoids, DBSCAN, shapelet classification |
| `correlation` | ACF, PACF, CCF, DTW, coherence |
| `regression` | DL, ARDL, ECM, regression with ARIMA errors |
| `transformations` | Box-Cox, differencing, normalization, stationarity |
| `tests` | Unit root and stationarity tests (ADF, KPSS, PP) |
| `evaluation` | MASE, SMAPE, CRPS, coverage, PIT, reliability |
| `financial` | Technical indicators, GARCH, financial metrics |
| `environmental` | Climate indices, drought, weather analysis |
| `biomedical` | ECG, EEG, EMG signal analysis |
| `iot_sensors` | Sensor fusion, predictive maintenance |

---

## Feature Flags

| Flag | Description |
|---|---|
| `parallel` | Rayon parallel computation for large datasets |
| `simd` | SIMD-accelerated operations via `scirs2-core` |
| `serde` | Serialization support |
| `wasm` | WebAssembly bindings |
| `python` | Python interop layer |

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
