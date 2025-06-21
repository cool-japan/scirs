# scirs2-series TODO

This module provides time series analysis functionality similar to the time series components in pandas and statsmodels.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] Decomposition module
  - [x] Classical decomposition (additive and multiplicative)
  - [x] STL decomposition
  - [x] Seasonal decomposition using moving averages
  - [x] Trend extraction

- [x] Forecasting module
  - [x] Exponential smoothing models
  - [x] ARIMA/SARIMA models
  - [x] Simple forecasting methods (naive, mean, drift)
  - [x] Prophet-like API for complex time series

- [x] Features module
  - [x] Time series feature extraction
  - [x] Autocorrelation features
  - [x] Statistical features
  - [x] Frequency domain features

- [x] Utilities
  - [x] Resampling and frequency conversion
  - [x] Missing value interpolation
  - [x] Outlier detection
  - [x] Date manipulation helpers

## Time Series Decomposition

- [x] Enhanced decomposition methods
  - [ ] Robust decomposition variants
  - [x] Singular Spectrum Analysis (SSA)
  - [x] TBATS decomposition (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and Seasonal components)
  - [x] MSTL (Multiple Seasonal-Trend decomposition using LOESS)
  - [x] STR (Seasonal-Trend decomposition using Regularization)
- [ ] Multi-seasonal decomposition
  - [ ] Multiple nested seasonal patterns
  - [ ] Complex seasonality identification
  - [ ] Automatic period detection
  - [ ] Flexible seasonal representation
- [x] Non-linear trend components
  - [x] Spline-based trend estimation
  - [x] Robust trend filtering
  - [x] Piecewise trends with breakpoints
  - [x] Trend confidence intervals

## Statistical Modeling

- [ ] Autoregressive models
  - [ ] AR model improvements
  - [ ] MA model enhancements
  - [ ] ARMA model optimization
  - [ ] ARIMA/SARIMA models with automatic order selection
  - [ ] ARIMAX with exogenous variables
- [x] State-space models
  - [x] Structural time series models
  - [x] Dynamic linear models (DLM)
  - [x] Unobserved components models
  - [x] Kalman filtering and smoothing
- [ ] VAR family models
  - [ ] Vector Autoregressive (VAR) models
  - [ ] Vector Error Correction Models (VECM)
  - [ ] Structural VAR models
  - [ ] VARMAX with exogenous variables
- [ ] Markov-switching models
  - [ ] Regime-switching AR models
  - [ ] Hidden Markov Models (HMM)
  - [ ] Threshold autoregressive models (TAR)
  - [ ] Smooth transition autoregressive models (STAR)

## Advanced Forecasting Methods

- [ ] Exponential smoothing extensions
  - [ ] State space exponential smoothing (ETS)
  - [ ] Damped trend methods
  - [ ] Multiple seasonal Holt-Winters
  - [ ] Robust exponential smoothing
- [ ] Neural forecasting models
  - [ ] Temporal convolutional networks
  - [ ] RNN/LSTM-based forecasting
  - [ ] DeepAR-like models
  - [ ] Transformer-based time series models
  - [ ] N-BEATS implementation
- [ ] Probabilistic forecasting
  - [ ] Forecast distribution estimation
  - [ ] Quantile forecasting
  - [ ] Conformal prediction intervals
  - [ ] Ensemble forecasting methods
- [ ] Hierarchical forecasting
  - [ ] Bottom-up approaches
  - [ ] Top-down approaches
  - [ ] Middle-out methods
  - [ ] Optimal reconciliation

## Feature Engineering and Analysis

- [ ] Time domain features
  - [x] Expanded statistical features
  - [x] Window-based aggregations
  - [x] Entropy measures
  - [x] Complexity measures (approximate entropy, sample entropy)
  - [x] Turning points analysis
- [x] Frequency domain features
  - [x] Spectral analysis utilities
  - [x] Periodogram enhancements
  - [x] Wavelet-based features
  - [x] Hilbert-Huang transform (EMD)
- [x] Temporal pattern mining
  - [x] Motif discovery
  - [x] Shapelets extraction
  - [x] Symbolic representations (SAX)
  - [x] Time series discord detection
- [x] Feature selection
  - [x] Filter methods for time series
  - [x] Wrapper methods
  - [x] Feature importance calculation
  - [x] Mutual information criteria

## Time Series Transformations

- [x] Stationarity transformations
  - [x] Box-Cox transformations
  - [x] Differencing operations
  - [x] Trend and seasonality removal
  - [x] Stationarity tests (ADF, KPSS)
- [x] Normalization and scaling
  - [x] Z-score normalization
  - [x] Min-max scaling
  - [x] Robust scaling
  - [x] Adaptive normalization
- [x] Dimensionality reduction
  - [x] PCA for time series
  - [x] Functional PCA
  - [x] Dynamic time warping barycenter averaging
  - [x] Symbolic approximation

## Change Point Detection and Anomaly Detection

- [x] Change point detection
  - [x] PELT algorithm
  - [x] Binary segmentation
  - [x] Bayesian online changepoint detection
  - [x] CUSUM methods
  - [x] Kernel-based change detection
- [x] Anomaly detection
  - [x] Statistical process control
  - [x] Isolation forest for time series
  - [x] One-class SVM for time series (simplified implementation)
  - [x] Distance-based approaches
  - [x] Prediction-based approaches
  - [x] Z-score and modified Z-score methods
  - [x] Interquartile range (IQR) detection
- [ ] Advanced detection utilities
  - [ ] Multi-dimensional change detection
  - [ ] Group anomaly detection
  - [ ] Contextual anomaly detection
  - [ ] Seasonal-aware anomaly detection

## Causality and Relationship Analysis

- [ ] Causality testing
  - [ ] Granger causality testing
  - [ ] Transfer entropy measures
  - [ ] Convergent cross mapping
  - [ ] Causal impact analysis
- [ ] Time series regression
  - [ ] Distributed lag models
  - [ ] Autoregressive distributed lag (ARDL)
  - [ ] Error correction models
  - [ ] Regression with ARIMA errors
- [ ] Correlation analysis
  - [ ] Cross-correlation functions
  - [ ] Dynamic time warping
  - [ ] Time-frequency analysis
  - [ ] Coherence analysis
- [ ] Clustering and classification
  - [ ] Time series clustering algorithms
  - [ ] Distance measures for time series
  - [ ] Time series classification methods
  - [ ] Shape-based clustering

## Performance and Optimization

- [ ] Parallel implementation
  - [ ] Parallel algorithm variants
  - [ ] Multi-threaded decomposition
  - [ ] Parallelized forecasting
  - [ ] Distributed time series processing
- [ ] Memory optimization
  - [ ] Out-of-core processing for large time series
  - [ ] Sparse representation for irregular series
  - [ ] Memory-efficient algorithms
  - [ ] Streaming implementations
- [ ] Computational efficiency
  - [ ] Fast implementations of key algorithms
  - [ ] Approximate methods for large-scale data
  - [ ] Early stopping criteria
  - [ ] Incremental computation support

## Integration and Interoperability

- [ ] Integration with ML ecosystem
  - [ ] Feature pipelines for ML models
  - [ ] Cross-validation utilities for time series
  - [ ] Model selection frameworks
  - [ ] Integration with scirs2-neural
- [ ] Data handling improvements
  - [ ] Enhanced datetime indexing
  - [ ] Time zone handling
  - [ ] Irregular time series support
  - [ ] Event-based time series
- [ ] Visualization utilities
  - [ ] Time series plotting functions
  - [ ] Decomposition visualization
  - [ ] Forecast visualization with uncertainty
  - [ ] Interactive visualization helpers

## Domain-Specific Extensions

- [ ] Financial time series
  - [ ] Technical indicators
  - [ ] Volatility modeling (GARCH family)
  - [ ] Financial forecasting models
  - [ ] Risk metrics calculation
- [ ] Environmental time series
  - [ ] Weather data analysis
  - [ ] Seasonal adjustment for climate data
  - [ ] Extreme value analysis
  - [ ] Spatial-temporal modeling
- [ ] Biomedical time series
  - [ ] Physiological signal processing
  - [ ] Wearable device data analysis
  - [ ] Clinical time series utilities
  - [ ] Epidemic curve analysis

## Testing and Quality Assurance

- [x] Unit tests for basic functionality
- [ ] Comprehensive test coverage
  - [ ] Function-specific tests
  - [ ] Integration tests
  - [ ] Performance regression tests
- [ ] Benchmarks against reference implementations
  - [ ] Accuracy benchmarks
  - [ ] Speed benchmarks
  - [ ] Memory usage benchmarks
- [ ] Edge case handling
  - [ ] Missing data robustness
  - [ ] Irregular time series handling
  - [ ] Extreme value handling
  - [ ] Various data frequencies

## Documentation and Examples

- [x] Basic API documentation
- [ ] Comprehensive documentation
  - [ ] Mathematical background
  - [ ] Implementation details
  - [ ] Performance considerations
  - [ ] Algorithm selection guides
- [ ] Tutorial notebooks
  - [ ] Basic time series analysis
  - [ ] Forecasting workflows
  - [ ] Feature engineering examples
  - [ ] Advanced modeling techniques
- [ ] Domain-specific examples
  - [ ] Financial data analysis
  - [ ] IoT data examples
  - [ ] Economic time series
  - [ ] Environmental monitoring

## Long-term Goals

- [ ] Feature parity with pandas/statsmodels time series functionality
- [ ] Real-time streaming time series analysis
- [ ] Integration with deep learning models
- [ ] High-performance implementation for large-scale data
- [ ] Domain-specific toolkits
- [ ] Symbolic mathematics for model specification
- [ ] Automated model selection and hyperparameter tuning