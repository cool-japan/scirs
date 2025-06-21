//! Time series analysis module for SciRS2
//!
//! This module provides functionality for time series analysis, including:
//! - Time series decomposition (trend, seasonality, residual)
//!   - Classical decomposition (additive and multiplicative)
//!   - STL decomposition
//!   - Singular Spectrum Analysis (SSA)
//!   - Multiple seasonal decomposition (MSTL)
//!   - TBATS (Trigonometric, Box-Cox, ARMA, Trend, Seasonal)
//!   - STR (Seasonal-Trend decomposition using Regression)
//!   - Exponential smoothing decomposition
//! - Change point detection
//!   - PELT (Pruned Exact Linear Time) algorithm
//!   - Binary segmentation
//!   - CUSUM methods
//!   - Bayesian online change point detection
//!   - Kernel-based change detection
//! - Anomaly detection
//!   - Statistical process control (SPC)
//!   - Isolation forest for time series
//!   - Z-score and modified Z-score methods
//!   - Interquartile range (IQR) detection
//!   - Distance-based and prediction-based approaches
//! - Automatic pattern detection
//!   - Period detection using ACF, FFT, and wavelets
//!   - Automatic seasonal decomposition with period detection
//! - Advanced trend analysis
//!   - Non-linear trend estimation using splines
//!   - Cubic splines, B-splines, and P-splines
//!   - Robust trend filtering with confidence intervals
//! - State-space models
//!   - Kalman filtering and smoothing
//!   - Structural time series models
//!   - Dynamic linear models
//!   - Unobserved components models
//! - Causality testing and relationship analysis
//!   - Granger causality testing with F-statistics and p-values
//!   - Transfer entropy measures with bootstrap significance testing
//!   - Convergent cross mapping for nonlinear causality detection
//!   - Causal impact analysis for intervention assessment
//! - Correlation and relationship analysis
//!   - Cross-correlation functions with confidence intervals
//!   - Dynamic time warping with multiple constraint types
//!   - Time-frequency analysis (STFT, CWT, Morlet wavelets)
//!   - Coherence analysis for frequency domain relationships
//! - Time series clustering and classification
//!   - K-means, hierarchical, and DBSCAN clustering algorithms
//!   - Multiple distance measures (DTW, Euclidean, correlation-based)
//!   - k-NN classification with DTW and other distance functions
//!   - Shapelet discovery and shapelet-based classification
//! - Vector autoregressive models
//!   - VAR model fitting and prediction
//!   - Impulse response functions
//!   - Variance decomposition
//!   - Granger causality testing
//!   - VECM for cointegrated series
//!   - Automatic order selection
//! - ARIMA models with enhanced functionality
//!   - Automatic order selection with multiple criteria
//!   - Stepwise and grid search optimization
//!   - Seasonal ARIMA (SARIMA) support
//!   - Model diagnostics and information criteria
//! - Time series transformations
//!   - Box-Cox transformations with automatic lambda estimation
//!   - Differencing and seasonal differencing
//!   - Stationarity tests (ADF, KPSS)
//!   - Normalization and scaling (Z-score, Min-Max, Robust)
//!   - Detrending and stationarity transformations
//! - Dimensionality reduction for time series
//!   - Principal Component Analysis (PCA) for time series
//!   - Functional PCA for functional time series data
//!   - Dynamic Time Warping barycenter averaging
//!   - Symbolic approximation methods (SAX, APCA, PLA)
//! - Time series regression models
//!   - Distributed lag models (DL) with flexible lag structures
//!   - Autoregressive distributed lag (ARDL) models with automatic lag selection
//!   - Error correction models (ECM) for cointegrated series
//!   - Regression with ARIMA errors for correlated residuals
//! - Forecasting methods (ARIMA, exponential smoothing)
//!   - Automatic model selection
//!   - Seasonal and non-seasonal models
//! - Feature extraction for time series
//! - Feature selection methods for time series
//!   - Filter methods (correlation, variance, mutual information, statistical tests)
//!   - Wrapper methods (forward selection, backward elimination, recursive elimination)
//!   - Embedded methods (LASSO, Ridge, tree-based importance)
//!   - Time series specific methods (lag-based, seasonal, cross-correlation, Granger causality)
//! - Utility functions for time series operations

#![warn(missing_docs)]

pub mod anomaly;
pub mod arima_models;
pub mod causality;
pub mod change_point;
pub mod clustering;
pub mod correlation;
pub mod decomposition; // Directory-based modular structure
pub mod decomposition_compat; // For backward compatibility
pub mod detection;
pub mod diagnostics;
pub mod dimensionality_reduction;
pub mod enhanced_arma;
pub mod error;
pub mod feature_selection;
pub mod features;
pub mod forecasting;
pub mod optimization;
pub mod regression;
pub mod sarima_models;
pub mod state_space;
pub mod tests;
pub mod transformations;
pub mod trends;
pub mod utils;
pub mod validation;
pub mod var_models;
