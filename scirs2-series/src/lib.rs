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
//! - Forecasting methods (ARIMA, exponential smoothing)
//!   - Automatic model selection
//!   - Seasonal and non-seasonal models
//! - Feature extraction for time series
//! - Utility functions for time series operations

#![warn(missing_docs)]

pub mod arima_models;
pub mod decomposition; // Directory-based modular structure
pub mod decomposition_compat; // For backward compatibility
pub mod detection;
pub mod diagnostics;
pub mod error;
pub mod features;
pub mod forecasting;
pub mod optimization;
pub mod sarima_models;
pub mod state_space;
pub mod tests;
pub mod trends;
pub mod utils;
pub mod validation;
pub mod var_models;
