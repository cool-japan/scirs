#![allow(deprecated)]
#![allow(clippy::all)]
#![allow(dead_code)]
#![allow(unreachable_patterns)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(private_interfaces)]
//! # SciRS2 Series - Time Series Analysis
//!
//! **scirs2-series** provides comprehensive time series analysis capabilities,
//! offering decomposition, forecasting (ARIMA, VAR, Prophet-style), anomaly detection,
//! change point detection, and advanced methods with parallel processing and streaming support.
//!
//! ## 🎯 Key Features
//!
//! - **Decomposition**: STL, classical, SSA, MSTL, TBATS for trend/seasonality extraction
//! - **Forecasting**: ARIMA, SARIMA, VAR, VECM, exponential smoothing
//! - **Anomaly Detection**: Statistical, isolation forest, distance-based methods
//! - **Change Point Detection**: PELT, binary segmentation, CUSUM, Bayesian online
//! - **Causality Testing**: Granger causality, transfer entropy, convergent cross mapping
//! - **Clustering**: Time series k-means, hierarchical, DTW-based clustering
//! - **State-Space Models**: Kalman filtering, structural models, dynamic linear models
//! - **Transformations**: Box-Cox, differencing, stationarity tests (ADF, KPSS)
//!
//! ## 📦 Module Overview
//!
//! | SciRS2 Module | Python Equivalent | Description |
//! |---------------|-------------------|-------------|
//! | `decomposition` | `statsmodels.tsa.seasonal.STL` | Time series decomposition |
//! | `arima` | `statsmodels.tsa.arima.model.ARIMA` | ARIMA forecasting |
//! | `var` | `statsmodels.tsa.vector_ar.var_model.VAR` | Vector autoregression |
//! | `anomaly` | - | Anomaly/outlier detection |
//! | `changepoint` | `ruptures` | Change point detection |
//! | `causality` | `statsmodels.tsa.stattools.grangercausalitytests` | Granger causality testing |
//!
//! ## 🚀 Quick Start
//!
//! ```toml
//! [dependencies]
//! scirs2-series = "0.1.0-rc.1"
//! ```
//!
//! ```rust,no_run
//! use scirs2_series::decomposition::stl::{stl_decomposition, STLOptions};
//! use scirs2_core::ndarray::array;
//!
//! // STL decomposition
//! let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]; // Seasonal data
//! let options = STLOptions::default();
//! let result = stl_decomposition(&data, 4, &options).unwrap();
//! // result.trend, result.seasonal, result.residual
//! ```
//!
//! ## 🔒 Version: 0.1.0-rc.1 (October 03, 2025)
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
//! - Environmental and climate data analysis
//!   - Temperature analysis (heat waves, growing degree days, climate normals)
//!   - Precipitation analysis (drought detection, SPI, rainfall classification)
//!   - Atmospheric analysis (storm detection, wind power, wind rose statistics)
//!   - Climate indices (SOI, NAO, PDSI)
//!   - Environmental stress index calculation
//! - Biomedical signal processing
//!   - ECG analysis (R-peak detection, HRV, arrhythmia detection)
//!   - EEG analysis (seizure detection, frequency bands, connectivity)
//!   - EMG analysis (muscle activation, fatigue detection, onset detection)
//!   - Cross-signal synchronization and health assessment
//! - IoT sensor data analysis
//!   - Environmental sensors (temperature, humidity, pressure, light)
//!   - Motion sensors (accelerometer, GPS, activity recognition)
//!   - Data quality assessment and sensor malfunction detection
//!   - Predictive maintenance and system health monitoring
//! - Comprehensive visualization capabilities
//!   - Interactive time series plotting with zoom and pan
//!   - Forecasting visualization with confidence intervals
//!   - Decomposition result visualization (trend, seasonal, residual components)
//!   - Multi-series plotting and comparison
//!   - Seasonal pattern visualization
//!   - Anomaly and change point highlighting
//!   - Dashboard generation utilities
//!   - Export capabilities (PNG, SVG, HTML)
//! - Quantum-inspired time series analysis
//!   - Quantum attention mechanisms using superposition principles
//!   - Variational quantum circuits for pattern recognition
//!   - Quantum kernel methods for similarity measures
//!   - Quantum annealing optimization for hyperparameter tuning
//!   - Quantum state representations for complex time series patterns
//! - Advanced training methodologies
//!   - Model-Agnostic Meta-Learning (MAML) for few-shot forecasting
//!   - Neural Ordinary Differential Equations (NODEs) for continuous-time modeling
//!   - Variational autoencoders with uncertainty quantification
//!   - Bayesian neural networks for probabilistic forecasting
//!   - Gradient-based meta-learning optimization techniques
//! - Out-of-core processing for massive datasets
//!   - Chunked processing with configurable chunk sizes and overlap
//!   - Memory-mapped file I/O for efficient disk access
//!   - Streaming statistics computation (mean, variance, quantiles)
//!   - Parallel processing of chunks with progress tracking
//!   - CSV and binary file format support
//! - Distributed computing support for large-scale time series processing
//!   - Multi-node cluster coordination and task distribution
//!   - Load balancing and fault tolerance mechanisms
//!   - Distributed forecasting, feature extraction, and anomaly detection
//!   - Task dependency management and priority scheduling
//!   - Real-time cluster monitoring and performance metrics
//! - Advanced Fusion Intelligence - Next-Generation AI Systems
//!   - Quantum-Neuromorphic fusion processors combining quantum and biological computing
//!   - Meta-learning systems that learn how to learn from time series patterns
//!   - Self-evolving neural architectures that redesign themselves autonomously
//!   - Consciousness-inspired computing with attention and self-awareness mechanisms
//!   - Temporal hypercomputing for multi-dimensional time analysis
//!   - Autonomous mathematical discovery and pattern recognition
//!   - Advanced-predictive analytics for impossible event prediction
//!   - Distributed quantum networks with planet-scale processing capabilities
//! - Utility functions for time series operations

#![warn(missing_docs)]

pub mod advanced_advanced_visualization;
pub mod advanced_fusion_intelligence;
pub mod advanced_training;
pub mod anomaly;
pub mod arima_models;
pub mod biomedical;
pub mod causality;
pub mod change_point;
#[cfg(feature = "wasm")]
pub mod cloud_deployment;
pub mod clustering;
pub mod correlation;
pub mod decomposition; // Directory-based modular structure
pub mod decomposition_compat; // For backward compatibility
pub mod detection;
pub mod diagnostics;
pub mod dimensionality_reduction;
pub mod distributed;
pub mod enhanced_arma;
pub mod ensemble_automl;
pub mod environmental;
pub mod error;
pub mod feature_selection;
pub mod features;
pub mod financial;
pub mod financial_advanced;
pub mod forecasting;
pub mod gpu_acceleration;
pub mod iot_sensors;
pub mod neural_forecasting;
pub mod neuromorphic_computing;
pub mod optimization;
pub mod out_of_core;
pub mod quantum_forecasting;
pub mod regression;
pub mod sarima_models;
pub mod state_space;
pub mod streaming;
pub mod tests;
pub mod transformations;
pub mod trends;
pub mod utils;
pub mod validation;
pub mod var_models;
pub mod visualization;

// Optional WASM bindings for browser-based time series analysis
#[cfg(feature = "wasm")]
pub mod wasm_bindings;

// Optional Python bindings for seamless pandas/statsmodels integration
#[cfg(feature = "python")]
pub mod python_bindings;

// Optional R integration for R ecosystem compatibility
#[cfg(feature = "r")]
pub mod r_integration;
