//! Time series feature extraction
//!
//! This module provides functions for extracting statistical and
//! spectral features from time series data. These features can be
//! used for signal characterization, classification, and similarity analysis.

// Re-export modules
pub mod activity;
pub mod batch;
pub mod entropy;
pub mod options;
pub mod peaks;
pub mod spectral;
pub mod statistical;
pub mod trend;
pub mod zero_crossing;

// Re-export main types and functions
pub use activity::activity_recognition_features;
pub use batch::{extract_features, extract_features_batch};
pub use options::FeatureOptions;

// Re-export key functionality from submodules
pub use entropy::extract_entropy_features;
pub use entropy::{
    calculate_approximate_entropy, calculate_permutation_entropy, calculate_sample_entropy,
    calculate_shannon_entropy,
};
pub use peaks::extract_peak_features;
pub use spectral::extract_spectral_features;
pub use statistical::extract_statistical_features;
pub use statistical::{calculate_kurtosis, calculate_quantile, calculate_skewness, calculate_std};
pub use trend::extract_trend_features;
pub use trend::{linear_regression, quadratic_regression};
pub use zero_crossing::extract_zero_crossing_features;
