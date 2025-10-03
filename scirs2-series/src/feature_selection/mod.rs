//! Time series feature selection methods
//!
//! This module provides comprehensive feature selection methods specifically designed for time series data.
//! It includes filter methods, wrapper methods, embedded methods, and time series specific approaches.

use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

// Re-export submodules
pub mod embedded;
pub mod filter;
pub mod selector;
pub mod time_series;
pub mod wrapper;

#[cfg(test)]
mod tests;

// Re-export main types and structs
pub use embedded::EmbeddedMethods;
pub use filter::FilterMethods;
pub use selector::FeatureSelector;
pub use time_series::TimeSeriesMethods;
pub use wrapper::WrapperMethods;

/// Feature selection result containing selected features and their scores
#[derive(Debug, Clone)]
pub struct FeatureSelectionResult {
    /// Indices of selected features
    pub selected_features: Vec<usize>,
    /// Feature scores (higher is better)
    pub feature_scores: Array1<f64>,
    /// Selection method used
    pub method: String,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

/// Configuration for feature selection methods
#[derive(Debug, Clone)]
pub struct FeatureSelectionConfig {
    /// Number of features to select (None = automatic)
    pub n_features: Option<usize>,
    /// Scoring method for wrapper methods
    pub scoring_method: ScoringMethod,
    /// Cross-validation folds for wrapper methods
    pub cv_folds: usize,
    /// Significance level for statistical tests
    pub alpha: f64,
    /// Minimum correlation threshold for filter methods
    pub correlation_threshold: f64,
    /// Minimum variance threshold for variance-based filtering
    pub variance_threshold: f64,
    /// Maximum number of iterations for wrapper methods
    pub max_iterations: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Regularization parameter for embedded methods
    pub regularization_alpha: f64,
    /// Maximum lag for time series specific methods
    pub max_lag: usize,
    /// Seasonal period for seasonal feature selection
    pub seasonal_period: Option<usize>,
}

impl Default for FeatureSelectionConfig {
    fn default() -> Self {
        Self {
            n_features: None,
            scoring_method: ScoringMethod::MeanSquaredError,
            cv_folds: 5,
            alpha: 0.05,
            correlation_threshold: 0.1,
            variance_threshold: 0.01,
            max_iterations: 100,
            random_seed: None,
            regularization_alpha: 1.0,
            max_lag: 10,
            seasonal_period: None,
        }
    }
}

/// Scoring methods for feature selection
#[derive(Debug, Clone)]
pub enum ScoringMethod {
    /// Mean squared error
    MeanSquaredError,
    /// Mean absolute error
    MeanAbsoluteError,
    /// R-squared
    RSquared,
    /// Akaike Information Criterion
    AIC,
    /// Bayesian Information Criterion
    BIC,
    /// Cross-validation score
    CrossValidation,
}
